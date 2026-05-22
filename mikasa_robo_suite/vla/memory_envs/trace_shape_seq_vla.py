"""Trace-shape-sequence procedural memory tasks for the VLA benchmark."""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

from mikasa_robo_suite.vla.utils import shapes


class TraceShapeSeqVLABaseEnv(BaseEnv):
    """Watch a sequence of red traces, then reproduce all traces in order.

    The robot observes multiple demonstrations in sequence. For each element,
    the red cube traces one shape (circle / square / triangle depending on the
    difficulty variant). During the action phase, the robot must reproduce the
    same sequence with the green cube. After finishing all traces, the robot
    must press the submit button.

    Success (`success=True`):
    - Every sequence element must be completed in order.
    - A sequence element is complete only when all its checkpoints are visited
      and the contour is closed (return near checkpoint[0]).
    - After all elements are complete, the robot must press the button.
    """

    LANGUAGE_INSTRUCTION = (
        "Watch the red cube trace a sequence of shapes. When the lamp turns green, "
        "pick up the green cube and trace the same sequence in order. "
        "After finishing all shapes, press the button to submit your answer."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    HEIGHT_OFFSET = 1000.0

    SHAPE_CIRCLE = 0
    SHAPE_SQUARE = 1
    SHAPE_TRIANGLE = 2

    AVAILABLE_SHAPES: List[int] = [0]

    MIN_SEQUENCE_LENGTH = 2
    MAX_SEQUENCE_LENGTH = 5

    NUM_WAYPOINTS = 64
    NUM_CHECKPOINTS = 12
    CHECKPOINT_THRESH = 0.035

    PRE_DEMO_STEPS: List[int] = [3, 8]
    STEPS_PER_WAYPOINT = 1

    CUBE_HALFSIZE = 0.02
    SHAPE_RADIUS_RANGE = [0.078, 0.13]
    SHAPE_CENTER_X_RANGE = [-0.15, -0.05]
    SHAPE_CENTER_Y_RANGE = [-0.10, 0.10]
    GREEN_CUBE_OFFSET_X = -0.16

    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012
    LAMP_OFFSET_X = 0.25

    BUTTON_BASE_HALF_SIZE = np.array([0.065, 0.065, 0.015], dtype=np.float32)
    BUTTON_CAP_RADIUS = 0.03
    BUTTON_CAP_HALF_HEIGHT = 0.014
    BUTTON_CAP_TRAVEL = BUTTON_CAP_HALF_HEIGHT
    BUTTON_PRESS_EVENT_RATIO = 0.35
    BUTTON_RELEASE_READY_RATIO = 0.2
    BUTTON_PRESS_XY_RADIUS = 0.065
    BUTTON_PRESS_Z_MARGIN = 0.03
    BUTTON_OFFSET_FROM_LAMP_X = 0.02
    BUTTON_OFFSET_FROM_LAMP_Y = 0.16

    SUCCESS_BONUS = 40.0
    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.03
    QVEL_L2_COEF = 0.01

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        default_initial_pose = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALFSIZE,
            color=np.array([220, 50, 50, 255]) / 255.0,
            name="red_cube",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALFSIZE,
            color=np.array([50, 220, 50, 255]) / 255.0,
            name="green_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
        )

        lamp_kw = dict(
            body_type="kinematic",
            add_collision=False,
            initial_pose=default_initial_pose,
            base_radius=self.LAMP_BASE_RADIUS,
            base_half_height=self.LAMP_BASE_HALF_HEIGHT,
            stem_radius=self.LAMP_STEM_RADIUS,
            stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
            bulb_radius=self.LAMP_BULB_RADIUS,
        )
        lp_white = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="lamp_white",
            bulb_off_color=np.array([245, 245, 245, 255]) / 255.0,
            bulb_on_color=np.array([245, 245, 245, 255]) / 255.0,
            **lamp_kw,
        )
        lp_red = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="lamp_red",
            bulb_off_color=np.array([245, 245, 245, 255]) / 255.0,
            bulb_on_color=np.array([255, 0, 0, 255]) / 255.0,
            **lamp_kw,
        )
        lp_green = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="lamp_green",
            bulb_off_color=np.array([245, 245, 245, 255]) / 255.0,
            bulb_on_color=np.array([0, 255, 0, 255]) / 255.0,
            **lamp_kw,
        )

        self.lamp_body = lp_white["body"]
        self.lamp_white = lp_white["bulb_off"]
        self.lamp_red = lp_red["bulb_on"]
        self.lamp_green = lp_green["bulb_on"]

        shapes._set_actor_visual_rgba(
            self.lamp_red,
            np.array([255, 0, 0, 255]) / 255.0,
            emission_scale=20.0,
            remove_textures=True,
        )
        shapes._set_actor_visual_rgba(
            self.lamp_green,
            np.array([0, 255, 0, 255]) / 255.0,
            emission_scale=20.0,
            remove_textures=True,
        )

        self._lamp_aux = [
            lp_red["body"],
            lp_green["body"],
            lp_white["bulb_on"],
            lp_red["bulb_off"],
            lp_green["bulb_off"],
        ]

        base_builder = self.scene.create_actor_builder()
        base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
        base_builder.add_box_visual(
            half_size=self.BUTTON_BASE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=np.array([55, 64, 78, 255]) / 255.0),
        )
        self.button_base = _build_by_type(
            base_builder,
            name="trace_shape_seq_button_base",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        cap_builder = self.scene.create_actor_builder()
        cap_builder.add_cylinder_collision(radius=self.BUTTON_CAP_RADIUS, half_length=self.BUTTON_CAP_HALF_HEIGHT)
        cap_builder.add_cylinder_visual(
            radius=self.BUTTON_CAP_RADIUS,
            half_length=self.BUTTON_CAP_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(base_color=np.array([210, 80, 80, 255]) / 255.0),
        )
        self.button_cap = _build_by_type(
            cap_builder,
            name="trace_shape_seq_button_cap",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        n = self.num_envs
        d = self.device
        self.pre_demo_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.demo_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.cue_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)

        self.sequence_len = torch.zeros(n, dtype=torch.int64, device=d)
        self.shape_sequence = torch.full((n, self.MAX_SEQUENCE_LENGTH), -1, dtype=torch.int64, device=d)
        self.active_shape_idx = torch.zeros(n, dtype=torch.int64, device=d)

        self.waypoints = torch.zeros(
            n,
            self.MAX_SEQUENCE_LENGTH,
            self.NUM_WAYPOINTS,
            2,
            dtype=torch.float32,
            device=d,
        )
        self.checkpoints = torch.zeros(
            n,
            self.MAX_SEQUENCE_LENGTH,
            self.NUM_CHECKPOINTS,
            2,
            dtype=torch.float32,
            device=d,
        )
        self.checkpoint_visited = torch.zeros(
            n,
            self.MAX_SEQUENCE_LENGTH,
            self.NUM_CHECKPOINTS,
            dtype=torch.bool,
            device=d,
        )
        self.shape_closed = torch.zeros(n, self.MAX_SEQUENCE_LENGTH, dtype=torch.bool, device=d)

        self.shape_center_xy = torch.zeros(n, self.MAX_SEQUENCE_LENGTH, 2, dtype=torch.float32, device=d)

        self.lamp_on_pos = torch.zeros(n, 3, dtype=torch.float32, device=d)
        self.lamp_off_pos = torch.zeros(n, 3, dtype=torch.float32, device=d)

        self.button_xy = torch.zeros((n, 2), dtype=torch.float32, device=d)
        self.button_base_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_cap_unpressed_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_top_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_press_depth = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_pressed = torch.zeros(n, dtype=torch.bool, device=d)
        self.button_pressable = torch.zeros(n, dtype=torch.bool, device=d)
        self.press_ready = torch.ones(n, dtype=torch.bool, device=d)
        self.new_submit_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.submit_success_latched = torch.zeros(n, dtype=torch.bool, device=d)
        self.button_cap_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=d)

    def _generate_waypoints(self, shape_type, center_xy, radius, rotation, b):
        n = self.NUM_WAYPOINTS
        device = self.device
        waypoints = torch.zeros(b, n, 2, device=device)
        t = torch.linspace(0, 1.0, n + 1, device=device)[:-1]

        cm = shape_type == self.SHAPE_CIRCLE
        if cm.any():
            angles = t.unsqueeze(0) * 2 * np.pi + rotation[cm].unsqueeze(1)
            r = radius[cm].unsqueeze(1)
            waypoints[cm, :, 0] = center_xy[cm, 0:1] + r * torch.cos(angles)
            waypoints[cm, :, 1] = center_xy[cm, 1:2] + r * torch.sin(angles)

        sm = shape_type == self.SHAPE_SQUARE
        if sm.any():
            b_sq = sm.sum().item()
            s = radius[sm].unsqueeze(1)
            rot = rotation[sm]
            lx = torch.zeros(b_sq, n, device=device)
            ly = torch.zeros(b_sq, n, device=device)
            for side in range(4):
                lo, hi = side * 0.25, (side + 1) * 0.25
                mask = (t >= lo) & (t < hi)
                nm = mask.sum().item()
                frac = (t[mask] - lo) / 0.25
                if side == 0:
                    lx[:, mask] = -s.expand(-1, nm) + 2 * s * frac.unsqueeze(0)
                    ly[:, mask] = (-s).expand(-1, nm)
                elif side == 1:
                    lx[:, mask] = s.expand(-1, nm)
                    ly[:, mask] = -s.expand(-1, nm) + 2 * s * frac.unsqueeze(0)
                elif side == 2:
                    lx[:, mask] = s.expand(-1, nm) - 2 * s * frac.unsqueeze(0)
                    ly[:, mask] = s.expand(-1, nm)
                else:
                    lx[:, mask] = (-s).expand(-1, nm)
                    ly[:, mask] = s.expand(-1, nm) - 2 * s * frac.unsqueeze(0)
            cos_r = torch.cos(rot).unsqueeze(1)
            sin_r = torch.sin(rot).unsqueeze(1)
            waypoints[sm, :, 0] = center_xy[sm, 0:1] + lx * cos_r - ly * sin_r
            waypoints[sm, :, 1] = center_xy[sm, 1:2] + lx * sin_r + ly * cos_r

        tm = shape_type == self.SHAPE_TRIANGLE
        if tm.any():
            b_tr = tm.sum().item()
            r = radius[tm].unsqueeze(1)
            rot = rotation[tm]
            v_angles = torch.tensor([0, 2 * np.pi / 3, 4 * np.pi / 3], device=device)
            lx = torch.zeros(b_tr, n, device=device)
            ly = torch.zeros(b_tr, n, device=device)
            for side in range(3):
                lo = side / 3.0
                hi = (side + 1) / 3.0
                mask = (t >= lo) & (t < hi)
                frac = (t[mask] - lo) * 3.0
                a0, a1 = v_angles[side], v_angles[(side + 1) % 3]
                x0 = r * float(np.cos(a0.item()))
                y0 = r * float(np.sin(a0.item()))
                x1 = r * float(np.cos(a1.item()))
                y1 = r * float(np.sin(a1.item()))
                lx[:, mask] = x0 + (x1 - x0) * frac.unsqueeze(0)
                ly[:, mask] = y0 + (y1 - y0) * frac.unsqueeze(0)
            cos_r = torch.cos(rot).unsqueeze(1)
            sin_r = torch.sin(rot).unsqueeze(1)
            waypoints[tm, :, 0] = center_xy[tm, 0:1] + lx * cos_r - ly * sin_r
            waypoints[tm, :, 1] = center_xy[tm, 1:2] + lx * sin_r + ly * cos_r

        return waypoints

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            env_idx = env_idx.to(self.device)

            self.reward_dict = None

            shape_choices = torch.tensor(self.AVAILABLE_SHAPES, device=self.device, dtype=torch.int64)
            seq_len = torch.randint(
                self.MIN_SEQUENCE_LENGTH,
                self.MAX_SEQUENCE_LENGTH + 1,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )
            shape_seq = torch.full((b, self.MAX_SEQUENCE_LENGTH), -1, dtype=torch.int64, device=self.device)
            for i in range(b):
                l = int(seq_len[i].item())
                pick_idx = torch.randint(0, len(self.AVAILABLE_SHAPES), (l,), device=self.device)
                shape_seq[i, :l] = shape_choices[pick_idx]

            self.sequence_len[env_idx] = seq_len
            self.shape_sequence[env_idx] = shape_seq
            self.active_shape_idx[env_idx] = 0

            center_x = (
                torch.rand(b, device=self.device) * (self.SHAPE_CENTER_X_RANGE[1] - self.SHAPE_CENTER_X_RANGE[0])
                + self.SHAPE_CENTER_X_RANGE[0]
            )
            center_y = (
                torch.rand(b, device=self.device) * (self.SHAPE_CENTER_Y_RANGE[1] - self.SHAPE_CENTER_Y_RANGE[0])
                + self.SHAPE_CENTER_Y_RANGE[0]
            )
            center_xy = torch.stack([center_x, center_y], dim=-1).unsqueeze(1).repeat(1, self.MAX_SEQUENCE_LENGTH, 1)
            self.shape_center_xy[env_idx] = center_xy

            radius = (
                torch.rand(b, self.MAX_SEQUENCE_LENGTH, device=self.device)
                * (self.SHAPE_RADIUS_RANGE[1] - self.SHAPE_RADIUS_RANGE[0])
                + self.SHAPE_RADIUS_RANGE[0]
            )
            rotation = torch.rand(b, self.MAX_SEQUENCE_LENGTH, device=self.device) * 2 * np.pi

            all_waypoints = torch.zeros(
                b,
                self.MAX_SEQUENCE_LENGTH,
                self.NUM_WAYPOINTS,
                2,
                dtype=torch.float32,
                device=self.device,
            )
            for s_idx in range(self.MAX_SEQUENCE_LENGTH):
                shape_for_gen = shape_seq[:, s_idx].clone()
                invalid = shape_for_gen < 0
                shape_for_gen[invalid] = self.SHAPE_CIRCLE
                waypoints_s = self._generate_waypoints(
                    shape_for_gen,
                    center_xy[:, s_idx],
                    radius[:, s_idx],
                    rotation[:, s_idx],
                    b,
                )
                if invalid.any():
                    center_repeat = center_xy[:, s_idx][invalid].unsqueeze(1).repeat(1, self.NUM_WAYPOINTS, 1)
                    waypoints_s[invalid] = center_repeat
                all_waypoints[:, s_idx] = waypoints_s

            self.waypoints[env_idx] = all_waypoints

            step = max(1, self.NUM_WAYPOINTS // self.NUM_CHECKPOINTS)
            cp_idx = torch.arange(0, self.NUM_WAYPOINTS, step, device=self.device)[: self.NUM_CHECKPOINTS]
            self.checkpoints[env_idx] = all_waypoints[:, :, cp_idx]
            self.checkpoint_visited[env_idx] = False
            self.shape_closed[env_idx] = False

            pre_demo = torch.randint(
                self.PRE_DEMO_STEPS[0],
                self.PRE_DEMO_STEPS[1] + 1,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )
            per_shape_demo_steps = self.NUM_WAYPOINTS * self.STEPS_PER_WAYPOINT
            demo_steps = seq_len * per_shape_demo_steps
            self.pre_demo_steps_per_env[env_idx] = pre_demo
            self.demo_steps_per_env[env_idx] = demo_steps
            self.cue_steps_per_env[env_idx] = pre_demo + demo_steps

            red_xyz = torch.zeros(b, 3, device=self.device)
            red_xyz[:, :2] = all_waypoints[:, 0, 0]
            red_xyz[:, 2] = self.CUBE_HALFSIZE
            self.red_cube.set_pose(Pose.create_from_pq(p=red_xyz, q=[1, 0, 0, 0]))

            green_xyz = torch.zeros(b, 3, device=self.device)
            green_xyz[:, 0] = center_x + self.GREEN_CUBE_OFFSET_X
            green_xyz[:, 1] = center_y
            green_xyz[:, 2] = self.CUBE_HALFSIZE
            self.green_cube.set_pose(Pose.create_from_pq(p=green_xyz, q=[1, 0, 0, 0]))

            lamp_pos = torch.zeros(b, 3, device=self.device)
            lamp_pos[:, 0] = center_x + self.LAMP_OFFSET_X
            lamp_pos[:, 1] = center_y
            lamp_pos[:, 2] = 0.0
            lamp_off = lamp_pos.clone()
            lamp_off[:, 2] += self.HEIGHT_OFFSET
            self.lamp_on_pos[env_idx] = lamp_pos
            self.lamp_off_pos[env_idx] = lamp_off

            lamp_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.lamp_body.set_pose(Pose.create_from_pq(p=lamp_pos, q=lamp_q))
            self.lamp_white.set_pose(Pose.create_from_pq(p=lamp_pos, q=lamp_q))
            self.lamp_red.set_pose(Pose.create_from_pq(p=lamp_off, q=lamp_q))
            self.lamp_green.set_pose(Pose.create_from_pq(p=lamp_off, q=lamp_q))
            for aux in self._lamp_aux:
                aux.set_pose(Pose.create_from_pq(p=lamp_off, q=lamp_q))

            button_xy = torch.zeros((b, 2), device=self.device)
            button_xy[:, 0] = lamp_pos[:, 0] + self.BUTTON_OFFSET_FROM_LAMP_X
            button_xy[:, 1] = lamp_pos[:, 1] + self.BUTTON_OFFSET_FROM_LAMP_Y
            base_z = torch.full((b,), float(self.BUTTON_BASE_HALF_SIZE[2]), device=self.device)
            cap_unpressed_z = torch.full(
                (b,),
                float(self.BUTTON_BASE_HALF_SIZE[2]) * 2.0 + self.BUTTON_CAP_HALF_HEIGHT,
                device=self.device,
            )

            button_base_xyz = torch.zeros((b, 3), device=self.device)
            button_base_xyz[:, :2] = button_xy
            button_base_xyz[:, 2] = base_z
            button_cap_xyz = torch.zeros((b, 3), device=self.device)
            button_cap_xyz[:, :2] = button_xy
            button_cap_xyz[:, 2] = cap_unpressed_z

            button_base_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            button_cap_q = self.button_cap_quat.unsqueeze(0).repeat(b, 1)

            self.button_xy[env_idx] = button_xy
            self.button_base_z[env_idx] = base_z
            self.button_cap_unpressed_z[env_idx] = cap_unpressed_z
            self.button_top_z[env_idx] = cap_unpressed_z + self.BUTTON_CAP_HALF_HEIGHT
            self.button_press_depth[env_idx] = 0.0
            self.button_pressed[env_idx] = False
            self.button_pressable[env_idx] = False
            self.press_ready[env_idx] = True
            self.new_submit_event[env_idx] = False
            self.submit_success_latched[env_idx] = False

            self.button_base.set_pose(Pose.create_from_pq(p=button_base_xyz, q=button_base_q))
            self.button_cap.set_pose(Pose.create_from_pq(p=button_cap_xyz, q=button_cap_q))

            self.oracle_info = self.shape_sequence.to(torch.int64)
            self.task_cue = self.shape_sequence.to(torch.int64)

            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

            if hasattr(self, "_prev_action") and torch.is_tensor(self._prev_action):
                if self._prev_action.shape[0] >= int(env_idx.max().item()) + 1:
                    self._prev_action[env_idx] = 0

    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        pre_demo_mask = elapsed < self.pre_demo_steps_per_env
        demo_mask = (~pre_demo_mask) & (elapsed < self.cue_steps_per_env)
        action_mask = elapsed >= self.cue_steps_per_env

        for lamp_actor, on_mask in [
            (self.lamp_white, pre_demo_mask),
            (self.lamp_red, demo_mask),
            (self.lamp_green, action_mask),
        ]:
            pose = lamp_actor.pose.raw_pose.clone()
            pose[on_mask, :3] = self.lamp_on_pos[on_mask]
            pose[~on_mask, :3] = self.lamp_off_pos[~on_mask]
            lamp_actor.pose = pose

        red_pose = self.red_cube.pose.raw_pose.clone()

        red_pose[pre_demo_mask, 0] = self.waypoints[pre_demo_mask, 0, 0, 0]
        red_pose[pre_demo_mask, 1] = self.waypoints[pre_demo_mask, 0, 0, 1]
        red_pose[pre_demo_mask, 2] = self.CUBE_HALFSIZE

        if demo_mask.any():
            demo_elapsed = (elapsed[demo_mask] - self.pre_demo_steps_per_env[demo_mask]).clamp(min=0)
            per_shape_demo_steps = self.NUM_WAYPOINTS * self.STEPS_PER_WAYPOINT
            demo_shape_idx = demo_elapsed // per_shape_demo_steps
            max_demo_shape_idx = torch.clamp(self.sequence_len[demo_mask] - 1, min=0)
            demo_shape_idx = torch.minimum(demo_shape_idx, max_demo_shape_idx)
            wp_idx = ((demo_elapsed % per_shape_demo_steps) // self.STEPS_PER_WAYPOINT).clamp(
                max=self.NUM_WAYPOINTS - 1
            )

            batch_idx = torch.arange(self.waypoints.shape[0], device=self.device)[demo_mask]
            red_xy = self.waypoints[batch_idx, demo_shape_idx, wp_idx]
            red_pose[demo_mask, 0] = red_xy[:, 0]
            red_pose[demo_mask, 1] = red_xy[:, 1]
            red_pose[demo_mask, 2] = self.CUBE_HALFSIZE

        red_pose[action_mask, 2] = self.CUBE_HALFSIZE + self.HEIGHT_OFFSET
        self.red_cube.pose = red_pose

        green_xy = self.green_cube.pose.p[:, :2]

        if bool(action_mask.any().item()):
            action_env_ids = torch.where(action_mask)[0]
            for env_id_t in action_env_ids:
                env_id = int(env_id_t.item())
                seq_len_i = int(self.sequence_len[env_id].item())
                if seq_len_i <= 0:
                    continue

                active_i = int(self.active_shape_idx[env_id].item())
                if active_i < 0:
                    active_i = 0
                if active_i >= seq_len_i:
                    continue

                cp = self.checkpoints[env_id, active_i]
                dist = torch.linalg.norm(green_xy[env_id].unsqueeze(0) - cp, dim=-1)
                visited = self.checkpoint_visited[env_id, active_i] | (dist < self.CHECKPOINT_THRESH)
                self.checkpoint_visited[env_id, active_i] = visited

                all_visited_i = bool(visited.all().item())
                start_dist_i = float(dist[0].item())
                if all_visited_i and (start_dist_i < self.CHECKPOINT_THRESH):
                    self.shape_closed[env_id, active_i] = True
                    if active_i + 1 < seq_len_i:
                        self.active_shape_idx[env_id] = active_i + 1

        shape_range = torch.arange(self.MAX_SEQUENCE_LENGTH, device=self.device).unsqueeze(0)
        valid_shape_mask = shape_range < self.sequence_len.unsqueeze(1)
        all_shapes_closed = torch.where(valid_shape_mask, self.shape_closed, torch.ones_like(self.shape_closed)).all(
            dim=1
        )

        closed_count = (self.shape_closed & valid_shape_mask).sum(dim=1).float()
        sequence_progress = closed_count / torch.clamp(self.sequence_len.float(), min=1.0)

        batch = torch.arange(self.num_envs, device=self.device)
        safe_active_idx = torch.clamp(self.active_shape_idx, min=0, max=self.MAX_SEQUENCE_LENGTH - 1)
        active_checkpoints = self.checkpoints[batch, safe_active_idx]
        active_visited = self.checkpoint_visited[batch, safe_active_idx]
        active_visit_fraction = active_visited.float().mean(dim=1)
        active_all_visited = active_visited.all(dim=1)

        start_checkpoint_dist = torch.linalg.norm(green_xy - active_checkpoints[:, 0], dim=-1)
        is_active_contour_closed = active_all_visited & (start_checkpoint_dist < self.CHECKPOINT_THRESH)

        tcp_pos = self.agent.tcp.pose.p
        tcp_xy = tcp_pos[:, :2]
        tcp_z = tcp_pos[:, 2]

        self.button_pressable = all_shapes_closed & action_mask
        xy_dist_to_button = torch.linalg.norm(tcp_xy - self.button_xy, dim=1)

        raw_depth = self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
        depth = torch.clamp(raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
        depth = depth * (xy_dist_to_button < self.BUTTON_PRESS_XY_RADIUS).float()
        depth = depth * action_mask.float()
        self.button_press_depth = depth

        cap_pose = self.button_cap.pose.raw_pose.clone()
        cap_pose[:, 0:2] = self.button_xy
        cap_pose[:, 2] = self.button_cap_unpressed_z - depth
        cap_pose[:, 3:7] = self.button_cap_quat.repeat(cap_pose.shape[0], 1)
        self.button_cap.pose = cap_pose

        base_pose = self.button_base.pose.raw_pose.clone()
        base_pose[:, 0:2] = self.button_xy
        base_pose[:, 2] = self.button_base_z
        base_pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(base_pose.shape[0], 1)
        self.button_base.pose = base_pose

        pressed = depth >= (self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO)
        released = depth <= (self.BUTTON_CAP_TRAVEL * self.BUTTON_RELEASE_READY_RATIO)
        self.press_ready = self.press_ready | (released & action_mask)
        self.new_submit_event = pressed & self.press_ready & action_mask
        self.press_ready = self.press_ready & (~self.new_submit_event)
        self.button_pressed = self.button_pressed | self.new_submit_event

        submitted = self.new_submit_event
        successful_submit_event = submitted & all_shapes_closed
        self.submit_success_latched = self.submit_success_latched | successful_submit_event
        success = self.submit_success_latched
        failed_submit = submitted & (~all_shapes_closed)

        dist_to_active_cp = torch.linalg.norm(green_xy.unsqueeze(1) - active_checkpoints, dim=-1)
        dist_to_active_cp = dist_to_active_cp + active_visited.float() * 1000.0
        nearest_idx = dist_to_active_cp.min(dim=1).indices
        nearest_cp_xy = active_checkpoints[batch, nearest_idx]

        target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        target_pos[:, :2] = nearest_cp_xy
        target_pos[:, 2] = self.CUBE_HALFSIZE

        button_target_mask = all_shapes_closed & (~self.button_pressed) & action_mask
        target_pos[button_target_mask, :2] = self.button_xy[button_target_mask]
        target_pos[button_target_mask, 2] = self.button_top_z[button_target_mask] + 0.005

        self.obj_to_goal_pos = target_pos - tcp_pos

        active_shape_id = self.shape_sequence[batch, safe_active_idx]

        return {
            "success": success,
            "failed_submit": failed_submit,
            "submitted": submitted,
            "successful_submit_event": successful_submit_event,
            "action_mask": action_mask,
            "demo_mask": demo_mask,
            "pre_demo_mask": pre_demo_mask,
            "sequence_progress": sequence_progress,
            "all_shapes_closed": all_shapes_closed,
            "active_shape_idx": safe_active_idx,
            "active_shape_id": active_shape_id,
            "active_visit_fraction": active_visit_fraction,
            "active_all_visited": active_all_visited,
            "is_active_contour_closed": is_active_contour_closed,
            "start_checkpoint_dist": start_checkpoint_dist,
            "button_pressed": self.button_pressed,
            "button_pressable": self.button_pressable,
            "button_press_depth": self.button_press_depth,
            "xy_dist_to_button": xy_dist_to_button,
            "submit_success_latched": self.submit_success_latched,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                action_mask=info["action_mask"],
                sequence_progress=info["sequence_progress"],
                active_shape_idx=info["active_shape_idx"],
                active_shape_id=info["active_shape_id"],
                sequence_len=self.sequence_len,
                button_xy=self.button_xy,
                button_pressed=self.button_pressed,
                oracle_info=self.oracle_info,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict):
            submitted = info.get("submitted", None)
            success = info.get("success", None)
            if torch.is_tensor(terminated):
                term_bool = terminated.to(dtype=torch.bool)
                if torch.is_tensor(submitted):
                    term_bool = term_bool | submitted.to(dtype=torch.bool)
                elif submitted is not None:
                    term_bool = term_bool | bool(submitted)
                if torch.is_tensor(success):
                    term_bool = term_bool | success.to(dtype=torch.bool)
                elif success is not None:
                    term_bool = term_bool | bool(success)
                terminated = term_bool
            else:
                terminated = bool(terminated) or bool(submitted) or bool(success)
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        green_pos = self.green_cube.pose.p

        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - green_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)
        is_grasping = (tcp_to_cube_dist < 0.05).float()

        active_visit_fraction = info["active_visit_fraction"]
        sequence_progress = info["sequence_progress"]
        active_all_visited = info["active_all_visited"].float()

        start_checkpoint_dist = info["start_checkpoint_dist"]
        closure_reward = 1 - torch.tanh(5.0 * start_checkpoint_dist)

        button_target_mask = info["all_shapes_closed"] & (~info["button_pressed"]) & info["action_mask"]
        button_target_pos = torch.zeros_like(tcp_pos)
        button_target_pos[:, :2] = self.button_xy
        button_target_pos[:, 2] = self.button_top_z + 0.005
        tcp_to_button_dist = torch.linalg.norm(button_target_pos - tcp_pos, dim=-1)
        button_reach_reward = 1 - torch.tanh(8.0 * tcp_to_button_dist)
        button_press_reward = torch.clamp(info["button_press_depth"] / self.BUTTON_CAP_TRAVEL, min=0.0, max=1.0)

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)

        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, dim=-1)
        delta_action_l2 = torch.linalg.norm(delta_action, dim=-1)
        qvel_l2 = torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], dim=-1)
        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        act_f = info["action_mask"].float()
        cue_f = 1.0 - act_f
        button_target_f = button_target_mask.float()

        reward = (
            0.5 * cue_f * reaching_reward
            + 1.0 * act_f * reaching_reward
            + 1.8 * act_f * is_grasping
            + 2.5 * act_f * active_visit_fraction
            + 3.0 * act_f * sequence_progress
            + 2.0 * act_f * active_all_visited * closure_reward
            + 1.5 * button_target_f * button_reach_reward
            + 2.5 * button_target_f * button_press_reward
            - smooth_penalty
        )

        reward = torch.where(
            info["failed_submit"],
            torch.full_like(reward, -2.0),
            reward,
        )
        reward[info["success"]] = self.SUCCESS_BONUS

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "is_grasping": is_grasping,
            "active_visit_fraction": active_visit_fraction,
            "sequence_progress": sequence_progress,
            "closure_reward": closure_reward,
            "button_reach_reward": button_reach_reward,
            "button_press_reward": button_press_reward,
            "smooth_penalty": smooth_penalty,
            "start_checkpoint_dist": start_checkpoint_dist,
        }
        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


@register_env("TraceShapeSeqEasy-VLA-v0", max_episode_steps=1500)
class TraceShapeSeqEasyVLAEnv(TraceShapeSeqVLABaseEnv):
    """Sequence with circles only."""

    AVAILABLE_SHAPES: List[int] = [0]


@register_env("TraceShapeSeqMedium-VLA-v0", max_episode_steps=1500)
class TraceShapeSeqMediumVLAEnv(TraceShapeSeqVLABaseEnv):
    """Sequence with circles and squares."""

    AVAILABLE_SHAPES: List[int] = [0, 1]


@register_env("TraceShapeSeqHard-VLA-v0", max_episode_steps=1500)
class TraceShapeSeqHardVLAEnv(TraceShapeSeqVLABaseEnv):
    """Sequence with circles, squares, and triangles."""

    AVAILABLE_SHAPES: List[int] = [0, 1, 2]
