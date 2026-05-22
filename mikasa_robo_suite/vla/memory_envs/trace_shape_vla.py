"""Trace-shape procedural memory task for the VLA benchmark."""

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
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

from mikasa_robo_suite.vla.utils import shapes


class TraceShapeVLABaseEnv(BaseEnv):
    """Watch a red cube trace a shape, then reproduce it with a green cube.

    The robot observes a demonstration in which a red cube traces a geometric
    contour (circle, square, or triangle) on the table while a lamp glows red.
    Once the demonstration ends the lamp turns green, the red cube disappears,
    and the robot must pick up the nearby green cube and replicate the same
    contour.

    Episode flow:
    - Pre-demo: white lamp, both cubes visible on the table, nothing moves.
    - Demo: lamp turns red, the red cube traces the target shape.
    - Action: lamp turns green, red cube hidden, robot traces with green cube.

    Success (`success=True`):
    - The green cube must visit every checkpoint along the demonstrated path.
    - After that, the contour must be explicitly closed by returning near the
      first checkpoint (start point) within `CHECKPOINT_THRESH`.

    How to customize:
    - ``AVAILABLE_SHAPES`` controls which shapes can appear (difficulty).
    - ``NUM_WAYPOINTS`` controls the path resolution of the demonstration.
    - ``NUM_CHECKPOINTS`` controls how many points are checked for success.
    - ``CHECKPOINT_THRESH`` controls the required tracing accuracy.
    - ``SHAPE_RADIUS_RANGE`` controls shape size randomisation.
    """

    LANGUAGE_INSTRUCTION = (
        "Watch the red cube trace a shape on the table. When the lamp turns green, "
        "pick up the green cube and trace exactly the same shape."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    HEIGHT_OFFSET = 1000.0

    # Shape IDs
    SHAPE_CIRCLE = 0
    SHAPE_SQUARE = 1
    SHAPE_TRIANGLE = 2

    AVAILABLE_SHAPES: List[int] = [0]  # overridden by subclasses

    NUM_WAYPOINTS = 64
    NUM_CHECKPOINTS = 12
    CHECKPOINT_THRESH = 0.035

    # Timing
    PRE_DEMO_STEPS: List[int] = [3, 8]
    STEPS_PER_WAYPOINT = 1

    # Geometry
    CUBE_HALFSIZE = 0.02
    SHAPE_RADIUS_RANGE = [0.078, 0.13]
    SHAPE_CENTER_X_RANGE = [-0.15, -0.05]
    SHAPE_CENTER_Y_RANGE = [-0.10, 0.10]
    GREEN_CUBE_OFFSET_X = -0.16

    # Lamp
    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012
    LAMP_OFFSET_X = 0.25

    # Reward
    SUCCESS_BONUS = 30.0
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

    # ------------------------------------------------------------------
    # Simulation / camera config
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()
        default_initial_pose = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])

        # Red cube – kinematic (driven by the environment during demo)
        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALFSIZE,
            color=np.array([220, 50, 50, 255]) / 255.0,
            name="red_cube",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        # Green cube – dynamic (manipulated by the robot)
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALFSIZE,
            color=np.array([50, 220, 50, 255]) / 255.0,
            name="green_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
        )

        # ---- Lamp (white / red / green bulbs sharing one body) ----
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

        # Boost emission so the colour is clearly visible
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

        # Hide auxiliary actors that are not used visually
        self._lamp_aux = [
            lp_red["body"],
            lp_green["body"],
            lp_white["bulb_on"],
            lp_red["bulb_off"],
            lp_green["bulb_off"],
        ]

        # ---- Per-env buffers ----
        n = self.num_envs
        d = self.device
        self.pre_demo_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.demo_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.cue_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.shape_type = torch.zeros(n, dtype=torch.int64, device=d)
        self.waypoints = torch.zeros(n, self.NUM_WAYPOINTS, 2, dtype=torch.float32, device=d)
        self.checkpoints = torch.zeros(n, self.NUM_CHECKPOINTS, 2, dtype=torch.float32, device=d)
        self.checkpoint_visited = torch.zeros(n, self.NUM_CHECKPOINTS, dtype=torch.bool, device=d)
        self.lamp_on_pos = torch.zeros(n, 3, dtype=torch.float32, device=d)
        self.lamp_off_pos = torch.zeros(n, 3, dtype=torch.float32, device=d)

    # ------------------------------------------------------------------
    # Waypoint generation
    # ------------------------------------------------------------------
    def _generate_waypoints(self, shape_type, center_xy, radius, rotation, b):
        """Return (b, NUM_WAYPOINTS, 2) XY path for each env."""
        n = self.NUM_WAYPOINTS
        device = self.device
        waypoints = torch.zeros(b, n, 2, device=device)
        t = torch.linspace(0, 1.0, n + 1, device=device)[:-1]  # [0, 1)

        # ---- Circle ----
        cm = shape_type == self.SHAPE_CIRCLE
        if cm.any():
            angles = t.unsqueeze(0) * 2 * np.pi + rotation[cm].unsqueeze(1)
            r = radius[cm].unsqueeze(1)
            waypoints[cm, :, 0] = center_xy[cm, 0:1] + r * torch.cos(angles)
            waypoints[cm, :, 1] = center_xy[cm, 1:2] + r * torch.sin(angles)

        # ---- Square ----
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

        # ---- Triangle (equilateral) ----
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
                x0 = r * float(np.cos(a0.item())) if isinstance(a0, torch.Tensor) else r * np.cos(a0)
                y0 = r * float(np.sin(a0.item())) if isinstance(a0, torch.Tensor) else r * np.sin(a0)
                x1 = r * float(np.cos(a1.item())) if isinstance(a1, torch.Tensor) else r * np.cos(a1)
                y1 = r * float(np.sin(a1.item())) if isinstance(a1, torch.Tensor) else r * np.sin(a1)
                lx[:, mask] = x0 + (x1 - x0) * frac.unsqueeze(0)
                ly[:, mask] = y0 + (y1 - y0) * frac.unsqueeze(0)
            cos_r = torch.cos(rot).unsqueeze(1)
            sin_r = torch.sin(rot).unsqueeze(1)
            waypoints[tm, :, 0] = center_xy[tm, 0:1] + lx * cos_r - ly * sin_r
            waypoints[tm, :, 1] = center_xy[tm, 1:2] + lx * sin_r + ly * cos_r

        return waypoints

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            env_idx = env_idx.to(self.device)

            self.task_cue = None
            self.reward_dict = None

            # ---- Sample shape type ----
            shape_choices = torch.tensor(
                self.AVAILABLE_SHAPES,
                device=self.device,
                dtype=torch.int64,
            )
            choice_idx = torch.randint(0, len(self.AVAILABLE_SHAPES), (b,), device=self.device)
            shape_type = shape_choices[choice_idx]
            self.shape_type[env_idx] = shape_type

            # ---- Sample shape geometry ----
            rng = self.SHAPE_RADIUS_RANGE
            radius = torch.rand(b, device=self.device) * (rng[1] - rng[0]) + rng[0]
            cx_rng = self.SHAPE_CENTER_X_RANGE
            cy_rng = self.SHAPE_CENTER_Y_RANGE
            center_x = torch.rand(b, device=self.device) * (cx_rng[1] - cx_rng[0]) + cx_rng[0]
            center_y = torch.rand(b, device=self.device) * (cy_rng[1] - cy_rng[0]) + cy_rng[0]
            center_xy = torch.stack([center_x, center_y], dim=-1)
            rotation = torch.rand(b, device=self.device) * 2 * np.pi

            # ---- Waypoints & checkpoints ----
            waypoints = self._generate_waypoints(shape_type, center_xy, radius, rotation, b)
            self.waypoints[env_idx] = waypoints

            step = max(1, self.NUM_WAYPOINTS // self.NUM_CHECKPOINTS)
            cp_idx = torch.arange(0, self.NUM_WAYPOINTS, step, device=self.device)[: self.NUM_CHECKPOINTS]
            self.checkpoints[env_idx] = waypoints[:, cp_idx]
            self.checkpoint_visited[env_idx] = False

            # ---- Timing ----
            pre_demo = torch.randint(
                self.PRE_DEMO_STEPS[0],
                self.PRE_DEMO_STEPS[1] + 1,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )
            demo_steps = torch.full(
                (b,),
                self.NUM_WAYPOINTS * self.STEPS_PER_WAYPOINT,
                device=self.device,
                dtype=torch.int64,
            )
            self.pre_demo_steps_per_env[env_idx] = pre_demo
            self.demo_steps_per_env[env_idx] = demo_steps
            self.cue_steps_per_env[env_idx] = pre_demo + demo_steps

            # ---- Place red cube at first waypoint ----
            red_xyz = torch.zeros(b, 3, device=self.device)
            red_xyz[:, :2] = waypoints[:, 0]
            red_xyz[:, 2] = self.CUBE_HALFSIZE
            self.red_cube.set_pose(
                Pose.create_from_pq(p=red_xyz, q=[1, 0, 0, 0]),
            )

            # ---- Place green cube near the shape ----
            green_xyz = torch.zeros(b, 3, device=self.device)
            green_xyz[:, 0] = center_x + self.GREEN_CUBE_OFFSET_X
            green_xyz[:, 1] = center_y
            green_xyz[:, 2] = self.CUBE_HALFSIZE
            self.green_cube.set_pose(
                Pose.create_from_pq(p=green_xyz, q=[1, 0, 0, 0]),
            )

            # ---- Place lamp ----
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

            # ---- Oracle / task_cue ----
            self.oracle_info = self.shape_type[env_idx].to(torch.uint8)

            # ---- Reset robot ----
            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(
                    0,
                    self.robot_init_qpos_noise,
                    len(qpos) - 2,
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

            if hasattr(self, "_prev_action") and torch.is_tensor(self._prev_action):
                if self._prev_action.shape[0] >= int(env_idx.max().item()) + 1:
                    self._prev_action[env_idx] = 0

    # ------------------------------------------------------------------
    # Evaluate (runs every step)
    # ------------------------------------------------------------------
    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        pre_demo_mask = elapsed < self.pre_demo_steps_per_env
        demo_mask = (~pre_demo_mask) & (elapsed < self.cue_steps_per_env)
        action_mask = elapsed >= self.cue_steps_per_env

        # ---- Lamp switching ----
        for lamp_actor, on_mask in [
            (self.lamp_white, pre_demo_mask),
            (self.lamp_red, demo_mask),
            (self.lamp_green, action_mask),
        ]:
            pose = lamp_actor.pose.raw_pose.clone()
            pose[on_mask, :3] = self.lamp_on_pos[on_mask]
            pose[~on_mask, :3] = self.lamp_off_pos[~on_mask]
            lamp_actor.pose = pose

        # ---- Red cube animation ----
        red_pose = self.red_cube.pose.raw_pose.clone()

        # Pre-demo: sit at first waypoint
        red_pose[pre_demo_mask, 0] = self.waypoints[pre_demo_mask, 0, 0]
        red_pose[pre_demo_mask, 1] = self.waypoints[pre_demo_mask, 0, 1]
        red_pose[pre_demo_mask, 2] = self.CUBE_HALFSIZE

        # Demo: follow waypoints
        if demo_mask.any():
            demo_elapsed = (elapsed[demo_mask] - self.pre_demo_steps_per_env[demo_mask]).clamp(min=0)
            wp_idx = (demo_elapsed // self.STEPS_PER_WAYPOINT).clamp(
                max=self.NUM_WAYPOINTS - 1,
            )
            batch_idx = torch.arange(self.waypoints.shape[0], device=self.device)[demo_mask]
            red_xy = self.waypoints[batch_idx, wp_idx]
            red_pose[demo_mask, 0] = red_xy[:, 0]
            red_pose[demo_mask, 1] = red_xy[:, 1]
            red_pose[demo_mask, 2] = self.CUBE_HALFSIZE

        # Action: hide red cube
        red_pose[action_mask, 2] = self.CUBE_HALFSIZE + self.HEIGHT_OFFSET

        self.red_cube.pose = red_pose

        # ---- Checkpoint tracking (action phase only) ----
        green_xy = self.green_cube.pose.p[:, :2]
        dist = torch.linalg.norm(
            green_xy.unsqueeze(1) - self.checkpoints,
            dim=-1,
        )
        newly_visited = (dist < self.CHECKPOINT_THRESH) & action_mask.unsqueeze(1)
        self.checkpoint_visited = self.checkpoint_visited | newly_visited

        all_visited = self.checkpoint_visited.all(dim=1)
        start_checkpoint = self.checkpoints[:, 0]
        start_checkpoint_dist = torch.linalg.norm(green_xy - start_checkpoint, dim=-1)
        is_contour_closed = all_visited & (start_checkpoint_dist < self.CHECKPOINT_THRESH)
        success = is_contour_closed & action_mask
        visit_fraction = self.checkpoint_visited.float().mean(dim=1)

        self.obj_to_goal_pos = self.green_cube.pose.p - self.agent.tcp.pose.p

        return {
            "success": success,
            "action_mask": action_mask,
            "demo_mask": demo_mask,
            "pre_demo_mask": pre_demo_mask,
            "visit_fraction": visit_fraction,
            "all_visited": all_visited,
            "is_contour_closed": is_contour_closed,
            "start_checkpoint_dist": start_checkpoint_dist,
            "checkpoint_visited": self.checkpoint_visited,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
                action_mask=info["action_mask"],
                visit_fraction=info["visit_fraction"],
                oracle_info=self.oracle_info,
            )
        return obs

    # ------------------------------------------------------------------
    # Step override – terminate on success
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict):
            success = info.get("success", None)
            if torch.is_tensor(terminated) and torch.is_tensor(success):
                terminated = terminated.to(dtype=torch.bool) | success.to(dtype=torch.bool)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        green_pos = self.green_cube.pose.p

        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - green_pos, dim=-1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)
        is_grasping = (tcp_to_cube_dist < 0.05).float()

        visit_fraction = info["visit_fraction"]
        all_visited = info["all_visited"].float()

        # Nearest unvisited checkpoint reward
        green_xy = green_pos[:, :2]
        dist_to_cp = torch.linalg.norm(
            green_xy.unsqueeze(1) - self.checkpoints,
            dim=-1,
        )
        dist_to_cp = dist_to_cp + self.checkpoint_visited.float() * 1000.0
        nearest_dist = dist_to_cp.min(dim=1).values
        nearest_cp_reward = 1 - torch.tanh(5.0 * nearest_dist)

        start_checkpoint_dist = info["start_checkpoint_dist"]
        closure_reward = 1 - torch.tanh(5.0 * start_checkpoint_dist)

        # Smoothness penalties
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

        reward = (
            0.5 * cue_f * reaching_reward
            + 1.0 * act_f * reaching_reward
            + 2.0 * act_f * is_grasping
            + 3.0 * act_f * visit_fraction
            + 2.0 * act_f * is_grasping * nearest_cp_reward
            + 2.0 * act_f * all_visited * closure_reward
            - smooth_penalty
        )
        reward[info["success"]] = self.SUCCESS_BONUS

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "is_grasping": is_grasping,
            "visit_fraction": visit_fraction,
            "nearest_cp_reward": nearest_cp_reward,
            "closure_reward": closure_reward,
            "smooth_penalty": smooth_penalty,
            "tcp_to_cube_dist": tcp_to_cube_dist,
            "start_checkpoint_dist": start_checkpoint_dist,
        }
        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


# =====================================================================
# Difficulty variants
# =====================================================================
@register_env("TraceShapeEasy-VLA-v0", max_episode_steps=250)
class TraceShapeEasyVLAEnv(TraceShapeVLABaseEnv):
    """Circle only."""

    AVAILABLE_SHAPES: List[int] = [0]


@register_env("TraceShapeMedium-VLA-v0", max_episode_steps=300)
class TraceShapeMediumVLAEnv(TraceShapeVLABaseEnv):
    """Circle or square."""

    AVAILABLE_SHAPES: List[int] = [0, 1]


@register_env("TraceShapeHard-VLA-v0", max_episode_steps=350)
class TraceShapeHardVLAEnv(TraceShapeVLABaseEnv):
    """Circle, square, or triangle."""

    AVAILABLE_SHAPES: List[int] = [0, 1, 2]
