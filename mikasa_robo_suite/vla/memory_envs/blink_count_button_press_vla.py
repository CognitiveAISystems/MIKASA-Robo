"""Blink-counting and button-press tasks for the VLA benchmark."""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

from mikasa_robo_suite.vla.utils import shapes


class BlinkCountButtonPressVLABaseEnv(BaseEnv):
    """Count a visual cue and reproduce it with discrete button presses.

    The robot first observes a lamp blinking a sampled number of times. After
    the cue ends, it must press the button exactly that many times. This task
    is simple to understand but sensitive to temporal memory and to clean press
    cycles, because repeated partial contacts should not be mistaken for new
    presses.

    Episode flow:
    - The lamp waits briefly, then blinks `N` times.
    - After the cue phase, the robot starts pressing the red button.
    - Each press must be followed by a release and lift before the next one.
    - When done counting, the robot presses the black button to submit.

    Success (`success=True`):
    - Success is produced only when the black submit button is pressed.
    - At submit time, the counted number of valid red-button presses must
      exactly match the target blink count.

    How to customize:
    - `BLINK_COUNT_RANGE` changes the memory difficulty by changing how many
      blinks the agent may need to remember.
    - `PRE_BLINK_OFF_STEPS` changes how long the task waits before cue onset.
    - `BLINK_ON_STEPS` and `BLINK_OFF_STEPS` change the timing pattern of each
      blink and therefore how easy the cue is to parse visually.
    - `BUTTON_*` parameters change the physical button geometry and the press
      detection thresholds.
    - `REQUIRED_LIFT_HEIGHT` changes how much the end effector must lift after a
      press before the next press can be counted reliably.
    """

    LANGUAGE_INSTRUCTION = (
        "Count how many times the blue lamp blinks, press the red button exactly that many times "
        "when the red lamp turns green, then press the black button to submit your answer."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    HEIGHT_OFFSET = 1000.0

    BLINK_COUNT_RANGE: List[int] = [1, 5]
    PRE_BLINK_OFF_STEPS: List[int] = [2, 4]
    BLINK_ON_STEPS: List[int] = [1, 2]
    BLINK_OFF_STEPS: List[int] = [2, 4]

    BUTTON_BASE_HALF_SIZE = np.array([0.065, 0.065, 0.015], dtype=np.float32)
    BUTTON_CAP_RADIUS = 0.03
    BUTTON_CAP_HALF_HEIGHT = 0.014
    BUTTON_CAP_TRAVEL = BUTTON_CAP_HALF_HEIGHT
    BUTTON_PRESS_EVENT_RATIO = 0.35
    BUTTON_RELEASE_READY_RATIO = 0.2
    BUTTON_PRESS_XY_RADIUS = 0.065
    BUTTON_PRESS_Z_MARGIN = 0.03
    CONFIRM_BUTTON_Y_OFFSET = 0.16
    REQUIRED_LIFT_HEIGHT = 0.1
    LIFT_CONFIRM_TOL = 0.015

    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012
    INDICATOR_FORWARD_OFFSET = 0.22
    PHASE_INDICATOR_X_OFFSET = 0.2
    INDICATOR_HEIGHT = 0.0
    DEFAULT_BLINK_COLOR = np.array([0, 0, 255, 255], dtype=np.float32) / 255.0
    PHASE_WAIT_COLOR = np.array([255, 0, 0, 255], dtype=np.float32) / 255.0
    PHASE_READY_COLOR = np.array([0, 255, 0, 255], dtype=np.float32) / 255.0

    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.03
    QVEL_L2_COEF = 0.01

    SUCCESS_BONUS = 30.0
    FAILURE_PENALTY = 25.0

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        blink_color=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if blink_color is None:
            blink_color_arr = self.DEFAULT_BLINK_COLOR.copy()
        else:
            blink_color_arr = np.asarray(blink_color, dtype=np.float32)
            if blink_color_arr.shape[0] == 3:
                blink_color_arr = np.concatenate([blink_color_arr, np.array([1.0], dtype=np.float32)])
            if float(np.max(blink_color_arr)) > 1.0:
                blink_color_arr = blink_color_arr / 255.0
        self.blink_color = blink_color_arr
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

        base_builder = self.scene.create_actor_builder()
        base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
        base_builder.add_box_visual(
            half_size=self.BUTTON_BASE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=np.array([55, 64, 78, 255]) / 255.0),
        )
        self.button_base = _build_by_type(
            base_builder,
            name="button_base",
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
            name="button_cap",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        confirm_base_builder = self.scene.create_actor_builder()
        confirm_base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
        confirm_base_builder.add_box_visual(
            half_size=self.BUTTON_BASE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=np.array([40, 40, 40, 255]) / 255.0),
        )
        self.confirm_button_base = _build_by_type(
            confirm_base_builder,
            name="confirm_button_base",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        confirm_cap_builder = self.scene.create_actor_builder()
        confirm_cap_builder.add_cylinder_collision(
            radius=self.BUTTON_CAP_RADIUS, half_length=self.BUTTON_CAP_HALF_HEIGHT
        )
        confirm_cap_builder.add_cylinder_visual(
            radius=self.BUTTON_CAP_RADIUS,
            half_length=self.BUTTON_CAP_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(base_color=np.array([20, 20, 20, 255]) / 255.0),
        )
        self.confirm_button_cap = _build_by_type(
            confirm_cap_builder,
            name="confirm_button_cap",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        lamp_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="blink_lamp",
            body_type="kinematic",
            add_collision=False,
            initial_pose=default_initial_pose,
            base_radius=self.LAMP_BASE_RADIUS,
            base_half_height=self.LAMP_BASE_HALF_HEIGHT,
            stem_radius=self.LAMP_STEM_RADIUS,
            stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
            bulb_radius=self.LAMP_BULB_RADIUS,
            bulb_on_color=self.blink_color,
        )
        self.lamp_body = lamp_parts["body"]
        self.lamp_bulb_off = lamp_parts["bulb_off"]
        self.lamp_bulb_on = lamp_parts["bulb_on"]

        phase_lamp_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="phase_lamp",
            body_type="kinematic",
            add_collision=False,
            initial_pose=default_initial_pose,
            base_radius=self.LAMP_BASE_RADIUS,
            base_half_height=self.LAMP_BASE_HALF_HEIGHT,
            stem_radius=self.LAMP_STEM_RADIUS,
            stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
            bulb_radius=self.LAMP_BULB_RADIUS,
            bulb_off_color=self.PHASE_WAIT_COLOR,
            bulb_on_color=self.PHASE_READY_COLOR,
        )
        self.phase_lamp_body = phase_lamp_parts["body"]
        self.phase_lamp_red = phase_lamp_parts["bulb_off"]
        self.phase_lamp_green = phase_lamp_parts["bulb_on"]
        shapes._set_actor_visual_rgba(
            self.phase_lamp_red,
            self.PHASE_WAIT_COLOR,
            emission_scale=3.0,
            remove_textures=True,
        )
        shapes._set_actor_visual_rgba(
            self.phase_lamp_green,
            self.PHASE_READY_COLOR,
            emission_scale=3.0,
            remove_textures=True,
        )

        n = self.num_envs
        d = self.device
        self.cue_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.empty_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.pre_blink_steps_per_env = torch.zeros(n, dtype=torch.int64, device=d)
        self.max_blinks = int(self.BLINK_COUNT_RANGE[1])
        self.blink_on_schedule = torch.zeros((n, self.max_blinks), dtype=torch.int64, device=d)
        self.blink_off_schedule = torch.zeros((n, self.max_blinks), dtype=torch.int64, device=d)
        self.blink_start_steps = torch.zeros((n, self.max_blinks), dtype=torch.int64, device=d)

        self.target_blinks = torch.zeros(n, dtype=torch.int64, device=d)
        self.press_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.raw_press_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.press_ready = torch.ones(n, dtype=torch.bool, device=d)
        self.pending_press = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_raw_press_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_press_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_release_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.failed = torch.zeros(n, dtype=torch.bool, device=d)
        self.submit_attempted = torch.zeros(n, dtype=torch.bool, device=d)
        self.submit_success = torch.zeros(n, dtype=torch.bool, device=d)
        self.confirm_press_ready = torch.ones(n, dtype=torch.bool, device=d)
        self.new_confirm_press_event = torch.zeros(n, dtype=torch.bool, device=d)

        self.button_xy = torch.zeros((n, 2), dtype=torch.float32, device=d)
        self.button_cap_unpressed_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_top_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_press_depth = torch.zeros(n, dtype=torch.float32, device=d)
        self.confirm_button_xy = torch.zeros((n, 2), dtype=torch.float32, device=d)
        self.confirm_button_cap_unpressed_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.confirm_button_top_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.confirm_button_press_depth = torch.zeros(n, dtype=torch.float32, device=d)
        self.press_start_tcp_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.indicator_on_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.indicator_off_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.phase_indicator_on_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.phase_indicator_off_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.button_cap_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=d)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            env_idx = env_idx.to(self.device)

            self.task_cue = None
            self.reward_dict = None

            button_xyz = torch.zeros((b, 3), device=self.device)
            button_xyz[..., 0] = torch.rand((b,), device=self.device) * 0.10 - 0.15
            button_xyz[..., 1] = (torch.rand((b,), device=self.device) - 0.5) * 0.24
            button_xyz[..., 2] = float(self.BUTTON_BASE_HALF_SIZE[2])

            button_base_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.button_base.set_pose(Pose.create_from_pq(p=button_xyz, q=button_base_q))

            unpressed_z = float(self.BUTTON_BASE_HALF_SIZE[2]) * 2.0 + self.BUTTON_CAP_HALF_HEIGHT
            cap_xyz = button_xyz.clone()
            cap_xyz[..., 2] = unpressed_z
            cap_q = self.button_cap_quat.unsqueeze(0).repeat(b, 1)
            self.button_cap.set_pose(Pose.create_from_pq(p=cap_xyz, q=cap_q))

            confirm_button_xyz = button_xyz.clone()
            confirm_button_xyz[..., 1] += self.CONFIRM_BUTTON_Y_OFFSET
            confirm_button_xyz[..., 1] = torch.clamp(confirm_button_xyz[..., 1], -0.28, 0.28)
            self.confirm_button_base.set_pose(Pose.create_from_pq(p=confirm_button_xyz, q=button_base_q))

            confirm_cap_xyz = confirm_button_xyz.clone()
            confirm_cap_xyz[..., 2] = unpressed_z
            self.confirm_button_cap.set_pose(Pose.create_from_pq(p=confirm_cap_xyz, q=cap_q))

            indicator_on_xyz = button_xyz.clone()
            indicator_on_xyz[..., 0] += self.INDICATOR_FORWARD_OFFSET
            indicator_on_xyz[..., 2] = self.INDICATOR_HEIGHT
            indicator_off_xyz = indicator_on_xyz.clone()
            indicator_off_xyz[..., 2] += self.HEIGHT_OFFSET
            self.lamp_body.set_pose(Pose.create_from_pq(p=indicator_on_xyz, q=button_base_q))
            self.lamp_bulb_off.set_pose(Pose.create_from_pq(p=indicator_on_xyz, q=button_base_q))
            self.lamp_bulb_on.set_pose(Pose.create_from_pq(p=indicator_off_xyz, q=button_base_q))

            phase_indicator_on_xyz = indicator_on_xyz.clone()
            phase_indicator_on_xyz[..., 1] -= self.PHASE_INDICATOR_X_OFFSET
            phase_indicator_off_xyz = phase_indicator_on_xyz.clone()
            phase_indicator_off_xyz[..., 2] += self.HEIGHT_OFFSET
            self.phase_lamp_body.set_pose(Pose.create_from_pq(p=phase_indicator_on_xyz, q=button_base_q))
            self.phase_lamp_red.set_pose(Pose.create_from_pq(p=phase_indicator_on_xyz, q=button_base_q))
            self.phase_lamp_green.set_pose(Pose.create_from_pq(p=phase_indicator_off_xyz, q=button_base_q))

            self.button_xy[env_idx] = button_xyz[:, :2]
            self.button_cap_unpressed_z[env_idx] = unpressed_z
            self.button_top_z[env_idx] = unpressed_z + self.BUTTON_CAP_HALF_HEIGHT
            self.button_press_depth[env_idx] = 0.0
            self.confirm_button_xy[env_idx] = confirm_button_xyz[:, :2]
            self.confirm_button_cap_unpressed_z[env_idx] = unpressed_z
            self.confirm_button_top_z[env_idx] = unpressed_z + self.BUTTON_CAP_HALF_HEIGHT
            self.confirm_button_press_depth[env_idx] = 0.0
            self.press_start_tcp_z[env_idx] = 0.0
            self.indicator_on_pos[env_idx] = indicator_on_xyz
            self.indicator_off_pos[env_idx] = indicator_off_xyz
            self.phase_indicator_on_pos[env_idx] = phase_indicator_on_xyz
            self.phase_indicator_off_pos[env_idx] = phase_indicator_off_xyz

            blink_count = torch.randint(
                low=self.BLINK_COUNT_RANGE[0],
                high=self.BLINK_COUNT_RANGE[1] + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )
            pre_steps = torch.randint(
                low=self.PRE_BLINK_OFF_STEPS[0],
                high=self.PRE_BLINK_OFF_STEPS[1] + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )
            on_schedule = torch.randint(
                low=self.BLINK_ON_STEPS[0],
                high=self.BLINK_ON_STEPS[1] + 1,
                size=(b, self.max_blinks),
                device=self.device,
                dtype=torch.int64,
            )
            off_schedule = torch.randint(
                low=self.BLINK_OFF_STEPS[0],
                high=self.BLINK_OFF_STEPS[1] + 1,
                size=(b, self.max_blinks),
                device=self.device,
                dtype=torch.int64,
            )
            blink_idx = torch.arange(self.max_blinks, device=self.device).unsqueeze(0)
            valid_blink_mask = blink_idx < blink_count.unsqueeze(1)
            on_schedule = on_schedule * valid_blink_mask.to(torch.int64)
            off_schedule = off_schedule * valid_blink_mask.to(torch.int64)
            blink_durations = on_schedule + off_schedule
            start_offsets = torch.cumsum(blink_durations, dim=1) - blink_durations
            blink_start_steps = pre_steps.unsqueeze(1) + start_offsets
            cue_steps = pre_steps + torch.sum(blink_durations, dim=1)

            self.target_blinks[env_idx] = blink_count
            self.pre_blink_steps_per_env[env_idx] = pre_steps
            self.blink_on_schedule[env_idx] = on_schedule
            self.blink_off_schedule[env_idx] = off_schedule
            self.blink_start_steps[env_idx] = blink_start_steps
            self.cue_steps_per_env[env_idx] = cue_steps
            self.empty_steps_per_env[env_idx] = 0

            self.press_count[env_idx] = 0
            self.raw_press_count[env_idx] = 0
            self.press_ready[env_idx] = True
            self.pending_press[env_idx] = False
            self.new_raw_press_event[env_idx] = False
            self.new_press_event[env_idx] = False
            self.new_release_event[env_idx] = False
            self.failed[env_idx] = False
            self.submit_attempted[env_idx] = False
            self.submit_success[env_idx] = False
            self.confirm_press_ready[env_idx] = True
            self.new_confirm_press_event[env_idx] = False

            self.oracle_info = self.target_blinks.to(torch.uint8)

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

        cue_mask = elapsed < self.cue_steps_per_env
        action_mask = ~cue_mask

        elapsed_col = elapsed.unsqueeze(1)
        blink_idx = torch.arange(self.max_blinks, device=self.device).unsqueeze(0)
        valid_blinks = blink_idx < self.target_blinks.unsqueeze(1)
        start = self.blink_start_steps
        end = start + self.blink_on_schedule
        in_on_window = (elapsed_col >= start) & (elapsed_col < end)
        light_on = torch.any(valid_blinks & in_on_window, dim=1) & cue_mask

        shapes._set_actor_visual_rgba(
            self.lamp_bulb_on,
            self.blink_color,
            emission_scale=20.0,
            remove_textures=True,
        )

        off_pose = self.lamp_bulb_off.pose.raw_pose.clone()
        off_pose[light_on, :3] = self.indicator_off_pos[light_on]
        off_pose[~light_on, :3] = self.indicator_on_pos[~light_on]
        self.lamp_bulb_off.pose = off_pose

        on_pose = self.lamp_bulb_on.pose.raw_pose.clone()
        on_pose[light_on, :3] = self.indicator_on_pos[light_on]
        on_pose[~light_on, :3] = self.indicator_off_pos[~light_on]
        self.lamp_bulb_on.pose = on_pose

        phase_ready = action_mask
        phase_red_pose = self.phase_lamp_red.pose.raw_pose.clone()
        phase_red_pose[phase_ready, :3] = self.phase_indicator_off_pos[phase_ready]
        phase_red_pose[~phase_ready, :3] = self.phase_indicator_on_pos[~phase_ready]
        self.phase_lamp_red.pose = phase_red_pose

        phase_green_pose = self.phase_lamp_green.pose.raw_pose.clone()
        phase_green_pose[phase_ready, :3] = self.phase_indicator_on_pos[phase_ready]
        phase_green_pose[~phase_ready, :3] = self.phase_indicator_off_pos[~phase_ready]
        self.phase_lamp_green.pose = phase_green_pose

        tcp_pos = self.agent.tcp.pose.p
        tcp_xy = tcp_pos[:, :2]
        tcp_z = tcp_pos[:, 2]

        xy_dist = torch.linalg.norm(tcp_xy - self.button_xy, axis=1)
        raw_depth = self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
        depth = torch.clamp(raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
        depth = depth * (xy_dist < self.BUTTON_PRESS_XY_RADIUS).float()
        self.button_press_depth = depth

        confirm_xy_dist = torch.linalg.norm(tcp_xy - self.confirm_button_xy, axis=1)
        confirm_raw_depth = self.confirm_button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
        confirm_depth = torch.clamp(confirm_raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
        confirm_depth = confirm_depth * (confirm_xy_dist < self.BUTTON_PRESS_XY_RADIUS).float()
        self.confirm_button_press_depth = confirm_depth

        cap_pose = self.button_cap.pose.raw_pose.clone()
        cap_pose[:, 0:2] = self.button_xy
        cap_pose[:, 2] = self.button_cap_unpressed_z - depth
        cap_pose[:, 3:7] = self.button_cap_quat.repeat(cap_pose.shape[0], 1)
        self.button_cap.pose = cap_pose

        confirm_cap_pose = self.confirm_button_cap.pose.raw_pose.clone()
        confirm_cap_pose[:, 0:2] = self.confirm_button_xy
        confirm_cap_pose[:, 2] = self.confirm_button_cap_unpressed_z - confirm_depth
        confirm_cap_pose[:, 3:7] = self.button_cap_quat.repeat(confirm_cap_pose.shape[0], 1)
        self.confirm_button_cap.pose = confirm_cap_pose

        pressed = depth >= (self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO)
        released = depth <= (self.BUTTON_CAP_TRAVEL * self.BUTTON_RELEASE_READY_RATIO)
        self.new_release_event = (~self.press_ready) & released & action_mask
        self.press_ready = self.press_ready | self.new_release_event

        self.new_raw_press_event = pressed & self.press_ready & action_mask & (~self.failed) & (~self.pending_press)
        self.press_start_tcp_z[self.new_raw_press_event] = tcp_z[self.new_raw_press_event]
        self.raw_press_count = self.raw_press_count + self.new_raw_press_event.to(torch.int64)
        self.pending_press = self.pending_press | self.new_raw_press_event
        self.press_ready = self.press_ready & (~self.new_raw_press_event)

        lift_target_z = self.press_start_tcp_z + self.REQUIRED_LIFT_HEIGHT
        at_lift_target = tcp_z >= (lift_target_z - self.LIFT_CONFIRM_TOL)

        self.new_press_event = self.pending_press & at_lift_target & self.press_ready & action_mask & (~self.failed)
        self.press_count = self.press_count + self.new_press_event.to(torch.int64)
        self.pending_press = self.pending_press & (~self.new_press_event)

        confirm_pressed = confirm_depth >= (self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO)
        confirm_released = confirm_depth <= (self.BUTTON_CAP_TRAVEL * self.BUTTON_RELEASE_READY_RATIO)
        self.confirm_press_ready = self.confirm_press_ready | (confirm_released & action_mask)
        self.new_confirm_press_event = (
            confirm_pressed & self.confirm_press_ready & action_mask & (~self.submit_attempted)
        )
        self.confirm_press_ready = self.confirm_press_ready & (~self.new_confirm_press_event)

        self.failed = self.failed | (self.raw_press_count > self.target_blinks)
        count_correct = self.press_count == self.target_blinks
        self.submit_success = self.submit_success | (self.new_confirm_press_event & count_correct & (~self.failed))
        self.failed = self.failed | (self.new_confirm_press_event & (~count_correct))
        self.submit_attempted = self.submit_attempted | self.new_confirm_press_event
        success = action_mask & self.submit_success

        self.obj_to_goal_pos = self.button_cap.pose.p - self.agent.tcp.pose.p

        return {
            "success": success,
            "failed": self.failed,
            "submit_attempted": self.submit_attempted,
            "submit_success": self.submit_success,
            "action_mask": action_mask,
            "phase_ready": phase_ready,
            "light_on": light_on,
            "press_count": self.press_count,
            "raw_press_count": self.raw_press_count,
            "target_blinks": self.target_blinks,
            "new_raw_press_event": self.new_raw_press_event,
            "new_press_event": self.new_press_event,
            "new_release_event": self.new_release_event,
            "new_confirm_press_event": self.new_confirm_press_event,
            "press_ready": self.press_ready,
            "pending_press": self.pending_press,
            "lift_target_z": lift_target_z,
            "at_lift_target": at_lift_target,
            "xy_dist_to_button": xy_dist,
            "press_depth": self.button_press_depth,
            "xy_dist_to_confirm_button": confirm_xy_dist,
            "confirm_press_depth": self.confirm_button_press_depth,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            active_bulb_pose = self.lamp_bulb_off.pose.raw_pose.clone()
            light_on = info["light_on"]
            active_bulb_pose[light_on] = self.lamp_bulb_on.pose.raw_pose[light_on]
            phase_pose = self.phase_lamp_red.pose.raw_pose.clone()
            phase_ready = info["phase_ready"]
            phase_pose[phase_ready] = self.phase_lamp_green.pose.raw_pose[phase_ready]
            obs.update(
                button_base_pose=self.button_base.pose.raw_pose,
                button_cap_pose=self.button_cap.pose.raw_pose,
                confirm_button_base_pose=self.confirm_button_base.pose.raw_pose,
                confirm_button_cap_pose=self.confirm_button_cap.pose.raw_pose,
                indicator_pose=active_bulb_pose,
                phase_indicator_pose=phase_pose,
                press_count=self.press_count,
                target_blinks=self.target_blinks,
                action_mask=info["action_mask"],
                phase_ready=phase_ready,
                press_ready=info["press_ready"],
                pending_press=info["pending_press"],
                submit_attempted=info["submit_attempted"],
                submit_success=info["submit_success"],
                at_lift_target=info["at_lift_target"],
                oracle_info=self.oracle_info,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if isinstance(info, dict):
            success = info.get("success", None)
            failed = info.get("failed", None)

            if torch.is_tensor(terminated):
                terminated = terminated.to(dtype=torch.bool)
                if torch.is_tensor(success):
                    terminated = terminated | success.to(dtype=torch.bool)
                elif success is not None:
                    terminated = terminated | torch.as_tensor(success, device=terminated.device).to(dtype=torch.bool)

                if torch.is_tensor(failed):
                    terminated = terminated | failed.to(dtype=torch.bool)
                elif failed is not None:
                    terminated = terminated | torch.as_tensor(failed, device=terminated.device).to(dtype=torch.bool)
            else:
                terminated = bool(terminated)
                if success is not None:
                    if torch.is_tensor(success):
                        terminated = terminated or bool(success.any().item())
                    else:
                        terminated = terminated or bool(success)
                if failed is not None:
                    if torch.is_tensor(failed):
                        terminated = terminated or bool(failed.any().item())
                    else:
                        terminated = terminated or bool(failed)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        tcp_xy = tcp_pos[:, :2]
        tcp_z = tcp_pos[:, 2]
        xy_dist = torch.linalg.norm(tcp_xy - self.button_xy, axis=1)
        z_above_button = torch.clamp(tcp_z - self.button_top_z, min=0.0)
        tcp_to_button_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        tcp_to_confirm_dist = torch.linalg.norm(self.confirm_button_cap.pose.p - self.agent.tcp.pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(8.0 * tcp_to_button_dist)
        confirm_reaching_reward = 1 - torch.tanh(8.0 * tcp_to_confirm_dist)
        press_progress_reward = torch.clamp(info["press_depth"] / self.BUTTON_CAP_TRAVEL, min=0.0, max=1.0)
        confirm_press_progress_reward = torch.clamp(
            info["confirm_press_depth"] / self.BUTTON_CAP_TRAVEL, min=0.0, max=1.0
        )

        target = torch.clamp(info["target_blinks"].float(), min=1.0)
        count_error = torch.abs(info["target_blinks"].float() - info["press_count"].float())
        count_progress = 1.0 - torch.clamp(count_error / target, min=0.0, max=1.0)
        new_raw_press_reward = info["new_raw_press_event"].float()
        new_press_reward = info["new_press_event"].float()
        new_release_reward = info["new_release_event"].float()
        new_confirm_press_reward = info["new_confirm_press_event"].float()
        pending_press_f = info["pending_press"].float()
        phase_ready = info["action_mask"].float()
        cue_phase = 1.0 - phase_ready
        action_phase1 = phase_ready * (1.0 - pending_press_f)
        action_phase2 = phase_ready * pending_press_f
        lift_target_z = self.press_start_tcp_z + self.REQUIRED_LIFT_HEIGHT
        lift_gap = torch.clamp(lift_target_z - tcp_z, min=0.0)
        lift_reward = 1.0 - torch.tanh(4.0 * lift_gap)
        vertical_alignment_reward = 1.0 - torch.tanh(8.0 * xy_dist)

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)
        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, axis=1)
        delta_action_l2 = torch.linalg.norm(delta_action, axis=1)
        qvel_l2 = torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        cue_reach_reward = cue_phase * reaching_reward
        cue_no_press_reward = cue_phase * (1.0 - press_progress_reward)

        action_count_reward = action_phase1 * count_progress
        action_new_raw_press_reward = action_phase1 * new_raw_press_reward
        button_reach_reward = action_phase1 * reaching_reward
        button_press_reward = action_phase1 * press_progress_reward
        phase1_lift_penalty = action_phase1 * torch.clamp(
            (z_above_button - 0.20) / self.REQUIRED_LIFT_HEIGHT, min=0.0, max=1.0
        )
        phase1_hover_penalty = action_phase1 * torch.clamp(
            (z_above_button - 0.06) / self.REQUIRED_LIFT_HEIGHT, min=0.0, max=1.0
        )

        phase2_lift_reward = action_phase2 * lift_reward
        phase2_vertical_reward = action_phase2 * vertical_alignment_reward
        phase2_release_reward = action_phase2 * new_release_reward
        confirm_cycle_reward = action_phase2 * new_press_reward
        submit_phase = phase_ready * (info["press_count"] == info["target_blinks"]).float()

        early_press_penalty = cue_phase * press_progress_reward
        hold_down_penalty = action_phase2 * press_progress_reward
        reward = (
            0.25 * reaching_reward * phase_ready
            + 0.75 * cue_reach_reward
            + 0.75 * cue_no_press_reward
            + 1.5 * button_reach_reward
            + 2.0 * button_press_reward
            + 1.0 * action_count_reward
            + 4.0 * action_new_raw_press_reward
            + 5.0 * phase2_lift_reward
            + 2.0 * phase2_vertical_reward
            + 2.0 * phase2_release_reward
            + 6.0 * confirm_cycle_reward
            + 1.5 * submit_phase * confirm_reaching_reward
            + 2.0 * submit_phase * confirm_press_progress_reward
            + 10.0 * new_confirm_press_reward
            - 1.5 * early_press_penalty
            - 2.0 * hold_down_penalty
            - 1.5 * phase1_lift_penalty
            - 3.0 * phase1_hover_penalty
            - smooth_penalty
        )
        reward = reward * phase_ready
        reward -= self.FAILURE_PENALTY * info["failed"].float()
        reward[info["success"]] = self.SUCCESS_BONUS

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "press_progress_reward": press_progress_reward,
            "count_progress": count_progress,
            "new_raw_press_reward": new_raw_press_reward,
            "new_press_reward": new_press_reward,
            "new_release_reward": new_release_reward,
            "cue_reach_reward": cue_reach_reward,
            "cue_no_press_reward": cue_no_press_reward,
            "action_phase1": action_phase1,
            "action_phase2": action_phase2,
            "button_reach_reward": button_reach_reward,
            "button_press_reward": button_press_reward,
            "action_count_reward": action_count_reward,
            "action_new_raw_press_reward": action_new_raw_press_reward,
            "phase2_lift_reward": phase2_lift_reward,
            "phase2_vertical_reward": phase2_vertical_reward,
            "phase2_release_reward": phase2_release_reward,
            "confirm_cycle_reward": confirm_cycle_reward,
            "early_press_penalty": early_press_penalty,
            "phase1_lift_penalty": phase1_lift_penalty,
            "phase1_hover_penalty": phase1_hover_penalty,
            "hold_down_penalty": hold_down_penalty,
            "tcp_to_button_dist": tcp_to_button_dist,
            "tcp_to_confirm_dist": tcp_to_confirm_dist,
            "xy_dist_to_button": xy_dist,
            "lift_gap": lift_gap,
            "confirm_reaching_reward": confirm_reaching_reward,
            "confirm_press_progress_reward": confirm_press_progress_reward,
            "new_confirm_press_reward": new_confirm_press_reward,
            "submit_phase": submit_phase,
            "press_count": info["press_count"].float(),
            "target_blinks": info["target_blinks"].float(),
            "failed": info["failed"].float(),
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
        }
        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


# ----- Standard tasks -----
@register_env("BlinkCountButtonPressEasy-VLA-v0", max_episode_steps=150)
class BlinkCountButtonPressEasyVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [1, 3]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 10]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]


@register_env("BlinkCountButtonPressMedium-VLA-v0", max_episode_steps=200)
class BlinkCountButtonPressMediumVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [1, 5]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 10]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]


@register_env("BlinkCountButtonPressHard-VLA-v0", max_episode_steps=300)
class BlinkCountButtonPressHardVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [1, 7]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 10]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]


# ----- Long-horizon tasks -----
@register_env("BlinkCountButtonPressEasy-Long-VLA-v0", max_episode_steps=1200)
class BlinkCountButtonPressEasyLongVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [1, 10]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 50]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]


@register_env("BlinkCountButtonPressMedium-Long-VLA-v0", max_episode_steps=1200)
class BlinkCountButtonPressMediumLongVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [10, 20]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 50]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]


@register_env("BlinkCountButtonPressHard-Long-VLA-v0", max_episode_steps=1200)
class BlinkCountButtonPressHardLongVLAEnv(BlinkCountButtonPressVLABaseEnv):
    BLINK_COUNT_RANGE: List[int] = [20, 30]
    PRE_BLINK_OFF_STEPS: List[int] = [5, 50]
    BLINK_ON_STEPS: List[int] = [3, 3]
    BLINK_OFF_STEPS: List[int] = [3, 5]
