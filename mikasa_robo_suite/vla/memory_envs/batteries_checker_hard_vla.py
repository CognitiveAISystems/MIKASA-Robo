"""Batteries Checker tasks for the VLA memory benchmark."""

from typing import Any, Dict, Union

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


class BatteriesCheckerVLABaseEnv(BaseEnv):
    """Battery-testing task with memory over repeated check cycles.

    The scene contains a tray of batteries, a socket with a lamp, and a button.
    The robot must test batteries one at a time, remember which ones worked,
    and only then move on to the next candidate. Because each battery must be
    returned to its original slot before the next confirmation, the task mixes
    memory with careful sequential manipulation.

    Episode flow:
    - Pick one battery from the tray and insert it into the socket.
    - Read the lamp outcome: lit means the battery is working.
    - Remove the battery and return it to its original tray slot.
    - Press the button to mark that this battery has been checked.

    Success (`success=True`):
    - Every working battery must be discovered through the full insert-return-
      confirm procedure. Partial progress does not count as success.

    How to customize:
    - `ACTIVE_BATTERY_COUNT` controls how many batteries are present in the
      episode and therefore how long the search can become.
    - `WORKING_BATTERY_COUNT` controls how many of those batteries are true
      positives that the agent must eventually identify.
    - `SOCKET_INSERT_XY_TOL` and `SOCKET_INSERT_Z_TOL` control how precisely a
      battery must be placed before the environment counts it as inserted.
    - `SLOT_RETURN_XY_TOL` and `SLOT_RETURN_Z_TOL` control how accurately the
      battery must be put back into its home slot.
    - `BUTTON_*` parameters control the size, travel, and press thresholds of
      the confirmation button.
    - `LAMP_AFTERGLOW_STEPS` controls how long the lamp remains visibly on after
      a successful working-battery test.
    """

    LANGUAGE_INSTRUCTION = "Find all working batteries by inserting each one into the socket, observing the lamp result, returning it from the socket to its initial slot, and then pressing the button to confirm."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    HEIGHT_OFFSET = 1000.0

    TRAY_ROWS = 5
    TRAY_COLS = 3
    NUM_BATTERIES = TRAY_ROWS * TRAY_COLS
    ACTIVE_BATTERY_COUNT = 15
    WORKING_BATTERY_COUNT = 3
    SLOT_SPACING_X = 0.052
    SLOT_SPACING_Y = 0.050

    BATTERY_RADIUS = 0.010
    BATTERY_HALF_HEIGHT = 0.030
    BATTERY_COLOR = np.array([70, 190, 90, 255], dtype=np.float32) / 255.0
    BATTERY_STATIC_FRICTION = 2.0
    BATTERY_DYNAMIC_FRICTION = 2.0
    BATTERY_RESTITUTION = 0.0

    TRAY_HALF_HEIGHT = BATTERY_HALF_HEIGHT * 0.5
    TRAY_PADDING_X = 0.028
    TRAY_PADDING_Y = 0.026
    TRAY_COLOR = np.array([72, 86, 108, 255], dtype=np.float32) / 255.0
    SLOT_VISUAL_COLOR = np.array([42, 48, 62, 255], dtype=np.float32) / 255.0

    SOCKET_HALF_SIZE = np.array([0.048, 0.040, BATTERY_HALF_HEIGHT * 0.5], dtype=np.float32)
    SOCKET_SLOT_RADIUS = BATTERY_RADIUS * 1.6
    SOCKET_COLOR = np.array([88, 92, 98, 255], dtype=np.float32) / 255.0
    SOCKET_SLOT_COLOR = np.array([32, 32, 36, 255], dtype=np.float32) / 255.0

    SOCKET_X_OFFSET_FROM_TRAY = 0.2
    BUTTON_X_OFFSET_FROM_TRAY = 0.00
    BUTTON_Y_OFFSET_FROM_TRAY = 0.24

    BUTTON_BASE_HALF_SIZE = np.array([0.075, 0.075, 0.015], dtype=np.float32)
    BUTTON_CAP_RADIUS = 0.033
    BUTTON_CAP_HALF_HEIGHT = 0.014
    BUTTON_CAP_TRAVEL = BUTTON_CAP_HALF_HEIGHT
    BUTTON_PRESS_EVENT_RATIO = 0.35
    BUTTON_RELEASE_READY_RATIO = 0.2
    BUTTON_PRESS_XY_RADIUS = 0.075
    BUTTON_PRESS_Z_MARGIN = 0.030

    SOCKET_INSERT_XY_TOL = 0.010
    SOCKET_INSERT_Z_TOL = 0.025
    SLOT_RETURN_XY_TOL = 0.023
    SLOT_RETURN_Z_TOL = 0.024

    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012
    LAMP_X_OFFSET_FROM_SOCKET = 0.08
    LAMP_Y_OFFSET_FROM_SOCKET = 0.0
    LAMP_HEIGHT = 0.0
    LAMP_OFF_COLOR = np.array([255, 255, 255, 255], dtype=np.float32) / 255.0
    LAMP_ON_COLOR = np.array([255, 236, 110, 255], dtype=np.float32) / 255.0
    LAMP_AFTERGLOW_STEPS = 7

    STAGE_INSERT = 0
    STAGE_RETURN = 1
    STAGE_CONFIRM = 2

    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.02
    QVEL_L2_COEF = 0.01

    SUCCESS_BONUS = 40.0

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
                max_rigid_patch_count=2**20,
                max_rigid_contact_count=2**21,
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

    def _build_tray(self, initial_pose: sapien.Pose):
        tray_half_x = (self.TRAY_ROWS - 1) * self.SLOT_SPACING_X * 0.5 + self.TRAY_PADDING_X
        tray_half_y = (self.TRAY_COLS - 1) * self.SLOT_SPACING_Y * 0.5 + self.TRAY_PADDING_Y
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[tray_half_x, tray_half_y, self.TRAY_HALF_HEIGHT],
            material=sapien.render.RenderMaterial(base_color=self.TRAY_COLOR),
        )
        for r in range(self.TRAY_ROWS):
            for c in range(self.TRAY_COLS):
                local_x = (r - (self.TRAY_ROWS - 1) * 0.5) * self.SLOT_SPACING_X
                local_y = (c - (self.TRAY_COLS - 1) * 0.5) * self.SLOT_SPACING_Y
                slot_pose = sapien.Pose(
                    p=[local_x, local_y, self.TRAY_HALF_HEIGHT + 0.0015],
                    q=euler2quat(0, np.pi / 2, 0),
                )
                builder.add_cylinder_visual(
                    pose=slot_pose,
                    radius=self.BATTERY_RADIUS * 1.2,
                    half_length=0.0015,
                    material=sapien.render.RenderMaterial(base_color=self.SLOT_VISUAL_COLOR),
                )
        return _build_by_type(
            builder,
            name="battery_tray",
            body_type="kinematic",
            initial_pose=initial_pose,
        )

    def _build_socket_box(self, initial_pose: sapien.Pose):
        builder = self.scene.create_actor_builder()
        outer_hx, outer_hy, outer_hz = [float(v) for v in self.SOCKET_HALF_SIZE]
        inner_hx = min(self.SOCKET_SLOT_RADIUS * 1.15, outer_hx - 0.004)
        inner_hy = min(self.SOCKET_SLOT_RADIUS * 1.15, outer_hy - 0.004)
        inner_hx = max(inner_hx, 0.003)
        inner_hy = max(inner_hy, 0.003)

        bottom_hz = max(0.004, outer_hz * 0.35)
        builder.add_box_collision(
            half_size=[outer_hx, outer_hy, bottom_hz],
            pose=sapien.Pose(p=[0.0, 0.0, -outer_hz + bottom_hz]),
        )

        wall_x_hx = max(0.002, 0.5 * (outer_hx - inner_hx))
        wall_y_hy = max(0.002, 0.5 * (outer_hy - inner_hy))
        wall_hz = outer_hz

        builder.add_box_collision(
            half_size=[wall_x_hx, outer_hy, wall_hz],
            pose=sapien.Pose(p=[inner_hx + wall_x_hx, 0.0, 0.0]),
        )
        builder.add_box_collision(
            half_size=[wall_x_hx, outer_hy, wall_hz],
            pose=sapien.Pose(p=[-(inner_hx + wall_x_hx), 0.0, 0.0]),
        )
        builder.add_box_collision(
            half_size=[inner_hx, wall_y_hy, wall_hz],
            pose=sapien.Pose(p=[0.0, inner_hy + wall_y_hy, 0.0]),
        )
        builder.add_box_collision(
            half_size=[inner_hx, wall_y_hy, wall_hz],
            pose=sapien.Pose(p=[0.0, -(inner_hy + wall_y_hy), 0.0]),
        )

        builder.add_box_visual(
            half_size=self.SOCKET_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=self.SOCKET_COLOR),
        )
        slot_visual_pose = sapien.Pose(
            p=[0.0, 0.0, self.SOCKET_HALF_SIZE[2] + 0.0015],
            q=euler2quat(0, np.pi / 2, 0),
        )
        builder.add_cylinder_visual(
            pose=slot_visual_pose,
            radius=self.SOCKET_SLOT_RADIUS,
            half_length=0.0015,
            material=sapien.render.RenderMaterial(base_color=self.SOCKET_SLOT_COLOR),
        )
        return _build_by_type(
            builder,
            name="battery_socket_box",
            body_type="kinematic",
            initial_pose=initial_pose,
        )

    def _build_button(self, initial_pose: sapien.Pose):
        base_builder = self.scene.create_actor_builder()
        base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
        base_builder.add_box_visual(
            half_size=self.BUTTON_BASE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=np.array([58, 66, 80, 255]) / 255.0),
        )
        button_base = _build_by_type(
            base_builder,
            name="battery_button_base",
            body_type="kinematic",
            initial_pose=initial_pose,
        )

        cap_builder = self.scene.create_actor_builder()
        cap_builder.add_cylinder_collision(radius=self.BUTTON_CAP_RADIUS, half_length=self.BUTTON_CAP_HALF_HEIGHT)
        cap_builder.add_cylinder_visual(
            radius=self.BUTTON_CAP_RADIUS,
            half_length=self.BUTTON_CAP_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(base_color=np.array([205, 110, 110, 255]) / 255.0),
        )
        button_cap = _build_by_type(
            cap_builder,
            name="battery_button_cap",
            body_type="kinematic",
            initial_pose=initial_pose,
        )
        return button_base, button_cap

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        hidden_pose = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])
        self.tray = self._build_tray(hidden_pose)
        self.socket_box = self._build_socket_box(hidden_pose)
        self.button_base, self.button_cap = self._build_button(hidden_pose)

        lamp_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="battery_checker_lamp",
            body_type="kinematic",
            add_collision=False,
            initial_pose=hidden_pose,
            base_radius=self.LAMP_BASE_RADIUS,
            base_half_height=self.LAMP_BASE_HALF_HEIGHT,
            stem_radius=self.LAMP_STEM_RADIUS,
            stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
            bulb_radius=self.LAMP_BULB_RADIUS,
            bulb_off_color=self.LAMP_OFF_COLOR,
            bulb_on_color=self.LAMP_ON_COLOR,
        )
        self.lamp_body = lamp_parts["body"]
        self.lamp_bulb_off = lamp_parts["bulb_off"]
        self.lamp_bulb_on = lamp_parts["bulb_on"]
        shapes._set_actor_visual_rgba(self.lamp_bulb_on, self.LAMP_ON_COLOR, emission_scale=3.0, remove_textures=True)

        battery_mat = sapien.physx.PhysxMaterial(
            static_friction=self.BATTERY_STATIC_FRICTION,
            dynamic_friction=self.BATTERY_DYNAMIC_FRICTION,
            restitution=self.BATTERY_RESTITUTION,
        )
        self.batteries = []
        for i in range(self.NUM_BATTERIES):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_collision(
                radius=self.BATTERY_RADIUS,
                half_length=self.BATTERY_HALF_HEIGHT,
                material=battery_mat,
            )
            builder.add_cylinder_visual(
                radius=self.BATTERY_RADIUS,
                half_length=self.BATTERY_HALF_HEIGHT,
                material=sapien.render.RenderMaterial(base_color=self.BATTERY_COLOR),
            )
            battery = _build_by_type(
                builder,
                name=f"battery_{i}",
                body_type="dynamic",
                initial_pose=hidden_pose,
            )
            self.batteries.append(battery)

        n = self.num_envs
        d = self.device

        self.action_stage = torch.zeros(n, dtype=torch.int64, device=d)
        self.active_battery_idx = torch.full((n,), -1, dtype=torch.int64, device=d)

        self.working_mask = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.bool, device=d)
        self.checked_mask = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.bool, device=d)
        self.found_working_mask = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.bool, device=d)
        self.active_mask = torch.zeros((n, self.NUM_BATTERIES), dtype=torch.bool, device=d)
        self.active_battery_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.target_working_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.checked_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.found_working_count = torch.zeros(n, dtype=torch.int64, device=d)
        oracle_width = int(np.clip(self.WORKING_BATTERY_COUNT, 1, self.NUM_BATTERIES))
        self.oracle_info = torch.zeros((n, oracle_width), dtype=torch.uint8, device=d)

        self.new_insert_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_return_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_confirm_event = torch.zeros(n, dtype=torch.bool, device=d)
        self.new_button_press_event = torch.zeros(n, dtype=torch.bool, device=d)

        self.tray_slot_positions = torch.zeros((n, self.NUM_BATTERIES, 3), device=d)
        self.socket_slot_pos = torch.zeros((n, 3), device=d)
        self.lamp_on_pos = torch.zeros((n, 3), device=d)
        self.lamp_off_pos = torch.zeros((n, 3), device=d)
        self.button_xy = torch.zeros((n, 2), device=d)
        self.button_cap_unpressed_z = torch.zeros(n, device=d)
        self.button_top_z = torch.zeros(n, device=d)
        self.button_press_depth = torch.zeros(n, device=d)
        self.lamp_afterglow_steps = torch.zeros(n, dtype=torch.int64, device=d)

        self.press_ready = torch.ones(n, dtype=torch.bool, device=d)
        self.button_cap_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=d)
        self.battery_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=d)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            env_idx = env_idx.to(self.device)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            tray_xyz = torch.zeros((b, 3), device=self.device)
            tray_xyz[:, 0] = torch.rand((b,), device=self.device) * 0.08 - 0.09
            tray_xyz[:, 1] = torch.rand((b,), device=self.device) * 0.10 - 0.05
            tray_xyz[:, 2] = self.TRAY_HALF_HEIGHT
            tray_q = torch.tensor(euler2quat(0, 0, np.pi / 2), dtype=torch.float32, device=self.device).repeat(b, 1)
            self.tray.set_pose(Pose.create_from_pq(p=tray_xyz, q=tray_q))

            row_offsets = (
                torch.arange(self.TRAY_ROWS, device=self.device, dtype=torch.float32) - (self.TRAY_ROWS - 1) * 0.5
            ) * self.SLOT_SPACING_X
            col_offsets = (
                torch.arange(self.TRAY_COLS, device=self.device, dtype=torch.float32) - (self.TRAY_COLS - 1) * 0.5
            ) * self.SLOT_SPACING_Y
            rr, cc = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
            local_xy = torch.stack([rr.reshape(-1), cc.reshape(-1)], dim=1)

            tray_slot_pos = torch.zeros((b, self.NUM_BATTERIES, 3), device=self.device)
            tray_slot_pos[:, :, 0] = tray_xyz[:, 0:1] - local_xy[:, 1].unsqueeze(0)
            tray_slot_pos[:, :, 1] = tray_xyz[:, 1 : 1 + 1] + local_xy[:, 0].unsqueeze(0)
            tray_slot_pos[:, :, 2] = self.TRAY_HALF_HEIGHT * 2.0
            self.tray_slot_positions[env_idx] = tray_slot_pos

            active_count = int(np.clip(self.ACTIVE_BATTERY_COUNT, 1, self.NUM_BATTERIES))
            working_count = int(np.clip(self.WORKING_BATTERY_COUNT, 1, active_count))
            active_mask_local = torch.zeros((b, self.NUM_BATTERIES), dtype=torch.bool, device=self.device)
            active_mask_local[:, :active_count] = True
            self.active_mask[env_idx] = active_mask_local
            self.active_battery_count[env_idx] = active_count
            self.target_working_count[env_idx] = working_count

            hidden_pos = torch.zeros((b, 3), device=self.device)
            hidden_pos[:, 2] = self.HEIGHT_OFFSET
            for i, battery in enumerate(self.batteries):
                pos_i = tray_slot_pos[:, i, :]
                present_i = active_mask_local[:, i]
                pos_i = torch.where(present_i.unsqueeze(-1), pos_i, hidden_pos)
                q_i = self.battery_quat.unsqueeze(0).repeat(b, 1)
                battery.set_pose(Pose.create_from_pq(p=pos_i, q=q_i))
                battery.set_linear_velocity(torch.zeros((b, 3), device=self.device))
                battery.set_angular_velocity(torch.zeros((b, 3), device=self.device))

            socket_xyz = torch.zeros((b, 3), device=self.device)
            socket_xyz[:, 0] = tray_xyz[:, 0] + self.SOCKET_X_OFFSET_FROM_TRAY
            socket_xyz[:, 1] = tray_xyz[:, 1]
            socket_xyz[:, 2] = float(self.SOCKET_HALF_SIZE[2])
            self.socket_box.set_pose(Pose.create_from_pq(p=socket_xyz, q=tray_q))

            socket_slot_pos = socket_xyz.clone()
            socket_slot_pos[:, 2] = float(self.SOCKET_HALF_SIZE[2]) * 2.0
            self.socket_slot_pos[env_idx] = socket_slot_pos

            lamp_pos = socket_xyz.clone()
            lamp_pos[:, 0] += self.LAMP_X_OFFSET_FROM_SOCKET
            lamp_pos[:, 1] += self.LAMP_Y_OFFSET_FROM_SOCKET
            lamp_pos[:, 2] = self.LAMP_HEIGHT
            lamp_off = lamp_pos.clone()
            lamp_off[:, 2] += self.HEIGHT_OFFSET
            self.lamp_body.set_pose(Pose.create_from_pq(p=lamp_pos, q=tray_q))
            self.lamp_bulb_off.set_pose(Pose.create_from_pq(p=lamp_pos, q=tray_q))
            self.lamp_bulb_on.set_pose(Pose.create_from_pq(p=lamp_off, q=tray_q))
            self.lamp_on_pos[env_idx] = lamp_pos
            self.lamp_off_pos[env_idx] = lamp_off

            button_xyz = torch.zeros((b, 3), device=self.device)
            button_xyz[:, 0] = tray_xyz[:, 0] + self.BUTTON_X_OFFSET_FROM_TRAY
            button_xyz[:, 1] = tray_xyz[:, 1] + self.BUTTON_Y_OFFSET_FROM_TRAY
            button_xyz[:, 2] = float(self.BUTTON_BASE_HALF_SIZE[2])
            self.button_base.set_pose(Pose.create_from_pq(p=button_xyz, q=tray_q))

            unpressed_z = float(self.BUTTON_BASE_HALF_SIZE[2]) * 2.0 + self.BUTTON_CAP_HALF_HEIGHT
            cap_xyz = button_xyz.clone()
            cap_xyz[:, 2] = unpressed_z
            cap_q = self.button_cap_quat.unsqueeze(0).repeat(b, 1)
            self.button_cap.set_pose(Pose.create_from_pq(p=cap_xyz, q=cap_q))

            self.button_xy[env_idx] = button_xyz[:, :2]
            self.button_cap_unpressed_z[env_idx] = unpressed_z
            self.button_top_z[env_idx] = unpressed_z + self.BUTTON_CAP_HALF_HEIGHT
            self.button_press_depth[env_idx] = 0.0
            self.lamp_afterglow_steps[env_idx] = 0

            self.action_stage[env_idx] = self.STAGE_INSERT
            self.active_battery_idx[env_idx] = -1
            self.checked_mask[env_idx] = ~active_mask_local
            self.found_working_mask[env_idx] = False
            self.checked_count[env_idx] = 0
            self.found_working_count[env_idx] = 0
            self.press_ready[env_idx] = True

            rand_scores = torch.rand((b, self.NUM_BATTERIES), device=self.device)
            rand_scores = torch.where(
                active_mask_local,
                rand_scores,
                torch.full_like(rand_scores, -1.0),
            )
            working_idx = torch.topk(rand_scores, k=working_count, dim=1, largest=True, sorted=False).indices
            working_mask_local = torch.zeros((b, self.NUM_BATTERIES), dtype=torch.bool, device=self.device)
            working_mask_local.scatter_(1, working_idx, True)
            self.working_mask[env_idx] = working_mask_local

            self.oracle_info[env_idx] = 255
            self.oracle_info[env_idx, :working_count] = working_idx.to(torch.uint8)

            self.new_insert_event[env_idx] = False
            self.new_return_event[env_idx] = False
            self.new_confirm_event[env_idx] = False
            self.new_button_press_event[env_idx] = False

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

    def _stack_battery_poses(self):
        pos = torch.stack([bat.pose.p for bat in self.batteries], dim=1)
        raw_pose = torch.stack([bat.pose.raw_pose for bat in self.batteries], dim=1)
        grasp = torch.stack([self.agent.is_grasping(bat) for bat in self.batteries], dim=1)
        return pos, raw_pose, grasp

    def evaluate(self):
        cue_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        action_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.new_insert_event[:] = False
        self.new_return_event[:] = False
        self.new_confirm_event[:] = False

        battery_pos, battery_raw_pose, grasp_mask = self._stack_battery_poses()
        socket_xy = self.socket_slot_pos[:, :2]
        socket_z = self.socket_slot_pos[:, 2]

        dist_xy_to_socket = torch.linalg.norm(battery_pos[:, :, :2] - socket_xy.unsqueeze(1), dim=2)
        dist_z_to_socket = torch.abs(battery_pos[:, :, 2] - socket_z.unsqueeze(1))
        in_socket = (
            (dist_xy_to_socket <= self.SOCKET_INSERT_XY_TOL)
            & (dist_z_to_socket <= self.SOCKET_INSERT_Z_TOL)
            & (~grasp_mask)
            & self.active_mask
        )
        socket_has_battery = in_socket.any(dim=1)
        socket_battery_idx = torch.argmax(in_socket.to(torch.int64), dim=1)
        socket_has_working = (in_socket & self.working_mask).any(dim=1)

        self.lamp_afterglow_steps = torch.clamp(self.lamp_afterglow_steps - action_mask.to(torch.int64), min=0)

        stage_insert_mask = action_mask & (self.action_stage == self.STAGE_INSERT)
        insert_events = stage_insert_mask & socket_has_battery
        self.new_insert_event = insert_events
        if insert_events.any():
            ins_idx = torch.where(insert_events)[0]
            self.action_stage[ins_idx] = self.STAGE_RETURN
            self.active_battery_idx[ins_idx] = socket_battery_idx[ins_idx]
            working_ins_idx = ins_idx[socket_has_working[ins_idx]]
            if working_ins_idx.numel() > 0:
                self.lamp_afterglow_steps[working_ins_idx] = self.LAMP_AFTERGLOW_STEPS

            # Snap inserted battery to exact socket center so extraction is reliable.
            for ei in ins_idx.tolist():
                bat_id = int(socket_battery_idx[ei].item())
                raw_pose = self.batteries[bat_id].pose.raw_pose.clone()
                raw_pose[ei, 0] = self.socket_slot_pos[ei, 0]
                raw_pose[ei, 1] = self.socket_slot_pos[ei, 1]
                raw_pose[ei, 2] = self.socket_slot_pos[ei, 2]
                raw_pose[ei, 3:7] = self.battery_quat
                self.batteries[bat_id].pose = raw_pose
                zero_vel = torch.zeros((self.num_envs, 3), device=self.device)
                self.batteries[bat_id].set_linear_velocity(zero_vel)
                self.batteries[bat_id].set_angular_velocity(zero_vel)

        lamp_on = action_mask & (socket_has_working | (self.lamp_afterglow_steps > 0))
        lamp_off_pose = self.lamp_bulb_off.pose.raw_pose.clone()
        lamp_on_pose = self.lamp_bulb_on.pose.raw_pose.clone()
        lamp_off_pose[lamp_on, :3] = self.lamp_off_pos[lamp_on]
        lamp_off_pose[~lamp_on, :3] = self.lamp_on_pos[~lamp_on]
        lamp_on_pose[lamp_on, :3] = self.lamp_on_pos[lamp_on]
        lamp_on_pose[~lamp_on, :3] = self.lamp_off_pos[~lamp_on]
        self.lamp_bulb_off.pose = lamp_off_pose
        self.lamp_bulb_on.pose = lamp_on_pose

        tcp_pos = self.agent.tcp.pose.p
        tcp_xy = tcp_pos[:, :2]
        tcp_z = tcp_pos[:, 2]
        button_xy_dist = torch.linalg.norm(tcp_xy - self.button_xy, dim=1)
        raw_depth = self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
        depth = torch.clamp(raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
        depth = depth * (button_xy_dist <= self.BUTTON_PRESS_XY_RADIUS).float()
        self.button_press_depth = depth

        cap_pose = self.button_cap.pose.raw_pose.clone()
        cap_pose[:, 0:2] = self.button_xy
        cap_pose[:, 2] = self.button_cap_unpressed_z - depth
        cap_pose[:, 3:7] = self.button_cap_quat.repeat(cap_pose.shape[0], 1)
        self.button_cap.pose = cap_pose

        pressed = depth >= (self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO)
        released = depth <= (self.BUTTON_CAP_TRAVEL * self.BUTTON_RELEASE_READY_RATIO)
        self.press_ready = self.press_ready | ((~self.press_ready) & released & action_mask)
        self.new_button_press_event = pressed & self.press_ready & action_mask
        self.press_ready = self.press_ready & (~self.new_button_press_event)

        stage_return_mask = action_mask & (self.action_stage == self.STAGE_RETURN)
        if stage_return_mask.any():
            r_idx = torch.where(stage_return_mask)[0]
            active_idx = self.active_battery_idx[r_idx]
            valid = active_idx >= 0
            if valid.any():
                r_idx = r_idx[valid]
                active_idx = active_idx[valid]

                cur_pos = battery_pos[r_idx, active_idx]
                home_pos = self.tray_slot_positions[r_idx, active_idx]
                return_xy = torch.linalg.norm(cur_pos[:, :2] - home_pos[:, :2], dim=1)
                return_z = torch.abs(cur_pos[:, 2] - home_pos[:, 2])
                not_grasped = ~grasp_mask[r_idx, active_idx]
                not_in_socket = ~in_socket[r_idx, active_idx]
                returned = (
                    (return_xy <= self.SLOT_RETURN_XY_TOL)
                    & (return_z <= self.SLOT_RETURN_Z_TOL)
                    & not_grasped
                    & not_in_socket
                )
                if returned.any():
                    returned_idx = r_idx[returned]
                    self.new_return_event[returned_idx] = True
                    self.action_stage[returned_idx] = self.STAGE_CONFIRM

        stage_confirm_mask = action_mask & (self.action_stage == self.STAGE_CONFIRM)
        confirm_events = stage_confirm_mask & self.new_button_press_event
        self.new_confirm_event = confirm_events
        if confirm_events.any():
            c_idx = torch.where(confirm_events)[0]
            active_idx = self.active_battery_idx[c_idx]
            valid = active_idx >= 0
            if valid.any():
                c_idx = c_idx[valid]
                active_idx = active_idx[valid]
                already_checked = self.checked_mask[c_idx, active_idx]
                newly_checked = ~already_checked
                self.checked_mask[c_idx, active_idx] = True
                self.found_working_mask[c_idx, active_idx] = self.found_working_mask[c_idx, active_idx] | (
                    self.working_mask[c_idx, active_idx] & newly_checked
                )
                self.checked_count[c_idx] = self.checked_count[c_idx] + newly_checked.to(torch.int64)
                self.found_working_count[c_idx] = torch.sum(self.found_working_mask[c_idx], dim=1).to(torch.int64)
            self.action_stage[c_idx] = self.STAGE_INSERT
            self.active_battery_idx[c_idx] = -1

        self.found_working_count = torch.sum(self.found_working_mask, dim=1).to(torch.int64)

        # Success requires all checked batteries to be back near their home slots.
        # Use relaxed tolerances (3x stage transition thresholds) — battery is
        # clearly on the tray but may be slightly shifted after placement.
        SUCCESS_RETURN_XY_TOL = self.SLOT_RETURN_XY_TOL * 3.0
        SUCCESS_RETURN_Z_TOL = self.SLOT_RETURN_Z_TOL * 3.0
        all_returned = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for env_i in range(self.num_envs):
            # Only check active batteries (inactive ones are pre-marked checked but hidden).
            checked = self.checked_mask[env_i] & self.active_mask[env_i]
            if not checked.any():
                continue
            checked_ids = torch.where(checked)[0]
            for bat_id in checked_ids.tolist():
                bat_pos = self.batteries[bat_id].pose.p[env_i]
                home_pos = self.tray_slot_positions[env_i, bat_id]
                xy_dist = torch.linalg.norm(bat_pos[:2] - home_pos[:2])
                z_dist = torch.abs(bat_pos[2] - home_pos[2])
                if xy_dist > SUCCESS_RETURN_XY_TOL or z_dist > SUCCESS_RETURN_Z_TOL:
                    all_returned[env_i] = False
                    break

        all_checked = self.checked_count >= self.active_battery_count
        success = action_mask & (self.found_working_count >= self.target_working_count) & all_checked & all_returned

        self.obj_to_goal_pos = self.socket_slot_pos - self.agent.tcp.pose.p
        self.battery_raw_pose = battery_raw_pose
        self.socket_battery_idx = socket_battery_idx
        self.socket_has_working = socket_has_working

        return {
            "success": success,
            "action_mask": action_mask,
            "cue_mask": cue_mask,
            "action_stage": self.action_stage,
            "active_battery_idx": self.active_battery_idx,
            "socket_has_battery": socket_has_battery,
            "socket_battery_idx": socket_battery_idx,
            "socket_has_working": socket_has_working,
            "active_battery_count": self.active_battery_count,
            "checked_count": self.checked_count,
            "found_working_count": self.found_working_count,
            "target_working_count": self.target_working_count,
            "all_checked": all_checked,
            "new_insert_event": self.new_insert_event,
            "new_return_event": self.new_return_event,
            "new_confirm_event": self.new_confirm_event,
            "new_button_press_event": self.new_button_press_event,
            "button_press_depth": self.button_press_depth,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            battery_pose_flat = self.battery_raw_pose.reshape(self.num_envs, -1)
            obs.update(
                battery_pose=battery_pose_flat,
                tray_pose=self.tray.pose.raw_pose,
                socket_pose=self.socket_box.pose.raw_pose,
                button_base_pose=self.button_base.pose.raw_pose,
                button_cap_pose=self.button_cap.pose.raw_pose,
                lamp_off_pose=self.lamp_bulb_off.pose.raw_pose,
                lamp_on_pose=self.lamp_bulb_on.pose.raw_pose,
                action_stage=info["action_stage"],
                action_mask=info["action_mask"],
                active_mask=self.active_mask,
                oracle_info=self.oracle_info,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            if torch.is_tensor(success):
                success = success.to(dtype=torch.bool)
                if torch.is_tensor(terminated):
                    terminated = terminated.to(dtype=torch.bool) | success
                else:
                    terminated = bool(terminated) or bool(success.any().item())
            else:
                terminated = bool(terminated) or bool(success)
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        stage_insert = (info["action_stage"] == self.STAGE_INSERT).float()
        stage_return = (info["action_stage"] == self.STAGE_RETURN).float()
        stage_confirm = (info["action_stage"] == self.STAGE_CONFIRM).float()
        action_mask_f = info["action_mask"].float()

        battery_pos = torch.stack([bat.pose.p for bat in self.batteries], dim=1)
        dist_tcp_to_battery = torch.linalg.norm(battery_pos - tcp_pos.unsqueeze(1), dim=2)
        unchecked_mask = (~self.checked_mask).float()
        unchecked_dist = dist_tcp_to_battery + (1.0 - unchecked_mask) * 1e6
        min_unchecked_dist = torch.min(unchecked_dist, dim=1).values
        has_unchecked = (unchecked_mask.sum(dim=1) > 0).float()
        reach_unchecked_reward = (1 - torch.tanh(4.0 * min_unchecked_dist)) * has_unchecked

        grasp_mask = torch.stack([self.agent.is_grasping(bat) for bat in self.batteries], dim=1)
        has_grasp = grasp_mask.any(dim=1)
        grasp_idx = torch.argmax(grasp_mask.to(torch.int64), dim=1)
        grasped_battery_pos = battery_pos[torch.arange(self.num_envs, device=self.device), grasp_idx]
        grasped_to_socket_dist = torch.linalg.norm(grasped_battery_pos - self.socket_slot_pos, dim=1)
        carry_to_socket_reward = (1 - torch.tanh(4.0 * grasped_to_socket_dist)) * has_grasp.float()

        active_idx = torch.clamp(self.active_battery_idx, min=0)
        active_pos = battery_pos[torch.arange(self.num_envs, device=self.device), active_idx]
        active_home = self.tray_slot_positions[torch.arange(self.num_envs, device=self.device), active_idx]
        return_dist = torch.linalg.norm(active_pos - active_home, dim=1)
        return_reward = 1 - torch.tanh(6.0 * return_dist)

        tcp_to_button_dist = torch.linalg.norm(self.obj_to_goal_pos, dim=1)
        button_reach_reward = 1 - torch.tanh(8.0 * tcp_to_button_dist)
        button_press_reward = torch.clamp(info["button_press_depth"] / self.BUTTON_CAP_TRAVEL, min=0.0, max=1.0)

        wrong_phase_press = info["new_button_press_event"] & (~stage_confirm.bool())
        newly_found_working = (
            info["new_confirm_event"]
            & (self.active_battery_idx >= 0)
            & self.working_mask[
                torch.arange(self.num_envs, device=self.device),
                torch.clamp(self.active_battery_idx, min=0),
            ]
        )

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

        reward = (
            stage_insert * (1.2 * reach_unchecked_reward + 2.0 * carry_to_socket_reward)
            + stage_return * (3.0 * return_reward)
            + stage_confirm * (1.8 * button_reach_reward + 2.5 * button_press_reward)
            + 5.0 * info["new_insert_event"].float()
            + 6.0 * info["new_return_event"].float()
            + 7.0 * info["new_confirm_event"].float()
            + 12.0 * newly_found_working.float()
            - 2.0 * wrong_phase_press.float()
            - smooth_penalty
        )
        reward = reward * action_mask_f
        reward[info["success"]] = self.SUCCESS_BONUS

        self.reward_dict = {
            "stage_insert": stage_insert,
            "stage_return": stage_return,
            "stage_confirm": stage_confirm,
            "reach_unchecked_reward": reach_unchecked_reward,
            "carry_to_socket_reward": carry_to_socket_reward,
            "return_reward": return_reward,
            "button_reach_reward": button_reach_reward,
            "button_press_reward": button_press_reward,
            "new_insert_event": info["new_insert_event"].float(),
            "new_return_event": info["new_return_event"].float(),
            "new_confirm_event": info["new_confirm_event"].float(),
            "newly_found_working": newly_found_working.float(),
            "wrong_phase_press": wrong_phase_press.float(),
            "checked_count": info["checked_count"].float(),
            "found_working_count": info["found_working_count"].float(),
            "target_working_count": info["target_working_count"].float(),
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
        }
        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


@register_env("BatteriesCheckerHard-3-VLA-v0", max_episode_steps=1080)
class BatteriesChecker3VLAEnv(BatteriesCheckerVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 3
    WORKING_BATTERY_COUNT = 1


@register_env("BatteriesCheckerHard-6-VLA-v0", max_episode_steps=2160)
class BatteriesChecker6VLAEnv(BatteriesCheckerVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 6
    WORKING_BATTERY_COUNT = 3


@register_env("BatteriesCheckerHard-9-VLA-v0", max_episode_steps=3240)
class BatteriesChecker9VLAEnv(BatteriesCheckerVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 9
    WORKING_BATTERY_COUNT = 5


@register_env("BatteriesCheckerHard-12-VLA-v0", max_episode_steps=4320)
class BatteriesChecker12VLAEnv(BatteriesCheckerVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 12
    WORKING_BATTERY_COUNT = 7


@register_env("BatteriesCheckerHard-15-VLA-v0", max_episode_steps=4320)
class BatteriesChecker15VLAEnv(BatteriesCheckerVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 15
    WORKING_BATTERY_COUNT = 9
