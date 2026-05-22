"""Gather-and-recall VLA task: move cubes to a disc and remember a lamp flash color."""

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


class GatherAndRecallVLABaseEnv(BaseEnv):
    """Move cubes onto a target disc while remembering a brief lamp color flash.

    Cubes start in a cluster on one side of the table, with a target disc on the
    other side. The agent picks up cubes one by one and places them on the disc.
    While the agent is moving cubes (after the first cube lands on the disc but
    before the last), a signal lamp briefly flashes one of three colors: red,
    green, or blue. After all cubes are placed on the disc, the agent must press
    the button whose color matches the flash.

    Episode flow:
    1. MOVE phase: pick and place cubes onto the disc.
    2. During moving, the lamp flashes a random color once (briefly).
    3. PRESS phase: once all cubes are on the disc, press the matching button.

    Success (`success=True`):
    - All cubes detected on the disc AND the correct color button is pressed.

    Failure (`failed=True`):
    - A wrong color button is pressed after all cubes are placed.

    How to customize:
    - `N_CUBES` controls difficulty (more cubes = longer distraction, harder memory).
    - `FLASH_DURATION_STEPS` controls how long the lamp stays on ([min, max]).
    - `DISC_RADIUS` controls the target disc size.
    - `DISC_ON_THRESH` controls the XY distance threshold for cube-on-disc detection.
    """

    LANGUAGE_INSTRUCTION = (
        "Move all cubes onto the disc. A lamp will briefly flash "
        "while you work. After all cubes are placed, press the button matching the flash color."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    N_CUBES: int = 5
    CUBE_HALF_SIZE: float = 0.02

    DISC_RADIUS: float = 0.12
    DISC_HALF_HEIGHT: float = 0.003
    DISC_ON_THRESH: float = 0.10

    HEIGHT_OFFSET: float = 1000.0

    FLASH_DURATION_STEPS: List[int] = [8, 14]

    # Flash / button colors (red, green, blue)
    FLASH_COLORS = [
        np.array([255, 0, 0, 255], dtype=np.float32) / 255.0,
        np.array([0, 255, 0, 255], dtype=np.float32) / 255.0,
        np.array([0, 0, 255, 255], dtype=np.float32) / 255.0,
    ]
    # Cube colors must match button colors exactly.
    CUBE_COLORS = FLASH_COLORS

    BUTTON_BASE_HALF_SIZE = np.array([0.04, 0.04, 0.015], dtype=np.float32)
    BUTTON_CAP_RADIUS = 0.025
    BUTTON_CAP_HALF_HEIGHT = 0.014
    BUTTON_CAP_TRAVEL = BUTTON_CAP_HALF_HEIGHT
    BUTTON_PRESS_EVENT_RATIO = 0.35
    BUTTON_PRESS_XY_RADIUS = 0.04
    BUTTON_PRESS_Z_MARGIN = 0.03
    BUTTON_SPACING = 0.14

    # Scene layout
    DISC_X_MIN = 0.00
    DISC_X_MAX = 0.08
    DISC_Y_MIN = 0.10
    DISC_Y_MAX = 0.18
    BUTTON_X_OFFSET_FROM_DISC = -0.22
    CUBE_CLUSTER_X_OFFSET = -0.08
    CUBE_CLUSTER_CENTER_Y = -0.26
    CUBE_CLUSTER_SPACING_SCALE = 4.5
    LAMP_X_OFFSET_FROM_DISC = 0.24
    LAMP_Y_OFFSET_FROM_DISC = 0.06

    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012

    GRASP_THRESH = 0.05
    CUBE_VEL_THRESH = 0.15

    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.03
    QVEL_L2_COEF = 0.01
    SUCCESS_BONUS = 50.0
    FAILURE_PENALTY = 25.0

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_contact_count=2**21,
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
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        default_hidden_pose = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])
        n = self.num_envs
        d = self.device

        # ── Cubes (dynamic) ──────────────────────────────────────────────
        self.cubes = []
        for i in range(self.N_CUBES):
            color = self.CUBE_COLORS[i % len(self.CUBE_COLORS)]
            cube = actors.build_cube(
                self.scene,
                half_size=self.CUBE_HALF_SIZE,
                color=color,
                name=f"gather_cube_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALF_SIZE]),
            )
            self.cubes.append(cube)

        # ── Target disc (kinematic) ──────────────────────────────────────
        disc_builder = self.scene.create_actor_builder()
        disc_builder.add_cylinder_collision(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
        )
        disc_builder.add_cylinder_visual(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(
                base_color=np.array([160, 160, 170, 255], dtype=np.float32) / 255.0,
            ),
        )
        self.disc = _build_by_type(
            disc_builder,
            name="target_disc",
            body_type="kinematic",
            initial_pose=default_hidden_pose,
        )
        self.disc_quat = torch.tensor(
            euler2quat(0, np.pi / 2, 0),
            dtype=torch.float32,
            device=d,
        )

        # ── 3 colored buttons (red / green / blue, kinematic) ────────────
        self.button_bases = []
        self.button_caps = []
        self.button_cap_quat = torch.tensor(
            euler2quat(0, np.pi / 2, 0),
            dtype=torch.float32,
            device=d,
        )

        for i, color in enumerate(self.FLASH_COLORS):
            base_builder = self.scene.create_actor_builder()
            base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
            base_builder.add_box_visual(
                half_size=self.BUTTON_BASE_HALF_SIZE,
                material=sapien.render.RenderMaterial(
                    base_color=np.array([55, 64, 78, 255]) / 255.0,
                ),
            )
            base = _build_by_type(
                base_builder,
                name=f"btn_base_{i}",
                body_type="kinematic",
                initial_pose=default_hidden_pose,
            )
            self.button_bases.append(base)

            cap_builder = self.scene.create_actor_builder()
            cap_builder.add_cylinder_collision(
                radius=self.BUTTON_CAP_RADIUS,
                half_length=self.BUTTON_CAP_HALF_HEIGHT,
            )
            cap_builder.add_cylinder_visual(
                radius=self.BUTTON_CAP_RADIUS,
                half_length=self.BUTTON_CAP_HALF_HEIGHT,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            cap = _build_by_type(
                cap_builder,
                name=f"btn_cap_{i}",
                body_type="kinematic",
                initial_pose=default_hidden_pose,
            )
            self.button_caps.append(cap)

        # ── Signal lamp (one body/off-bulb, three colored on-bulbs) ──────
        self.lamp_bulbs_on = []
        self._extra_lamp_bodies = []
        self._extra_lamp_offs = []

        for i, color in enumerate(self.FLASH_COLORS):
            parts = shapes.build_color_switch_lamp(
                scene=self.scene,
                name=f"signal_lamp_{i}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=default_hidden_pose,
                base_radius=self.LAMP_BASE_RADIUS,
                base_half_height=self.LAMP_BASE_HALF_HEIGHT,
                stem_radius=self.LAMP_STEM_RADIUS,
                stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
                bulb_radius=self.LAMP_BULB_RADIUS,
                bulb_on_color=color,
            )
            shapes._set_actor_visual_rgba(
                parts["bulb_on"],
                color,
                emission_scale=20.0,
                remove_textures=True,
            )
            self.lamp_bulbs_on.append(parts["bulb_on"])

            if i == 0:
                self.lamp_body = parts["body"]
                self.lamp_bulb_off = parts["bulb_off"]
            else:
                self._extra_lamp_bodies.append(parts["body"])
                self._extra_lamp_offs.append(parts["bulb_off"])

        # ── State tensors ────────────────────────────────────────────────
        self.cubes_on_disc = torch.zeros((n, self.N_CUBES), dtype=torch.bool, device=d)
        self.flash_color = torch.zeros(n, dtype=torch.int64, device=d)
        self.flash_trigger_count = torch.zeros(n, dtype=torch.int64, device=d)
        self.flash_triggered = torch.zeros(n, dtype=torch.bool, device=d)
        self.flash_start_step = torch.zeros(n, dtype=torch.int64, device=d)
        self.flash_duration = torch.zeros(n, dtype=torch.int64, device=d)

        self.disc_xy = torch.zeros((n, 2), dtype=torch.float32, device=d)
        self.disc_place_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.lamp_visible_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.lamp_hidden_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)

        self.buttons_xy = torch.zeros((3, n, 2), dtype=torch.float32, device=d)
        self.button_top_z = torch.zeros(n, dtype=torch.float32, device=d)
        self.button_cap_unpressed_z = torch.zeros(n, dtype=torch.float32, device=d)

        self.pressed_button = torch.full((n,), -1, dtype=torch.int64, device=d)
        self.failed = torch.zeros(n, dtype=torch.bool, device=d)
        self.success_flag = torch.zeros(n, dtype=torch.bool, device=d)

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

            # ── Reset state ──────────────────────────────────────────────
            self.cubes_on_disc[env_idx] = False
            self.flash_triggered[env_idx] = False
            self.flash_start_step[env_idx] = 0
            self.pressed_button[env_idx] = -1
            self.failed[env_idx] = False
            self.success_flag[env_idx] = False

            self.flash_color[env_idx] = torch.randint(
                0,
                3,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )
            # For N_CUBES=1, randint(1, 1, ...) is invalid; clamp upper bound.
            flash_trigger_hi = max(2, self.N_CUBES)
            self.flash_trigger_count[env_idx] = torch.randint(
                1,
                flash_trigger_hi,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )
            lo, hi = self.FLASH_DURATION_STEPS
            self.flash_duration[env_idx] = torch.randint(
                lo,
                hi + 1,
                (b,),
                device=self.device,
                dtype=torch.int64,
            )

            default_q = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                device=self.device,
            ).repeat(b, 1)

            # ── Position disc (far side, farther from robot) ─────────────
            disc_xyz = torch.zeros((b, 3), device=self.device)
            disc_xyz[:, 0] = torch.rand(b, device=self.device) * (self.DISC_X_MAX - self.DISC_X_MIN) + self.DISC_X_MIN
            disc_xyz[:, 1] = torch.rand(b, device=self.device) * (self.DISC_Y_MAX - self.DISC_Y_MIN) + self.DISC_Y_MIN
            disc_xyz[:, 2] = self.DISC_HALF_HEIGHT

            disc_q = self.disc_quat.unsqueeze(0).repeat(b, 1)
            self.disc.set_pose(Pose.create_from_pq(p=disc_xyz, q=disc_q))
            self.disc_xy[env_idx] = disc_xyz[:, :2]
            self.disc_place_pos[env_idx, :2] = disc_xyz[:, :2]
            self.disc_place_pos[env_idx, 2] = self.DISC_HALF_HEIGHT * 2 + self.CUBE_HALF_SIZE

            # ── Position cubes (cluster away from button row) ────────────
            cube_center_x = disc_xyz[:, 0] + self.CUBE_CLUSTER_X_OFFSET
            cube_center_y = self.CUBE_CLUSTER_CENTER_Y
            spacing = self.CUBE_HALF_SIZE * self.CUBE_CLUSTER_SPACING_SCALE
            n_rows = (self.N_CUBES + 2) // 3
            row_center = (n_rows - 1) / 2.0
            for i in range(self.N_CUBES):
                row = i // 3
                col = i % 3
                n_in_row = min(3, self.N_CUBES - row * 3)
                col_center = (n_in_row - 1) / 2.0

                cube_xyz = torch.zeros((b, 3), device=self.device)
                # Orient the cluster primarily along OX (was along OY).
                cube_xyz[:, 0] = cube_center_x + (col - col_center) * spacing
                cube_xyz[:, 1] = cube_center_y + (row - row_center) * spacing
                cube_xyz[:, 0] += (torch.rand(b, device=self.device) - 0.5) * 0.01
                cube_xyz[:, 1] += (torch.rand(b, device=self.device) - 0.5) * 0.01
                cube_xyz[:, 2] = self.CUBE_HALF_SIZE

                self.cubes[i].set_pose(
                    Pose.create_from_pq(p=cube_xyz, q=default_q),
                )
                lin_vel = self.cubes[i].linear_velocity.clone()
                ang_vel = self.cubes[i].angular_velocity.clone()
                lin_vel[env_idx] = 0
                ang_vel[env_idx] = 0
                self.cubes[i].set_linear_velocity(lin_vel)
                self.cubes[i].set_angular_velocity(ang_vel)

            # ── Position buttons (closer to robot than disc) ─────────────
            cap_q = self.button_cap_quat.unsqueeze(0).repeat(b, 1)
            base_z = float(self.BUTTON_BASE_HALF_SIZE[2])
            unpressed_z = base_z * 2.0 + self.BUTTON_CAP_HALF_HEIGHT
            self.button_cap_unpressed_z[env_idx] = unpressed_z
            self.button_top_z[env_idx] = unpressed_z + self.BUTTON_CAP_HALF_HEIGHT

            btn_base_x = disc_xyz[:, 0] + self.BUTTON_X_OFFSET_FROM_DISC
            btn_base_y = disc_xyz[:, 1]

            for btn_idx in range(3):
                btn_xyz = torch.zeros((b, 3), device=self.device)
                btn_xyz[:, 0] = btn_base_x
                btn_xyz[:, 1] = btn_base_y + (btn_idx - 1) * self.BUTTON_SPACING
                btn_xyz[:, 2] = base_z

                self.button_bases[btn_idx].set_pose(
                    Pose.create_from_pq(p=btn_xyz, q=default_q),
                )
                cap_xyz = btn_xyz.clone()
                cap_xyz[:, 2] = unpressed_z
                self.button_caps[btn_idx].set_pose(
                    Pose.create_from_pq(p=cap_xyz, q=cap_q),
                )
                self.buttons_xy[btn_idx, env_idx] = btn_xyz[:, :2]

            # ── Position lamp (farther from disc and robot) ─────────────
            lamp_xyz = torch.zeros((b, 3), device=self.device)
            lamp_xyz[:, 0] = disc_xyz[:, 0] + self.LAMP_X_OFFSET_FROM_DISC
            lamp_xyz[:, 1] = disc_xyz[:, 1] + self.LAMP_Y_OFFSET_FROM_DISC
            lamp_xyz[:, 2] = 0.0

            hidden_xyz = lamp_xyz.clone()
            hidden_xyz[:, 2] += self.HEIGHT_OFFSET

            self.lamp_visible_pos[env_idx] = lamp_xyz
            self.lamp_hidden_pos[env_idx] = hidden_xyz

            self.lamp_body.set_pose(
                Pose.create_from_pq(p=lamp_xyz, q=default_q),
            )
            self.lamp_bulb_off.set_pose(
                Pose.create_from_pq(p=lamp_xyz, q=default_q),
            )
            for on_bulb in self.lamp_bulbs_on:
                on_bulb.set_pose(
                    Pose.create_from_pq(p=hidden_xyz, q=default_q),
                )
            for body in self._extra_lamp_bodies:
                body.set_pose(
                    Pose.create_from_pq(p=hidden_xyz, q=default_q),
                )
            for off in self._extra_lamp_offs:
                off.set_pose(
                    Pose.create_from_pq(p=hidden_xyz, q=default_q),
                )

            # ── Oracle info (flash colour index) ─────────────────────────
            self.oracle_info = self.flash_color.to(torch.uint8)

            # ── Reset robot ──────────────────────────────────────────────
            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04],
                )
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
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        # ── Cube-on-disc detection (sticky + current) ────────────────────
        current_cubes_on_disc = torch.zeros(
            (self.num_envs, self.N_CUBES),
            dtype=torch.bool,
            device=self.device,
        )
        for i in range(self.N_CUBES):
            cube_pos = self.cubes[i].pose.p
            xy_dist = torch.linalg.norm(
                cube_pos[:, :2] - self.disc_xy,
                dim=1,
            )
            z_on_table = cube_pos[:, 2] < 0.5
            cube_vel = torch.linalg.norm(
                self.cubes[i].linear_velocity,
                dim=1,
            )
            on_disc = (xy_dist < self.DISC_ON_THRESH) & z_on_table & (cube_vel < self.CUBE_VEL_THRESH)
            current_cubes_on_disc[:, i] = on_disc
            self.cubes_on_disc[:, i] = self.cubes_on_disc[:, i] | on_disc

        n_on_disc = self.cubes_on_disc.sum(dim=1)
        all_on_disc = current_cubes_on_disc.all(dim=1)

        # ── Flash triggering ─────────────────────────────────────────────
        trigger = (~self.flash_triggered) & (n_on_disc >= self.flash_trigger_count)
        self.flash_triggered = self.flash_triggered | trigger
        self.flash_start_step[trigger] = elapsed[trigger]

        flash_elapsed = elapsed - self.flash_start_step
        flash_active = self.flash_triggered & (flash_elapsed >= 0) & (flash_elapsed < self.flash_duration)

        # ── Lamp control ─────────────────────────────────────────────────
        off_pose = self.lamp_bulb_off.pose.raw_pose.clone()
        off_pose[flash_active, :3] = self.lamp_hidden_pos[flash_active]
        off_pose[~flash_active, :3] = self.lamp_visible_pos[~flash_active]
        self.lamp_bulb_off.pose = off_pose

        for color_idx in range(3):
            color_mask = flash_active & (self.flash_color == color_idx)
            on_pose = self.lamp_bulbs_on[color_idx].pose.raw_pose.clone()
            on_pose[color_mask, :3] = self.lamp_visible_pos[color_mask]
            on_pose[~color_mask, :3] = self.lamp_hidden_pos[~color_mask]
            self.lamp_bulbs_on[color_idx].pose = on_pose

        # ── Button press detection ───────────────────────────────────────
        tcp_pos = self.agent.tcp.pose.p
        tcp_xy = tcp_pos[:, :2]
        tcp_z = tcp_pos[:, 2]

        for btn_idx in range(3):
            btn_xy = self.buttons_xy[btn_idx]
            xy_dist = torch.linalg.norm(tcp_xy - btn_xy, dim=1)
            raw_depth = self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
            depth = torch.clamp(raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
            depth = depth * (xy_dist < self.BUTTON_PRESS_XY_RADIUS).float()

            # Visual cap depression
            cap_pose = self.button_caps[btn_idx].pose.raw_pose.clone()
            cap_pose[:, 0:2] = btn_xy
            cap_pose[:, 2] = self.button_cap_unpressed_z - depth
            cap_pose[:, 3:7] = self.button_cap_quat.repeat(
                cap_pose.shape[0],
                1,
            )
            self.button_caps[btn_idx].pose = cap_pose

            # Detect press (only after all cubes placed, first press only)
            pressed = depth >= (self.BUTTON_CAP_TRAVEL * self.BUTTON_PRESS_EVENT_RATIO)
            new_press = pressed & all_on_disc & (self.pressed_button == -1)
            self.pressed_button[new_press] = btn_idx

        # ── Success / failure ────────────────────────────────────────────
        button_pressed_mask = self.pressed_button >= 0
        correct = button_pressed_mask & (self.pressed_button == self.flash_color) & all_on_disc
        wrong = button_pressed_mask & (self.pressed_button != self.flash_color)

        self.success_flag = self.success_flag | correct
        self.failed = self.failed | wrong
        success = self.success_flag & all_on_disc

        # ── Reaching target (obj_to_goal_pos) ────────────────────────────
        obj_to_goal = torch.zeros_like(tcp_pos)

        # Move phase: reach toward nearest unplaced cube or disc
        min_dist = torch.full(
            (self.num_envs,),
            float("inf"),
            device=self.device,
        )
        holding_any = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        for i in range(self.N_CUBES):
            unplaced = ~self.cubes_on_disc[:, i] & ~all_on_disc
            if not unplaced.any():
                continue
            cube_pos = self.cubes[i].pose.p
            diff = cube_pos - tcp_pos
            dist = torch.linalg.norm(diff, dim=1)

            grasped = unplaced & (dist < self.GRASP_THRESH)
            holding_any = holding_any | grasped

            reach_mask = unplaced & ~grasped
            closer = reach_mask & (dist < min_dist)
            obj_to_goal[closer] = diff[closer]
            min_dist = torch.where(closer, dist, min_dist)

        # If holding a cube, target the disc instead
        if holding_any.any():
            disc_target = self.disc_place_pos - tcp_pos
            obj_to_goal[holding_any] = disc_target[holding_any]

        # Press phase: target the correct button
        for btn_idx in range(3):
            btn_mask = all_on_disc & (self.flash_color == btn_idx) & ~button_pressed_mask
            if btn_mask.any():
                btn_pos = torch.zeros(self.num_envs, 3, device=self.device)
                btn_pos[:, :2] = self.buttons_xy[btn_idx]
                btn_pos[:, 2] = self.button_top_z + 0.005
                obj_to_goal[btn_mask] = (btn_pos - tcp_pos)[btn_mask]

        self.obj_to_goal_pos = obj_to_goal

        return {
            "success": success,
            "failed": self.failed,
            "n_on_disc": n_on_disc,
            "all_on_disc": all_on_disc,
            "flash_active": flash_active,
            "flash_triggered": self.flash_triggered,
            "flash_color": self.flash_color,
            "pressed_button": self.pressed_button,
            "obj_to_goal_pos": obj_to_goal,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    # ------------------------------------------------------------------
    # Observation extras
    # ------------------------------------------------------------------
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_to_goal_pos=self.obj_to_goal_pos,
                oracle_info=self.oracle_info,
                disc_pose=self.disc.pose.raw_pose,
                cubes_on_disc=self.cubes_on_disc,
                n_on_disc=info["n_on_disc"],
                all_on_disc=info["all_on_disc"],
                flash_active=info["flash_active"],
                flash_color=self.flash_color,
                pressed_button=self.pressed_button,
            )
            for i in range(self.N_CUBES):
                obs[f"cube_{i}_pose"] = self.cubes[i].pose.raw_pose
        return obs

    # ------------------------------------------------------------------
    # Step override (terminate on success or failure)
    # ------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if isinstance(info, dict):
            success = info.get("success", None)
            failed = info.get("failed", None)

            if torch.is_tensor(terminated):
                terminated = terminated.to(dtype=torch.bool)
                if torch.is_tensor(success):
                    terminated = terminated | success.to(dtype=torch.bool)
                if torch.is_tensor(failed):
                    terminated = terminated | failed.to(dtype=torch.bool)
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

    # ------------------------------------------------------------------
    # Dense reward
    # ------------------------------------------------------------------
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        n_on_disc = info["n_on_disc"].float()
        all_on_disc_f = info["all_on_disc"].float()

        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, dim=1)
        reaching_reward = 1 - torch.tanh(8.0 * tcp_to_obj_dist)

        # Progress reward: fraction of cubes placed
        place_progress = n_on_disc / self.N_CUBES

        # Button press reward (correct button depth)
        button_press_reward = torch.zeros(self.num_envs, device=self.device)
        tcp_xy = self.agent.tcp.pose.p[:, :2]
        tcp_z = self.agent.tcp.pose.p[:, 2]
        for btn_idx in range(3):
            btn_xy = self.buttons_xy[btn_idx]
            xy_dist = torch.linalg.norm(tcp_xy - btn_xy, dim=1)
            raw_depth = self.button_top_z + self.BUTTON_PRESS_Z_MARGIN - tcp_z
            depth = torch.clamp(raw_depth, min=0.0, max=self.BUTTON_CAP_TRAVEL)
            depth = depth * (xy_dist < self.BUTTON_PRESS_XY_RADIUS).float()
            depth_norm = torch.clamp(
                depth / self.BUTTON_CAP_TRAVEL,
                min=0.0,
                max=1.0,
            )
            correct_mask = (self.flash_color == btn_idx) & info["all_on_disc"]
            button_press_reward += correct_mask.float() * depth_norm

        # Smoothness penalty
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)

        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, dim=1)
        delta_action_l2 = torch.linalg.norm(delta_action, dim=1)
        qvel_l2 = torch.linalg.norm(
            self.agent.robot.get_qvel()[..., :-2],
            dim=1,
        )
        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        reward = (
            2.0 * reaching_reward
            + 10.0 * place_progress
            + 3.0 * all_on_disc_f * reaching_reward
            + 5.0 * all_on_disc_f * button_press_reward
            - smooth_penalty
        )

        reward[info["failed"]] = -self.FAILURE_PENALTY
        reward[info["success"]] = self.SUCCESS_BONUS

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "place_progress": place_progress,
            "n_on_disc": n_on_disc,
            "button_press_reward": button_press_reward,
            "smooth_penalty": smooth_penalty,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
        }

        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


# ── Standard variants ────────────────────────────────────────────────────
@register_env("GatherAndRecall1-VLA-v0", max_episode_steps=200)
class GatherAndRecall1VLAEnv(GatherAndRecallVLABaseEnv):
    N_CUBES = 1


@register_env("GatherAndRecall3-VLA-v0", max_episode_steps=400)
class GatherAndRecall3VLAEnv(GatherAndRecallVLABaseEnv):
    N_CUBES = 3


@register_env("GatherAndRecall5-VLA-v0", max_episode_steps=600)
class GatherAndRecall5VLAEnv(GatherAndRecallVLABaseEnv):
    N_CUBES = 5


@register_env("GatherAndRecall7-VLA-v0", max_episode_steps=800)
class GatherAndRecall7VLAEnv(GatherAndRecallVLABaseEnv):
    N_CUBES = 7


@register_env("GatherAndRecall9-VLA-v0", max_episode_steps=1000)
class GatherAndRecall9VLAEnv(GatherAndRecallVLABaseEnv):
    N_CUBES = 9
