"""Bunch-of-colors memory tasks for the VLA benchmark."""

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


class BunchOfColorsVLABaseEnv(BaseEnv):
    """Remember a set of colors shown at once, then recover that set later.

    During the cue, several target colors are shown simultaneously. After a
    delay, all cubes reappear in randomized positions and the robot must touch
    every cube that belonged to the original set while avoiding distractors.
    Unlike sequence tasks, order does not matter here: only set membership does.

    Episode flow:
    - A target subset of colors is displayed together.
    - All cubes disappear for a short memory delay.
    - All cubes return and the robot starts selecting targets.

    Success (`success=True`):
    - The robot must touch all target colors and avoid wrong selections.

    How to customize:
    - `SEQUENCE_LENGTH` controls how many target colors the agent must remember.
    - `COLORS` controls how many total color choices exist in the scene.
    - `CUE_PHASE_STEPS` controls how long the target set is visible.
    - `EMPTY_PHASE_STEPS` controls the length of the memory gap before action.
    - `GOAL_THRESH` controls how close the tool center point must get for a cube
      touch to count.
    - `CUBE_HALFSIZE` changes cube size, which also affects spacing and contact
      geometry.
    """

    LANGUAGE_INSTRUCTION = (
        "Observe which colored cubes appear during the cue, wait, then touch all of them in any order "
        "and press the center button."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    COLORS = 9
    GOAL_THRESH = 0.05
    CUBE_HALFSIZE = 0.02
    SEQUENCE_LENGTH = 5

    CUE_PHASE_STEPS: List[int] = [1, 5]
    EMPTY_PHASE_STEPS: List[int] = [1, 5]

    COLOR_MAPPING = {
        0: ("Red", [255, 0, 0, 255]),
        1: ("Lime", [0, 255, 0, 255]),
        2: ("Blue", [0, 0, 255, 255]),
        3: ("Yellow", [255, 255, 0, 255]),
        4: ("Magenta", [255, 0, 255, 255]),
        5: ("Cyan", [0, 255, 255, 255]),
        6: ("Maroon", [128, 0, 0, 255]),
        7: ("Olive", [255, 128, 0, 255]),
        8: ("Teal", [0, 128, 128, 255]),
    }

    ACTION_L2_COEF = 0.0
    ACTION_DELTA_L2_COEF = 0.0
    QVEL_L2_COEF = 0.0
    CUBE_DISPLACEMENT_PENALTY_COEF = 20.0
    MAX_ALLOWED_CUBE_DISPLACEMENT = 0.06

    BUTTON_BASE_HALF_SIZE = np.array([0.065, 0.065, 0.015], dtype=np.float32)
    BUTTON_CAP_RADIUS = 0.03
    BUTTON_CAP_HALF_HEIGHT = 0.014
    BUTTON_CAP_TRAVEL = BUTTON_CAP_HALF_HEIGHT
    BUTTON_PRESS_EVENT_RATIO = 0.35
    BUTTON_RELEASE_READY_RATIO = 0.2
    BUTTON_PRESS_XY_RADIUS = 0.065
    BUTTON_PRESS_Z_MARGIN = 0.03
    BUTTON_HIDDEN_Z = 1000.0
    REQUIRED_LIFT_HEIGHT = 0.1
    LIFT_CONFIRM_TOL = 0.015
    BUTTON_X_SHIFT_TOWARD_ROBOT = -0.04

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.color_dict = {k: np.array(v[1]) / 255.0 for k, v in list(self.COLOR_MAPPING.items())[: self.COLORS]}
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.initial_poses = {}
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
        pose = sapien_utils.look_at([0.5, 1, 1], [-0.3, 0, 0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        self.cubes = {}
        for key, color in self.color_dict.items():
            self.cubes[key] = actors.build_cube(
                self.scene,
                half_size=self.CUBE_HALFSIZE,
                color=color,
                name=f"cube_{key}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            )

        default_initial_pose = sapien.Pose(p=[0.0, 0.0, self.BUTTON_HIDDEN_Z])

        base_builder = self.scene.create_actor_builder()
        base_builder.add_box_collision(half_size=self.BUTTON_BASE_HALF_SIZE)
        base_builder.add_box_visual(
            half_size=self.BUTTON_BASE_HALF_SIZE,
            material=sapien.render.RenderMaterial(base_color=np.array([55, 64, 78, 255]) / 255.0),
        )
        self.button_base = _build_by_type(
            base_builder,
            name="center_button_base",
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
            name="center_button_cap",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

    def _ensure_phase_buffers(self, env_idx: torch.Tensor):
        target_size = int(env_idx.max().item()) + 1
        if not hasattr(self, "cue_steps_per_env") or self.cue_steps_per_env is None:
            self.cue_steps_per_env = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            self.empty_steps_per_env = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            return
        current_size = self.cue_steps_per_env.shape[0]
        if target_size > current_size:
            pad = target_size - current_size
            self.cue_steps_per_env = torch.cat(
                [
                    self.cue_steps_per_env,
                    torch.zeros(pad, dtype=torch.int64, device=self.device),
                ]
            )
            self.empty_steps_per_env = torch.cat(
                [
                    self.empty_steps_per_env,
                    torch.zeros(pad, dtype=torch.int64, device=self.device),
                ]
            )

    def _ensure_button_buffers(self, env_idx: torch.Tensor):
        target_size = int(env_idx.max().item()) + 1
        if not hasattr(self, "button_xy") or self.button_xy is None:
            self.button_xy = torch.zeros((target_size, 2), dtype=torch.float32, device=self.device)
            self.button_base_z = torch.zeros(target_size, dtype=torch.float32, device=self.device)
            self.button_cap_unpressed_z = torch.zeros(target_size, dtype=torch.float32, device=self.device)
            self.button_top_z = torch.zeros(target_size, dtype=torch.float32, device=self.device)
            self.button_press_depth = torch.zeros(target_size, dtype=torch.float32, device=self.device)
            self.button_pressed = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.button_pressable = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.press_ready = torch.ones(target_size, dtype=torch.bool, device=self.device)
            self.pending_press = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.new_raw_press_event = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.new_release_event = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.new_press_event = torch.zeros(target_size, dtype=torch.bool, device=self.device)
            self.press_start_tcp_z = torch.zeros(target_size, dtype=torch.float32, device=self.device)
            self.button_press_count = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            self.button_cap_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=self.device)
            return

        current_size = self.button_xy.shape[0]
        if target_size > current_size:
            pad = target_size - current_size
            self.button_xy = torch.cat(
                [
                    self.button_xy,
                    torch.zeros((pad, 2), dtype=torch.float32, device=self.device),
                ]
            )
            self.button_base_z = torch.cat(
                [
                    self.button_base_z,
                    torch.zeros(pad, dtype=torch.float32, device=self.device),
                ]
            )
            self.button_cap_unpressed_z = torch.cat(
                [
                    self.button_cap_unpressed_z,
                    torch.zeros(pad, dtype=torch.float32, device=self.device),
                ]
            )
            self.button_top_z = torch.cat(
                [
                    self.button_top_z,
                    torch.zeros(pad, dtype=torch.float32, device=self.device),
                ]
            )
            self.button_press_depth = torch.cat(
                [
                    self.button_press_depth,
                    torch.zeros(pad, dtype=torch.float32, device=self.device),
                ]
            )
            self.button_pressed = torch.cat(
                [
                    self.button_pressed,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.button_pressable = torch.cat(
                [
                    self.button_pressable,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.press_ready = torch.cat(
                [
                    self.press_ready,
                    torch.ones(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.pending_press = torch.cat(
                [
                    self.pending_press,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.new_raw_press_event = torch.cat(
                [
                    self.new_raw_press_event,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.new_release_event = torch.cat(
                [
                    self.new_release_event,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.new_press_event = torch.cat(
                [
                    self.new_press_event,
                    torch.zeros(pad, dtype=torch.bool, device=self.device),
                ]
            )
            self.press_start_tcp_z = torch.cat(
                [
                    self.press_start_tcp_z,
                    torch.zeros(pad, dtype=torch.float32, device=self.device),
                ]
            )
            self.button_press_count = torch.cat(
                [
                    self.button_press_count,
                    torch.zeros(pad, dtype=torch.int64, device=self.device),
                ]
            )

        if not hasattr(self, "button_cap_quat") or self.button_cap_quat is None:
            self.button_cap_quat = torch.tensor(euler2quat(0, np.pi / 2, 0), dtype=torch.float32, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            self.touched_cubes = torch.zeros(
                (b, len(self.color_dict)),
                dtype=torch.bool,
                device=self.device,
            )
            self.initial_poses = {}

            all_colors = list(self.color_dict.keys())
            sequence_indices = self._batched_episode_rng.choice(
                all_colors,
                size=self.SEQUENCE_LENGTH,
                replace=False,
            )
            self.true_color_indices = torch.from_numpy(sequence_indices).to(
                device=self.device,
                dtype=torch.uint8,
            )

            xyz_initial = torch.zeros((b, 3))
            self.center_pose = xyz_initial.clone()
            self.center_pose[..., 2] = self.CUBE_HALFSIZE
            self.center_pose = self.center_pose[0].unsqueeze(0)

            for key in self.color_dict:
                xyz_cube = xyz_initial.clone()
                if self.COLORS != 3:
                    curvature_scale = 1.15
                    angle = curvature_scale * np.pi * (key - (len(self.color_dict) // 2)) / len(self.color_dict)
                    radius = 0.3
                    xyz_cube[..., 0] = radius * np.cos(angle) - 0.25
                    xyz_cube[..., 1] = radius * np.sin(angle)
                    if self.COLORS in [5, 9]:
                        y_spread = 0.018
                        xyz_cube[..., 1] += (key - (len(self.color_dict) // 2)) * y_spread
                else:
                    y_spread = 0.07
                    xyz_cube[..., 1] += (key - (len(self.color_dict) // 2)) * y_spread
                xyz_cube[..., 2] = self.CUBE_HALFSIZE
                self.cubes[key].set_pose(Pose.create_from_pq(p=xyz_cube, q=[1, 0, 0, 0]))
                self.initial_poses[key] = xyz_cube.clone()

            with torch.device(self.device):
                min_distance = self.CUBE_HALFSIZE * 3.0
                noise_scale = self.CUBE_HALFSIZE * 0.5
                max_attempts = 50
                for env_i in range(b):
                    positions = [self.initial_poses[key][env_i].clone() for key in self.initial_poses]
                    for i in range(len(positions)):
                        attempt = 0
                        while attempt < max_attempts:
                            noise = torch.randn(2, device=self.device) * noise_scale
                            new_pos = positions[i].clone()
                            new_pos[:2] += noise
                            valid = all(torch.norm(new_pos[:2] - positions[j][:2]) >= min_distance for j in range(i))
                            if valid:
                                positions[i] = new_pos
                                break
                            attempt += 1
                    shuffled_indices = torch.randperm(len(positions))
                    shuffled_positions = [positions[idx] for idx in shuffled_indices]
                    for key, new_pos in zip(self.initial_poses.keys(), shuffled_positions):
                        self.initial_poses[key][env_i] = new_pos
                        current_pose = self.cubes[key].pose.raw_pose.clone()
                        current_pose[env_i, :3] = new_pos
                        current_pose[env_i, 3:7] = torch.tensor(
                            [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=current_pose.dtype
                        )
                        self.cubes[key].pose = current_pose
                        lin_vel = self.cubes[key].linear_velocity.clone()
                        ang_vel = self.cubes[key].angular_velocity.clone()
                        lin_vel[env_i] = 0
                        ang_vel[env_i] = 0
                        self.cubes[key].set_linear_velocity(lin_vel)
                        self.cubes[key].set_angular_velocity(ang_vel)

            self.initial_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes}
            self.oracle_info = self.true_color_indices

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

            self._ensure_phase_buffers(env_idx)
            cue_lo, cue_hi = self.CUE_PHASE_STEPS
            empty_lo, empty_hi = self.EMPTY_PHASE_STEPS
            self.cue_steps_per_env[env_idx] = torch.randint(
                low=cue_lo,
                high=cue_hi + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )
            self.empty_steps_per_env[env_idx] = torch.randint(
                low=empty_lo,
                high=empty_hi + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )

            self._ensure_button_buffers(env_idx)
            cube_xy_stack = torch.stack(
                [self.initial_poses[key][:, :2] for key in self.color_dict],
                dim=1,
            )
            button_xy = cube_xy_stack.mean(dim=1)
            button_xy[:, 0] += self.BUTTON_X_SHIFT_TOWARD_ROBOT
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
            self.pending_press[env_idx] = False
            self.new_raw_press_event[env_idx] = False
            self.new_release_event[env_idx] = False
            self.new_press_event[env_idx] = False
            self.press_start_tcp_z[env_idx] = 0.0
            self.button_press_count[env_idx] = 0

            self.button_base.set_pose(Pose.create_from_pq(p=button_base_xyz, q=button_base_q))
            self.button_cap.set_pose(Pose.create_from_pq(p=button_cap_xyz, q=button_cap_q))

    def evaluate(self):
        self.original_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes}

        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        cue_end = self.cue_steps_per_env
        empty_end = cue_end + self.empty_steps_per_env

        show_initial_cubes = elapsed < cue_end
        empty_table = (elapsed >= cue_end) & (elapsed < empty_end)
        show_all_cubes = elapsed >= empty_end
        stabilize_mask = show_all_cubes & (elapsed <= (empty_end + 4))
        self.active_phase = show_all_cubes

        hidden_shapes_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.color_dict}
        b_ = next(iter(hidden_shapes_poses.values())).shape[0]

        keys = torch.tensor(list(self.color_dict.keys()), device=self.device)
        angles = torch.pi * (keys - (len(self.color_dict) // 2)) / len(self.color_dict)
        radius = 0.22

        cos_angles = torch.cos(angles).to(device=self.device)
        sin_angles = torch.sin(angles).to(device=self.device)

        offsets = torch.stack(
            [
                radius * cos_angles,
                radius * sin_angles,
                torch.zeros_like(keys, device=self.device, dtype=torch.float32),
            ],
            dim=1,
        )

        y_adjustments = (keys - (len(self.color_dict) // 2)) * 0.015
        offsets[:, 1] -= y_adjustments

        is_target_cubes = torch.stack(
            [
                torch.tensor(
                    [key in self.true_color_indices[i] for i in range(self.true_color_indices.shape[0])],
                    device=self.device,
                )
                for key in self.color_dict
            ]
        )

        center_pose_expanded = self.center_pose.repeat(b_, 1)
        show_initial_expanded = show_initial_cubes.unsqueeze(-1)
        empty_table_expanded = empty_table.unsqueeze(-1)

        for key in self.color_dict:
            mask_target = is_target_cubes[key]
            new_pos = torch.where(
                mask_target.unsqueeze(-1) & show_initial_expanded,
                center_pose_expanded + offsets[key],
                torch.where(
                    empty_table_expanded | (~mask_target.unsqueeze(-1) & show_initial_expanded),
                    torch.tensor([0, 0, self.BUTTON_HIDDEN_Z], device=self.device),
                    self.original_poses[key][..., :3],
                ),
            )
            hidden_shapes_poses[key][..., :3] = new_pos
            if bool(stabilize_mask.any().item()):
                hidden_shapes_poses[key][stabilize_mask, :3] = self.initial_poses[key][stabilize_mask, :3]
                hidden_shapes_poses[key][stabilize_mask, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device, dtype=hidden_shapes_poses[key].dtype
                )
            self.cubes[key].pose = hidden_shapes_poses[key]
            if bool(stabilize_mask.any().item()):
                lin_vel = self.cubes[key].linear_velocity.clone()
                ang_vel = self.cubes[key].angular_velocity.clone()
                lin_vel[stabilize_mask] = 0
                ang_vel[stabilize_mask] = 0
                self.cubes[key].set_linear_velocity(lin_vel)
                self.cubes[key].set_angular_velocity(ang_vel)

        sequence_cubes_mask = torch.zeros(
            (b_, len(self.color_dict)),
            dtype=torch.bool,
            device=self.device,
        )
        sequence_cubes_mask.scatter_(1, self.true_color_indices.long(), True)
        self.sequence_cubes_mask = sequence_cubes_mask

        tcp_pos = self.agent.tcp.pose.p
        self.current_touches = {}
        cube_touch_pos = None
        for key in self.color_dict:
            cube_pos = self.cubes[key].pose.p
            cube_touch_pos = Pose.create_from_pq(
                p=cube_pos + torch.tensor([0, 0, self.CUBE_HALFSIZE + 0.005], device=self.device),
            )
            distance = torch.norm(tcp_pos - cube_touch_pos.p, dim=-1)
            touch_mask = (distance < self.CUBE_HALFSIZE) * show_all_cubes
            touch_mask *= self.agent.is_static(0.2)
            self.current_touches[key] = touch_mask
            self.touched_cubes[:, key] |= touch_mask

        all_cubes_from_sequence_is_touched = torch.eq(
            self.touched_cubes,
            sequence_cubes_mask,
        ).all(1)
        self.all_cubes_from_sequence_is_touched = all_cubes_from_sequence_is_touched

        no_one_cube_not_from_sequence_is_touched = ~(self.touched_cubes & ~sequence_cubes_mask).any(1)
        self.no_one_cube_not_from_sequence_is_touched = no_one_cube_not_from_sequence_is_touched

        cube_displacements_xy = []
        for key in self.color_dict:
            current_xy = self.cubes[key].pose.p[..., :2]
            initial_xy = self.initial_poses[key][..., :2]
            cube_displacements_xy.append(torch.linalg.norm(current_xy - initial_xy, dim=1))
        self.cube_displacements_xy = torch.stack(cube_displacements_xy, dim=1)
        self.max_cube_displacement_xy = self.cube_displacements_xy.max(dim=1).values
        self.mean_cube_displacement_xy = self.cube_displacements_xy.mean(dim=1)
        self.strong_cube_displacement = (
            self.max_cube_displacement_xy > self.MAX_ALLOWED_CUBE_DISPLACEMENT
        ) & show_all_cubes

        action_mask = show_all_cubes
        button_ready = (
            all_cubes_from_sequence_is_touched
            & no_one_cube_not_from_sequence_is_touched
            & (~self.strong_cube_displacement)
        )
        self.button_pressable = button_ready & action_mask

        tcp_xy = tcp_pos[..., :2]
        tcp_z = tcp_pos[..., 2]
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
        self.new_release_event = (~self.press_ready) & released & action_mask
        self.press_ready = self.press_ready | self.new_release_event

        self.new_raw_press_event = (
            pressed & self.press_ready & action_mask & self.button_pressable & (~self.pending_press)
        )
        self.press_start_tcp_z[self.new_raw_press_event] = tcp_z[self.new_raw_press_event]
        self.pending_press = self.pending_press | self.new_raw_press_event
        self.press_ready = self.press_ready & (~self.new_raw_press_event)

        lift_target_z = self.press_start_tcp_z + self.REQUIRED_LIFT_HEIGHT
        at_lift_target = tcp_z >= (lift_target_z - self.LIFT_CONFIRM_TOL)

        self.new_press_event = (
            self.pending_press & at_lift_target & self.press_ready & action_mask & self.button_pressable
        )
        self.button_press_count = self.button_press_count + self.new_press_event.to(torch.int64)
        self.pending_press = self.pending_press & (~self.new_press_event)
        self.button_pressed = self.button_pressed | self.new_press_event

        success = (
            all_cubes_from_sequence_is_touched
            & no_one_cube_not_from_sequence_is_touched
            & self.button_pressed
            & self.agent.is_static(0.2)
        )
        success &= ~self.strong_cube_displacement
        success *= show_all_cubes

        next_target_mask = torch.zeros_like(
            sequence_cubes_mask,
            dtype=torch.bool,
            device=self.device,
        )
        untouched_sequence_cubes = sequence_cubes_mask & ~self.touched_cubes
        first_untouched_indices = torch.argmax(untouched_sequence_cubes.float(), dim=1)
        batch_indices = torch.arange(sequence_cubes_mask.shape[0], device=self.device)
        next_target_mask[batch_indices, first_untouched_indices] = True
        next_target_mask *= show_all_cubes.unsqueeze(1)
        next_target_mask *= untouched_sequence_cubes.any(dim=1).unsqueeze(1)

        self.obj_to_goal_pos = torch.zeros_like(
            cube_touch_pos.p,
            device=cube_touch_pos.p.device,
            dtype=cube_touch_pos.p.dtype,
        )
        for key in self.color_dict:
            cube_pos = self.cubes[key].pose.p
            cube_touch_pos = Pose.create_from_pq(
                p=cube_pos + torch.tensor([0, 0, self.CUBE_HALFSIZE + 0.005], device=self.device),
            )
            self.obj_to_goal_pos += (cube_touch_pos.p - self.agent.tcp.pose.p) * next_target_mask[:, key].unsqueeze(-1)

        button_target_mask = self.button_pressable & (~self.button_pressed)
        button_target_pos = torch.zeros_like(self.obj_to_goal_pos)
        button_target_pos[:, :2] = self.button_xy
        button_target_pos[:, 2] = self.button_top_z + 0.005
        self.obj_to_goal_pos = torch.where(
            button_target_mask.unsqueeze(-1),
            button_target_pos - self.agent.tcp.pose.p,
            self.obj_to_goal_pos,
        )

        self.next_target_mask = next_target_mask
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_robot_static": is_robot_static,
            "success": success,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "strong_cube_displacement": self.strong_cube_displacement,
            "max_cube_displacement_xy": self.max_cube_displacement_xy,
            "button_pressed": self.button_pressed,
            "button_pressable": self.button_pressable,
            "button_press_depth": self.button_press_depth,
            "xy_dist_to_button": xy_dist_to_button,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_to_goal_pos=self.obj_to_goal_pos,
                oracle_info=self.oracle_info,
                button_xy=self.button_xy,
                button_pressed=self.button_pressed,
            )
            for key in self.cubes:
                obs[f"cube_{key}_pose"] = self.cubes[key].pose.p
                obs[f"touched_{key}"] = self.touched_cubes[:, key]
                obs[f"cube_{key}_in_seq"] = self.sequence_cubes_mask[:, key]
                obs[f"next_target_{key}"] = self.next_target_mask[:, key]
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            if torch.is_tensor(success):
                success = success.to(dtype=torch.bool)
                if torch.is_tensor(terminated):
                    terminated = terminated.to(dtype=torch.bool) & (~success)
                else:
                    terminated = bool(terminated) and not bool(success.any().item())
            else:
                terminated = bool(terminated) and not bool(success)
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_obj_dist)

        correct_touches = (self.touched_cubes & self.sequence_cubes_mask).sum(1)
        correct_touch_reward = (correct_touches.float() / self.SEQUENCE_LENGTH) * 90.0

        wrong_touches = (self.touched_cubes & ~self.sequence_cubes_mask).sum(1)
        wrong_touch_penalty = 10.0 * wrong_touches.float()

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1),
        )

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)

        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, axis=1)
        delta_action_l2 = torch.linalg.norm(delta_action, axis=1)

        if hasattr(self, "elapsed_steps") and torch.is_tensor(self.elapsed_steps):
            first_step_mask = self.elapsed_steps <= 1
            delta_action_l2 = torch.where(
                first_step_mask,
                torch.zeros_like(delta_action_l2),
                delta_action_l2,
            )

        qvel = self.agent.robot.get_qvel()[..., :-2]
        qvel_l2 = torch.linalg.norm(qvel, axis=1)

        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )
        cube_displacement_penalty = (
            self.CUBE_DISPLACEMENT_PENALTY_COEF * self.mean_cube_displacement_xy * self.active_phase.float()
        )

        reward = (
            reaching_reward
            + 0.5 * static_reward
            + correct_touch_reward
            - wrong_touch_penalty
            - smooth_penalty
            - cube_displacement_penalty
        )
        reward *= self.active_phase
        reward[info["success"]] = 400.0

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "correct_touches": correct_touches,
            "wrong_touches": wrong_touches,
            "is_robot_static": info["is_robot_static"],
            "touched_cubes": self.touched_cubes.sum(1),
            "sequence_cubes_mask": self.sequence_cubes_mask.sum(1),
            "all_seq_touched": self.all_cubes_from_sequence_is_touched,
            "no_extra_touched": self.no_one_cube_not_from_sequence_is_touched,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
            "mean_cube_displacement_xy": self.mean_cube_displacement_xy,
            "max_cube_displacement_xy": self.max_cube_displacement_xy,
            "cube_displacement_penalty": cube_displacement_penalty,
            "strong_cube_displacement": self.strong_cube_displacement,
        }

        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 400.0


# ----- Standard tasks -----
@register_env("BunchOfColors3-VLA-v0", max_episode_steps=400)
class BunchOfColors3VLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 3
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("BunchOfColors5-VLA-v0", max_episode_steps=400)
class BunchOfColors5VLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 5
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("BunchOfColors7-VLA-v0", max_episode_steps=400)
class BunchOfColors7VLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 7
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


# ----- Long-horizon tasks -----
@register_env("BunchOfColors3-Long-VLA-v0", max_episode_steps=700)
class BunchOfColors3LongVLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 3
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 400]


@register_env("BunchOfColors5-Long-VLA-v0", max_episode_steps=700)
class BunchOfColors5LongVLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 5
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 400]


@register_env("BunchOfColors7-Long-VLA-v0", max_episode_steps=700)
class BunchOfColors7LongVLAEnv(BunchOfColorsVLABaseEnv):
    SEQUENCE_LENGTH = 7
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 400]
