"""Remember-color tasks for the VLA memory benchmark."""

from typing import Any, Dict, List

import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class RememberColorVLABaseEnv(BaseEnv):
    """Remember one target color and pick it out after a delay.

    The environment briefly shows a single target cube color. Then all cubes are
    hidden, and finally all candidate cubes reappear in randomized positions.
    The robot must remember only the color identity and ignore the new spatial
    arrangement during selection.

    Episode flow:
    - One target color is shown in the center as the cue.
    - All cubes disappear during the memory phase.
    - All candidate cubes reappear and the robot selects the correct one.

    Success (`success=True`):
    - The robot must reach the cube whose color matches the cue and satisfy the
      environment reach threshold.

    How to customize:
    - `COLORS` changes how many candidate colors compete with the target.
    - `CUE_PHASE_STEPS` changes how long the cue stays visible.
    - `EMPTY_PHASE_STEPS` changes the length of the memory delay.
    - `GOAL_THRESH` changes how strict the final selection reach criterion is.
    - `CUBE_HALFSIZE` changes cube size and indirectly affects spacing.
    """

    LANGUAGE_INSTRUCTION = "Observe the cube's color, wait, then touch the cube of the same color."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    COLORS = 3

    GOAL_THRESH = 0.05
    CUBE_HALFSIZE = 0.02

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

    CUE_PHASE_STEPS: List[int] = [1, 5]
    EMPTY_PHASE_STEPS: List[int] = [1, 5]

    MANIP_MIN_CUBE_DISTANCE = 0.09
    MANIP_WIDTH_AXIS = 1
    MANIP_WIDTH_SCALE = 2
    MANIP_WIDTH_CLAMP = 0.5

    ACTION_L2_COEF = 0.02
    ACTION_DELTA_L2_COEF = 0.05
    QVEL_L2_COEF = 0.01

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.color_dict = {k: np.array(v[1]) / 255.0 for k, v in list(self.COLOR_MAPPING.items())[: self.COLORS]}
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.initial_poses = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
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
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
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
                [self.cue_steps_per_env, torch.zeros(pad, dtype=torch.int64, device=self.device)], dim=0
            )
            self.empty_steps_per_env = torch.cat(
                [self.empty_steps_per_env, torch.zeros(pad, dtype=torch.int64, device=self.device)], dim=0
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None
            if hasattr(self, "_prev_action") and self._prev_action is not None:
                if torch.is_tensor(self._prev_action) and self._prev_action.shape[0] >= int(env_idx.max().item()) + 1:
                    self._prev_action[env_idx] = 0

            self.true_color_indices = self._batched_episode_rng.choice(list(self.color_dict.keys()))
            self.true_color_indices = torch.from_numpy(self.true_color_indices).to(
                device=self.device, dtype=torch.uint8
            )

            xyz_initial = torch.zeros((b, 3))
            self.center_pose = xyz_initial.clone()
            self.center_pose[..., 2] = self.CUBE_HALFSIZE
            self.center_pose = self.center_pose[0].unsqueeze(0)

            for key, color in self.color_dict.items():
                xyz_cube = xyz_initial.clone()
                if self.COLORS != 3:
                    angle = np.pi * (key - (len(self.color_dict) // 2)) / len(self.color_dict)
                    radius = 0.3
                    xyz_cube[..., 0] = radius * np.cos(angle) - 0.25
                    xyz_cube[..., 1] = radius * np.sin(angle)
                    if self.COLORS in [5, 9]:
                        xyz_cube[..., 1] -= (key - (len(self.color_dict) // 2)) * 0.025
                else:
                    xyz_cube[..., 1] -= (key - (len(self.color_dict) // 2)) * 0.1
                xyz_cube[..., 2] = self.CUBE_HALFSIZE
                self.cubes[key].set_pose(Pose.create_from_pq(p=xyz_cube, q=[1, 0, 0, 0]))
                self.initial_poses[key] = xyz_cube.clone()

            min_distance = self.CUBE_HALFSIZE * 3
            max_attempts = 50
            for env_i in range(b):
                positions = [self.initial_poses[key][env_i].clone() for key in self.initial_poses]
                for i in range(len(positions)):
                    attempt = 0
                    while attempt < max_attempts:
                        noise = torch.randn(2, device=self.device) * self.CUBE_HALFSIZE * 0.5
                        new_pos = positions[i].clone()
                        new_pos[:2] += noise
                        valid = all(torch.norm(new_pos[:2] - positions[j][:2]) >= min_distance for j in range(i))
                        if valid:
                            positions[i] = new_pos
                            break
                        attempt += 1
                shuffled_indices = torch.randperm(len(positions))
                for key, idx in zip(self.initial_poses, shuffled_indices):
                    self.initial_poses[key][env_i] = positions[idx]
                    pose = self.cubes[key].pose.raw_pose.clone()
                    pose[env_i, :3] = positions[idx]
                    self.cubes[key].pose = pose

            self.oracle_info = self.true_color_indices

            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
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
            if self.COLORS not in (5, 9):
                self._spread_cubes_for_manipulation_phase(env_idx)

    def _spread_cubes_for_manipulation_phase(self, env_idx: torch.Tensor):
        keys = list(self.initial_poses.keys())
        max_attempts = 80
        jitter_scale = self.CUBE_HALFSIZE * 0.6

        for env_i in env_idx.tolist():
            positions = [self.initial_poses[key][env_i].clone() for key in keys]
            for i in range(len(positions)):
                attempts = 0
                while attempts < max_attempts:
                    valid = all(
                        torch.norm(positions[i][:2] - positions[j][:2]) >= self.MANIP_MIN_CUBE_DISTANCE
                        for j in range(i)
                    )
                    if valid:
                        break
                    positions[i][:2] += torch.randn(2, device=self.device) * jitter_scale
                    attempts += 1

            for key, pos in zip(keys, positions):
                pos[self.MANIP_WIDTH_AXIS] = torch.clamp(
                    pos[self.MANIP_WIDTH_AXIS],
                    -self.MANIP_WIDTH_CLAMP,
                    self.MANIP_WIDTH_CLAMP,
                )
                self.initial_poses[key][env_i] = pos
                current_pose = self.cubes[key].pose.raw_pose.clone()
                current_pose[env_i, :3] = pos
                self.cubes[key].pose = current_pose

            self._stretch_width_axis(env_i, keys)

    def _stretch_width_axis(self, env_i: int, keys):
        axis = self.MANIP_WIDTH_AXIS
        axis_vals = torch.stack([self.initial_poses[key][env_i, axis] for key in keys], dim=0)
        center = axis_vals.mean()

        for key in keys:
            pos = self.initial_poses[key][env_i].clone()
            pos[axis] = center + (pos[axis] - center) * self.MANIP_WIDTH_SCALE
            pos[axis] = torch.clamp(pos[axis], -self.MANIP_WIDTH_CLAMP, self.MANIP_WIDTH_CLAMP)
            self.initial_poses[key][env_i] = pos
            current_pose = self.cubes[key].pose.raw_pose.clone()
            current_pose[env_i, :3] = pos
            self.cubes[key].pose = current_pose

    def evaluate(self):
        self.original_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes}

        elapsed_steps = self.elapsed_steps.to(torch.int64)
        cue_end = self.cue_steps_per_env
        empty_end = cue_end + self.empty_steps_per_env
        empty_mask = (elapsed_steps >= cue_end) & (elapsed_steps < empty_end)
        manip_mask = elapsed_steps >= empty_end
        hidden_phase_mask = ~manip_mask
        appeared_mask = manip_mask & (elapsed_steps == empty_end)

        hidden_shapes_poses = {}
        for key in self.color_dict:
            hidden_shapes_poses[key] = self.cubes[key].pose.raw_pose.clone()
            hidden_shapes_poses[key][hidden_phase_mask, 2] = 1000
            self.cubes[key].pose = hidden_shapes_poses[key]

        for key in self.color_dict:
            true_shape_mask = self.true_color_indices == key
            b_ = hidden_shapes_poses[key].shape[0]
            hidden_shapes_poses[key][true_shape_mask & hidden_phase_mask, :3] = self.center_pose.repeat(b_, 1)[
                true_shape_mask & hidden_phase_mask, :3
            ]
            hidden_shapes_poses[key][true_shape_mask & empty_mask, 2] = 1000
            self.cubes[key].pose = hidden_shapes_poses[key]

        for key in self.color_dict:
            hidden_shapes_poses[key][appeared_mask, :3] = self.initial_poses[key][appeared_mask, :3]
            self.cubes[key].pose = hidden_shapes_poses[key]

            lock_mask = hidden_phase_mask | appeared_mask
            if bool(lock_mask.any().item()):
                lin_vel = self.cubes[key].linear_velocity.clone()
                ang_vel = self.cubes[key].angular_velocity.clone()
                lin_vel[lock_mask] = 0
                ang_vel[lock_mask] = 0
                self.cubes[key].set_linear_velocity(lin_vel)
                self.cubes[key].set_angular_velocity(ang_vel)

        self.masks = {key: (self.true_color_indices == key).unsqueeze(-1) for key in self.color_dict}

        self.obj_to_goal_pos = torch.zeros_like(
            self.cubes[0].pose.p,
            device=self.cubes[0].pose.p.device,
            dtype=self.cubes[0].pose.p.dtype,
        )
        for key in self.color_dict:
            self.obj_to_goal_pos += (self.cubes[key].pose.p - self.agent.tcp.pose.p) * self.masks[key]

        is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success": is_obj_placed & is_robot_static,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs["oracle_info"] = self.oracle_info
            for key in self.cubes:
                obs[f"goal_{key}_pose"] = self.cubes[key].pose.p * self.masks[key]
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
                    terminated = bool(terminated) and (not bool(success.any().item()))
            else:
                terminated = bool(terminated) and (not bool(success))
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_obj_dist)

        qvel = self.agent.robot.get_qvel()[..., :-2]
        qvel_l2 = torch.linalg.norm(qvel, axis=1)
        static_reward = 1 - torch.tanh(5 * qvel_l2)

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)

        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, axis=1)
        delta_action_l2 = torch.linalg.norm(delta_action, axis=1)

        if hasattr(self, "elapsed_steps") and torch.is_tensor(self.elapsed_steps):
            first_step_mask = self.elapsed_steps <= 1
            delta_action_l2 = torch.where(first_step_mask, torch.zeros_like(delta_action_l2), delta_action_l2)

        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        reached = tcp_to_obj_dist < self.GOAL_THRESH

        reward = (
            1.0 * reaching_reward
            + 0.5 * static_reward
            + 0.5 * info["is_robot_static"] * info["is_obj_placed"]
            - smooth_penalty
        )
        reward[info["success"]] = 3.0

        self.reward_dict = {
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "reaching_reward": reaching_reward,
            "is_robot_static": info["is_robot_static"],
            "reached": reached,
            "success": info["success"],
            "static_reward": static_reward,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
            "obj_to_goal_pos_y": info["obj_to_goal_pos"][:, 1],
            "obj_to_goal_pos_x": info["obj_to_goal_pos"][:, 0],
        }

        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0


# ----- Standard tasks -----
@register_env("RememberColor3-VLA-v0", max_episode_steps=25)
class RememberColor3VLAEnv(RememberColorVLABaseEnv):
    COLORS = 3
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("RememberColor5-VLA-v0", max_episode_steps=25)
class RememberColor5VLAEnv(RememberColorVLABaseEnv):
    COLORS = 5
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("RememberColor9-VLA-v0", max_episode_steps=25)
class RememberColor9VLAEnv(RememberColorVLABaseEnv):
    COLORS = 9
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


# ----- Long-horizon tasks -----
@register_env("RememberColor3-Long-VLA-v0", max_episode_steps=600)
class RememberColor3LongVLAEnv(RememberColorVLABaseEnv):
    COLORS = 3
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 450]


@register_env("RememberColor5-Long-VLA-v0", max_episode_steps=600)
class RememberColor5LongVLAEnv(RememberColorVLABaseEnv):
    COLORS = 5
    CUE_PHASE_STEPS = [100, 100]
    EMPTY_PHASE_STEPS = [50, 450]


@register_env("RememberColor9-Long-VLA-v0", max_episode_steps=600)
class RememberColor9LongVLAEnv(RememberColorVLABaseEnv):
    COLORS = 9
    CUE_PHASE_STEPS = [100, 100]
    EMPTY_PHASE_STEPS = [50, 450]
