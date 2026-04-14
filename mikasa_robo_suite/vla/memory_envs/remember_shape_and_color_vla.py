"""Remember-shape-and-color tasks for the VLA memory benchmark."""

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

from mikasa_robo_suite.vla.utils import shapes


class RememberShapeAndColorVLABaseEnv(BaseEnv):
    """Remember a full object identity defined by both shape and color.

    The cue shows one target object, but the later scene contains many objects
    that may share only the shape or only the color. The robot must therefore
    retain the full conjunction, not just a single attribute.

    Episode flow:
    - One target object with a specific shape and color is shown.
    - All objects disappear for the memory delay.
    - All objects reappear in randomized positions and the robot selects one.

    Success (`success=True`):
    - The robot must reach the object whose shape and color both match the cue.

    How to customize:
    - `SHAPES` changes how many shape-color combinations appear in the episode.
    - `BASE_SHAPES` changes the geometry vocabulary available to the task.
    - `COLOR_PALETTE` changes the color vocabulary combined with those shapes.
    - `CUE_PHASE_STEPS` and `EMPTY_PHASE_STEPS` change cue duration and memory delay.
    - `GOAL_THRESH` changes how strict the final reach criterion is.
    - `SHAPE_SCALE` changes the size of all generated objects.
    """

    LANGUAGE_INSTRUCTION = (
        "Observe the object's shape and color, wait, then touch the object of the same shape and color."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SHAPES = 6

    GOAL_THRESH = 0.05
    SHAPE_SCALE = 0.02

    BASE_SHAPES = {
        0: "cube",
        1: "sphere",
        2: "t_shape",
        3: "cross",
        4: "torus",
    }

    COLOR_PALETTE = {
        0: np.array([255, 0, 0, 255]) / 255.0,
        1: np.array([0, 255, 0, 255]) / 255.0,
        2: np.array([0, 0, 255, 255]) / 255.0,
    }

    CUE_PHASE_STEPS: List[int] = [1, 5]
    EMPTY_PHASE_STEPS: List[int] = [1, 5]

    MANIP_MIN_DISTANCE = 0.09
    MANIP_WIDTH_AXIS = 1
    MANIP_WIDTH_SCALE = 2
    MANIP_WIDTH_CLAMP = 0.5

    ACTION_L2_COEF = 0.0
    ACTION_DELTA_L2_COEF = 0.0
    QVEL_L2_COEF = 0.0

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.shape_color_dict = {}
        idx = 0
        for shape_id, shape_name in self.BASE_SHAPES.items():
            for color_id, color in self.COLOR_PALETTE.items():
                if idx >= self.SHAPES:
                    break
                self.shape_color_dict[idx] = {
                    "shape": shape_name,
                    "color": color,
                    "shape_id": shape_id,
                    "color_id": color_id,
                }
                idx += 1

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.initial_poses = {}
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**22,
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

    def _build_shape_actor(self, shape_name: str, key: int, color):
        common = dict(
            color=color,
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
        )
        s = self.SHAPE_SCALE
        builders = {
            "cube": lambda: actors.build_cube(self.scene, half_size=s, name=f"cube_{key}", **common),
            "sphere": lambda: actors.build_sphere(self.scene, radius=s, name=f"sphere_{key}", **common),
            "cross": lambda: shapes.build_cross(
                self.scene, arm_length=s * 1.5, width=s * 0.75, name=f"cross_{key}", **common
            ),
            "torus": lambda: shapes.build_torus(self.scene, radius=s, tube_radius=s / 2, name=f"torus_{key}", **common),
            "t_shape": lambda: shapes.build_t_shape(
                self.scene, width=s * 2, height=s * 2, thickness=s * 0.75, name=f"t_shape_{key}", **common
            ),
        }
        if shape_name not in builders:
            raise NotImplementedError(shape_name)
        return builders[shape_name]()

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        self.shape_actors = {}
        for key, info in self.shape_color_dict.items():
            self.shape_actors[key] = self._build_shape_actor(info["shape"], key, info["color"])

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

            self.prompt = None
            self.reward_dict = None

            self.true_shape_indices = self._batched_episode_rng.choice(list(self.shape_color_dict.keys()))
            self.true_shape_indices = torch.from_numpy(self.true_shape_indices).to(
                device=self.device, dtype=torch.uint8
            )

            self.true_shapes_info = torch.tensor(
                [self.shape_color_dict[idx.item()]["shape_id"] for idx in self.true_shape_indices],
                device=self.device,
                dtype=torch.uint8,
            )
            self.true_colors_info = torch.tensor(
                [self.shape_color_dict[idx.item()]["color_id"] for idx in self.true_shape_indices],
                device=self.device,
                dtype=torch.uint8,
            )

            xyz_initial = torch.zeros((b, 3))
            self.center_pose = xyz_initial.clone()
            self.center_pose[..., 2] = self.SHAPE_SCALE
            self.center_pose = self.center_pose[0].unsqueeze(0)

            n = len(self.shape_color_dict)
            for key in self.shape_color_dict:
                xyz = xyz_initial.clone()
                if n % 2 == 0:
                    angle = np.pi * (key + 0.5 - (n / 2)) / n
                else:
                    angle = np.pi * (key - (n // 2)) / n
                radius = 0.3
                xyz[..., 0] = radius * np.cos(angle) - 0.25
                xyz[..., 1] = radius * np.sin(angle)
                if n % 2 != 0 and self.SHAPES in [5, 9]:
                    xyz[..., 1] -= (key - (n // 2)) * 0.025
                xyz[..., 2] = self.SHAPE_SCALE
                self.shape_actors[key].set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
                self.initial_poses[key] = xyz.clone()

            min_distance = self.SHAPE_SCALE * 3
            max_attempts = 20
            n_objects = len(self.initial_poses)
            positions = torch.stack([self.initial_poses[k] for k in self.initial_poses], dim=1)

            for env_i in range(b):
                best_positions = positions[env_i].clone()
                best_min_dist = 0.0

                for _ in range(max_attempts):
                    noise = torch.randn(n_objects, 2, device=self.device) * self.SHAPE_SCALE * 0.5
                    candidate = positions[env_i].clone()
                    candidate[:, :2] += noise

                    diffs = candidate[:, None, :2] - candidate[None, :, :2]
                    dists = torch.norm(diffs, dim=2)
                    dists.fill_diagonal_(float("inf"))
                    cur_min = dists.min().item()

                    if cur_min > best_min_dist:
                        best_positions = candidate
                        best_min_dist = cur_min
                        if cur_min >= min_distance:
                            break

                perm = torch.randperm(n_objects)
                best_positions = best_positions[perm]

                for key, new_pos in zip(self.initial_poses, best_positions):
                    self.initial_poses[key][env_i] = new_pos
                    pose = self.shape_actors[key].pose.raw_pose.clone()
                    pose[env_i, :3] = new_pos
                    self.shape_actors[key].pose = pose

            self.oracle_info = torch.stack([self.true_shapes_info, self.true_colors_info], dim=1)

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
            if self.SHAPES not in (6, 9, 15):
                self._spread_shapes_for_manipulation_phase(env_idx)

    def _spread_shapes_for_manipulation_phase(self, env_idx: torch.Tensor):
        keys = list(self.initial_poses.keys())
        max_attempts = 80
        jitter_scale = self.SHAPE_SCALE * 0.6

        for env_i in env_idx.tolist():
            positions = [self.initial_poses[key][env_i].clone() for key in keys]
            for i in range(len(positions)):
                attempts = 0
                while attempts < max_attempts:
                    valid = all(
                        torch.norm(positions[i][:2] - positions[j][:2]) >= self.MANIP_MIN_DISTANCE for j in range(i)
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
                current_pose = self.shape_actors[key].pose.raw_pose.clone()
                current_pose[env_i, :3] = pos
                self.shape_actors[key].pose = current_pose

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
            current_pose = self.shape_actors[key].pose.raw_pose.clone()
            current_pose[env_i, :3] = pos
            self.shape_actors[key].pose = current_pose

    def evaluate(self):
        self.original_poses = {key: self.shape_actors[key].pose.raw_pose.clone() for key in self.shape_actors}

        elapsed_steps = self.elapsed_steps.to(torch.int64)
        cue_end = self.cue_steps_per_env
        empty_end = cue_end + self.empty_steps_per_env
        empty_mask = (elapsed_steps >= cue_end) & (elapsed_steps < empty_end)
        manip_mask = elapsed_steps >= empty_end
        hidden_phase_mask = ~manip_mask
        appeared_mask = manip_mask & (elapsed_steps == empty_end)

        hidden_poses = {}
        for key in self.shape_color_dict:
            hidden_poses[key] = self.shape_actors[key].pose.raw_pose.clone()
            hidden_poses[key][hidden_phase_mask, 2] = 1000
            self.shape_actors[key].pose = hidden_poses[key]

        for key in self.shape_color_dict:
            true_mask = self.true_shape_indices == key
            b_ = hidden_poses[key].shape[0]
            hidden_poses[key][true_mask & hidden_phase_mask, :3] = self.center_pose.repeat(b_, 1)[
                true_mask & hidden_phase_mask, :3
            ]
            hidden_poses[key][true_mask & empty_mask, 2] = 1000
            self.shape_actors[key].pose = hidden_poses[key]

        for key in self.shape_color_dict:
            hidden_poses[key][appeared_mask, :3] = self.initial_poses[key][appeared_mask, :3]
            self.shape_actors[key].pose = hidden_poses[key]

            lock_mask = hidden_phase_mask | appeared_mask
            if bool(lock_mask.any().item()):
                lin_vel = self.shape_actors[key].linear_velocity.clone()
                ang_vel = self.shape_actors[key].angular_velocity.clone()
                lin_vel[lock_mask] = 0
                ang_vel[lock_mask] = 0
                self.shape_actors[key].set_linear_velocity(lin_vel)
                self.shape_actors[key].set_angular_velocity(ang_vel)

        self.masks = {key: (self.true_shape_indices == key).unsqueeze(-1) for key in self.shape_color_dict}

        self.obj_to_goal_pos = torch.zeros_like(
            self.shape_actors[0].pose.p,
            device=self.shape_actors[0].pose.p.device,
            dtype=self.shape_actors[0].pose.p.dtype,
        )
        for key in self.shape_color_dict:
            self.obj_to_goal_pos += (self.shape_actors[key].pose.p - self.agent.tcp.pose.p) * self.masks[key]

        is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success": is_obj_placed & is_robot_static,
            "prompt": self.prompt,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs["oracle_info"] = self.oracle_info
            for key in self.shape_actors:
                obs[f"goal_{key}_pose"] = self.shape_actors[key].pose.p * self.masks[key]
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
@register_env("RememberShapeAndColor3x2-VLA-v0", max_episode_steps=25)
class RememberShapeAndColor3x2VLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 3 * 2
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("RememberShapeAndColor3x3-VLA-v0", max_episode_steps=25)
class RememberShapeAndColor3x3VLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 3 * 3
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("RememberShapeAndColor5x3-VLA-v0", max_episode_steps=25)
class RememberShapeAndColor5x3VLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 5 * 3
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


# ----- Long-horizon tasks -----
@register_env("RememberShapeAndColor3x2-Long-VLA-v0", max_episode_steps=600)
class RememberShapeAndColor3x2LongVLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 3 * 2
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 450]


@register_env("RememberShapeAndColor3x3-Long-VLA-v0", max_episode_steps=600)
class RememberShapeAndColor3x3LongVLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 3 * 3
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 450]


@register_env("RememberShapeAndColor5x3-Long-VLA-v0", max_episode_steps=600)
class RememberShapeAndColor5x3LongVLAEnv(RememberShapeAndColorVLABaseEnv):
    SHAPES = 5 * 3
    CUE_PHASE_STEPS = [10, 100]
    EMPTY_PHASE_STEPS = [50, 450]
