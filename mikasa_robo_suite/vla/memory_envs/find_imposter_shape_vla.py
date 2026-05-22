"""Find-the-imposter-shape tasks for the VLA memory benchmark."""

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


class FindImposterShapeVLABaseEnv(BaseEnv):
    """Find the shape whose geometry was NOT present in the first phase.

    All shapes share the same blue color; only geometry distinguishes them.

    Episode flow:
    - Phase 1 (cue):  SHAPES-1 shapes are shown at spread positions.
                      One geometry from the pool is deliberately hidden.
    - Phase 2 (empty): All shapes disappear.
    - Phase 3 (manip): All SHAPES objects appear at spread positions.
                       Touch the shape whose geometry was absent in the cue.

    Success: TCP within GOAL_THRESH of the imposter shape in the manipulation phase.
    """

    LANGUAGE_INSTRUCTION = "Observe the shapes shown, wait, then touch the object whose shape was not present before."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SHAPES = 3

    GOAL_THRESH = 0.05
    SHAPE_SCALE = 0.02
    COLOR = [0, 0, 255, 255]

    SHAPE_MAPPING = {
        0: "cube",
        1: "sphere",
        2: "cylinder",
        3: "cross",
        4: "torus",
        5: "star",
        6: "pyramide",
        7: "t_shape",
        8: "crescent",
    }

    CUE_PHASE_STEPS: List[int] = [1, 5]
    EMPTY_PHASE_STEPS: List[int] = [1, 5]

    MANIP_MIN_SHAPE_DISTANCE = 0.09
    MANIP_WIDTH_AXIS = 1
    MANIP_WIDTH_SCALE = 2
    MANIP_WIDTH_CLAMP = 0.5

    ACTION_L2_COEF = 0.02
    ACTION_DELTA_L2_COEF = 0.05
    QVEL_L2_COEF = 0.01

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.shape_dict = dict(list(self.SHAPE_MAPPING.items())[: self.SHAPES])
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
            "cylinder": lambda: actors.build_cylinder(
                self.scene, radius=s, half_length=s, name=f"cylinder_{key}", **common
            ),
            "cross": lambda: shapes.build_cross(
                self.scene, arm_length=s * 1.5, width=s * 0.75, name=f"cross_{key}", **common
            ),
            "torus": lambda: shapes.build_torus(self.scene, radius=s, tube_radius=s / 2, name=f"torus_{key}", **common),
            "star": lambda: shapes.build_star(
                self.scene, radius=s * 1.5, thickness=s * 0.75, name=f"star_{key}", **common
            ),
            "pyramide": lambda: shapes.build_pyramid(
                self.scene, base_size=s, height=s, name=f"pyramide_{key}", **common
            ),
            "t_shape": lambda: shapes.build_t_shape(
                self.scene, width=s * 2, height=s * 2, thickness=s * 0.75, name=f"t_shape_{key}", **common
            ),
            "crescent": lambda: shapes.build_crescent(
                self.scene, outer_radius=s, height=s, thickness=s / 2, name=f"crescent_{key}", **common
            ),
        }
        if shape_name not in builders:
            raise NotImplementedError(shape_name)
        return builders[shape_name]()

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()

        color = np.array(self.COLOR) / 255.0
        self.shape_actors = {}
        for key, shape_name in self.shape_dict.items():
            self.shape_actors[key] = self._build_shape_actor(shape_name, key, color)

    def _ensure_phase_buffers(self, env_idx: torch.Tensor):
        target_size = int(env_idx.max().item()) + 1
        if not hasattr(self, "cue_steps_per_env") or self.cue_steps_per_env is None:
            self.cue_steps_per_env = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            self.empty_steps_per_env = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            self.manip_layout_applied = torch.zeros(target_size, dtype=torch.bool, device=self.device)
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
            self.manip_layout_applied = torch.cat(
                [self.manip_layout_applied, torch.zeros(pad, dtype=torch.bool, device=self.device)], dim=0
            )

    def _set_actor_state_for_mask(
        self, actor, target_pose: torch.Tensor, mask: torch.Tensor, *, zero_velocity: bool = False
    ) -> None:
        if not bool(mask.any().item()):
            return
        env_idx = torch.where(mask)[0]
        state = actor.get_state()[env_idx].clone()
        state[:, :7] = target_pose[env_idx]
        if zero_velocity:
            state[:, 7:13] = 0
        actor.set_state(state, env_idx=env_idx)

    def _compute_phase_masks(self, elapsed_steps: torch.Tensor):
        cue_end = self.cue_steps_per_env
        empty_end = cue_end + self.empty_steps_per_env
        cue_mask = elapsed_steps < cue_end
        empty_mask = (elapsed_steps >= cue_end) & (elapsed_steps < empty_end)
        manip_mask = elapsed_steps >= empty_end
        return cue_mask, empty_mask, manip_mask

    def _apply_phase_layout(self, elapsed_steps: torch.Tensor, env_mask: torch.Tensor = None) -> None:
        cue_mask, empty_mask, manip_mask = self._compute_phase_masks(elapsed_steps)
        if env_mask is not None:
            cue_mask = cue_mask & env_mask
            empty_mask = empty_mask & env_mask
            manip_mask = manip_mask & env_mask
        hidden_mask = ~manip_mask
        if env_mask is not None:
            hidden_mask = hidden_mask & env_mask
        just_entered_manip = manip_mask & (~self.manip_layout_applied)
        has_hidden = bool(hidden_mask.any().item())
        has_just_entered = bool(just_entered_manip.any().item())

        for key in self.shape_dict:
            actor = self.shape_actors[key]

            if has_hidden:
                hidden_pose = self._cue_raw_poses[key].clone()
                hidden_pose[empty_mask, 2] = 1000
                is_imposter = self.imposter_key == key
                hidden_pose[is_imposter & cue_mask, 2] = 1000
                self._set_actor_state_for_mask(actor, hidden_pose, hidden_mask, zero_velocity=True)

            if has_just_entered:
                self._set_actor_state_for_mask(
                    actor,
                    self._manip_raw_poses[key],
                    just_entered_manip,
                    zero_velocity=True,
                )

        self.manip_layout_applied[just_entered_manip] = True

    def _before_simulation_step(self):
        super()._before_simulation_step()
        if not hasattr(self, "cue_steps_per_env"):
            return
        next_elapsed = self.elapsed_steps.to(torch.int64) + 1
        self._apply_phase_layout(next_elapsed)
        if self._sim_device.is_cuda():
            self.scene.px.gpu_apply_rigid_dynamic_data()

    def _get_shape_quaternion(self, shape_name: str):
        if shape_name == "cylinder":
            return [0.7071068, 0, 0.7071068, 0]
        return [1, 0, 0, 0]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None
            if hasattr(self, "_prev_action") and self._prev_action is not None:
                if torch.is_tensor(self._prev_action) and self._prev_action.shape[0] >= int(env_idx.max().item()) + 1:
                    self._prev_action[env_idx] = 0

            # imposter_key[env_i]: which shape is hidden in cue but appears in manip
            self.imposter_key = self._batched_episode_rng.choice(list(self.shape_dict.keys()))
            self.imposter_key = torch.from_numpy(self.imposter_key).to(device=self.device, dtype=torch.uint8)

            xyz_initial = torch.zeros((b, 3))
            for key, shape_name in self.shape_dict.items():
                xyz = xyz_initial.clone()
                if self.SHAPES != 3:
                    angle = np.pi * (key - (len(self.shape_dict) // 2)) / len(self.shape_dict)
                    radius = 0.3
                    xyz[..., 0] = radius * np.cos(angle) - 0.25
                    xyz[..., 1] = radius * np.sin(angle)
                    if self.SHAPES in [5, 9]:
                        xyz[..., 1] -= (key - (len(self.shape_dict) // 2)) * 0.025
                else:
                    xyz[..., 1] -= (key - (len(self.shape_dict) // 2)) * 0.1
                xyz[..., 2] = self.SHAPE_SCALE
                q = self._get_shape_quaternion(shape_name)
                self.shape_actors[key].set_pose(Pose.create_from_pq(p=xyz, q=q))
                self.initial_poses[key] = xyz.clone()

            min_distance = self.SHAPE_SCALE * 3
            max_attempts = 50
            for env_i in range(b):
                positions = [self.initial_poses[key][env_i].clone() for key in self.initial_poses]
                for i in range(len(positions)):
                    attempt = 0
                    while attempt < max_attempts:
                        noise = torch.randn(2, device=self.device) * self.SHAPE_SCALE * 0.5
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
                    pose = self.shape_actors[key].pose.raw_pose.clone()
                    pose[env_i, :3] = positions[idx]
                    self.shape_actors[key].pose = pose

            self.oracle_info = self.imposter_key

            self._cue_raw_poses = {key: self.shape_actors[key].pose.raw_pose.clone() for key in self.shape_actors}

            # Phase-3 uses the same position slots as phase-1, but shuffled across keys.
            keys = list(self.shape_actors.keys())
            n = len(keys)
            self._manip_raw_poses = {key: self._cue_raw_poses[key].clone() for key in keys}
            for env_i in range(b):
                perm = torch.randperm(n, device=self.device)
                while n > 1 and torch.all(perm == torch.arange(n, device=self.device)):
                    perm = torch.randperm(n, device=self.device)
                positions = [self._cue_raw_poses[keys[k]][env_i, :3].clone() for k in range(n)]
                for k_idx, key in enumerate(keys):
                    self._manip_raw_poses[key][env_i, :3] = positions[int(perm[k_idx])]

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
            self.manip_layout_applied[env_idx] = False

            reset_mask = torch.zeros_like(self.manip_layout_applied, dtype=torch.bool, device=self.device)
            reset_mask[env_idx] = True
            if hasattr(self, "elapsed_steps") and torch.is_tensor(self.elapsed_steps):
                init_elapsed = self.elapsed_steps.to(torch.int64).clone()
            else:
                init_elapsed = torch.zeros_like(self.cue_steps_per_env, dtype=torch.int64, device=self.device)
            init_elapsed[reset_mask] = 0
            self._apply_phase_layout(init_elapsed, env_mask=reset_mask)

    def evaluate(self):
        elapsed_steps = self.elapsed_steps.to(torch.int64)
        _, _, manip_mask = self._compute_phase_masks(elapsed_steps)

        self.masks = {key: (self.imposter_key == key).unsqueeze(-1) for key in self.shape_dict}
        self.obj_to_goal_pos = torch.zeros_like(
            self.shape_actors[0].pose.p,
            device=self.shape_actors[0].pose.p.device,
            dtype=self.shape_actors[0].pose.p.dtype,
        )
        for key in self.shape_dict:
            self.obj_to_goal_pos += (self.shape_actors[key].pose.p - self.agent.tcp.pose.p) * self.masks[key]

        is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success": is_obj_placed & is_robot_static & manip_mask,
            "task_cue": self.task_cue,
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


@register_env("FindImposterShape3-VLA-v0", max_episode_steps=25)
class FindImposterShape3VLAEnv(FindImposterShapeVLABaseEnv):
    SHAPES = 3
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("FindImposterShape5-VLA-v0", max_episode_steps=25)
class FindImposterShape5VLAEnv(FindImposterShapeVLABaseEnv):
    SHAPES = 5
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]


@register_env("FindImposterShape9-VLA-v0", max_episode_steps=25)
class FindImposterShape9VLAEnv(FindImposterShapeVLABaseEnv):
    SHAPES = 9
    CUE_PHASE_STEPS = [1, 5]
    EMPTY_PHASE_STEPS = [1, 5]
