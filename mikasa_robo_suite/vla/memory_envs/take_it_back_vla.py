"""Take-it-back manipulation tasks for the VLA benchmark."""

from typing import Any, Dict, Union

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
from transforms3d.euler import euler2quat

from mikasa_robo_suite.vla.utils import shapes


class TakeItBackVLABaseEnv(BaseEnv):
    """Push an object away, then bring it back in the same episode.

    The cube starts on an initial target region. The robot must first move it to
    a separate goal region. Once that happens, the task switches and the robot
    must return the same cube back to where it started. The challenge is not
    only reaching both targets, but also reacting correctly to the stage change.

    Episode flow:
    - The cube starts on the initial region.
    - The robot pushes the cube to the goal region.
    - After the stage switch, the robot brings the cube back to the initial region.

    Success (`success=True`):
    - The cube must first reach the goal region and then end up back inside the
      initial region within the same episode.

    How to customize:
    - `GOAL_RADIUS` changes how large both target regions are and therefore how
      forgiving both placement checks become.
    - `CUBE_HALFSIZE` changes the cube geometry and contact behavior.
    - The stage switch is implicit: if you change goal logic in the task, you
      are also changing when the return stage begins.
    """

    LANGUAGE_INSTRUCTION = "Push the cube onto the red target, and when the target changes color, return the cube to its original position."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    GOAL_RADIUS: float = 0.08
    CUBE_HALFSIZE: float = 0.02

    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.08
    QVEL_L2_COEF = 0.01

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
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

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.CUBE_HALFSIZE,
            color=np.array([0, 255, 0, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
        )

        self.initial_region = shapes.build_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="initial_region",
            add_collision=False,
            body_type="kinematic",
            primary_color=np.array([0, 0, 255, 255]) / 255,
            secondary_color=np.array([255, 255, 255, 255]) / 255,
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_region = shapes.build_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            primary_color=np.array([194, 19, 22, 255]) / 255,
            secondary_color=np.array([255, 255, 255, 255]) / 255,
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.changed_goal_region = shapes.build_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="changed_goal_region",
            add_collision=False,
            body_type="kinematic",
            primary_color=np.array([147, 0, 211, 255]) / 255,
            secondary_color=np.array([255, 255, 255, 255]) / 255,
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_reached_status = torch.zeros(self.num_envs, dtype=torch.float32)
        self.goal_achieved = torch.zeros(self.num_envs, dtype=torch.bool)
        self._hidden_objects.append(self.initial_region)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.goal_reached_status = self.goal_reached_status.to(self.device)
        self.goal_achieved = self.goal_achieved.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            initial_positions = torch.from_numpy(self._batched_episode_rng.choice([-1, 0, 1])).to(self.device)

            xyz_initial = torch.zeros((b, 3))
            xyz_initial[..., 0] = (torch.rand((b,)) - 0.5) * 0.1
            xyz_initial[..., 1] = initial_positions * 0.1 + torch.rand((b,)) * 0.05
            xyz_initial[..., 2] = 1e-3

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = torch.rand((b,)) * 0.05 + 0.2
            xyz_goal[..., 1] = (torch.rand((b,)) - 0.5) * 0.05
            xyz_goal[..., 2] = 1e-3

            xyz_changed_goal = xyz_goal.clone()
            xyz_changed_goal[..., 2] = 1000

            self.initial_region.set_pose(
                Pose.create_from_pq(p=xyz_initial, q=euler2quat(0, np.pi / 2, 0)),
            )
            self.goal_region.set_pose(
                Pose.create_from_pq(p=xyz_goal, q=euler2quat(0, np.pi / 2, 0)),
            )
            self.changed_goal_region.set_pose(
                Pose.create_from_pq(p=xyz_changed_goal, q=euler2quat(0, np.pi / 2, 0)),
            )

            xyz_cube = xyz_initial.clone()
            xyz_cube[..., 2] = self.CUBE_HALFSIZE
            self.cube.set_pose(Pose.create_from_pq(p=xyz_cube, q=[1, 0, 0, 0]))

            self.oracle_info = xyz_initial.clone()

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

        self.goal_reached_status[env_idx] = False
        self.goal_achieved[env_idx] = False

    def evaluate(self):
        self.original_poses = {
            "goal_region": self.goal_region.pose.raw_pose.clone(),
            "changed_goal_region": self.changed_goal_region.pose.raw_pose.clone(),
            "cube": self.cube.pose.raw_pose.clone(),
        }

        new_goal_region_pose = self.original_poses["goal_region"]
        new_goal_region_pose[self.goal_achieved, 2] = 1000

        new_changed_goal_region_pose = self.original_poses["changed_goal_region"]
        new_changed_goal_region_pose[self.goal_achieved, 2] = 1e-3

        self.goal_region.pose = new_goal_region_pose
        self.changed_goal_region.pose = new_changed_goal_region_pose

        self.goal_achieved = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2],
                axis=1,
            )
            < self.GOAL_RADIUS
        )

        self.goal_reached_status = torch.logical_or(
            self.goal_reached_status,
            self.goal_achieved,
        )

        is_cube_returned = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.initial_region.pose.p[..., :2],
                axis=1,
            )
            < self.GOAL_RADIUS
        )

        return {
            "success": is_cube_returned & self.goal_reached_status,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "goal_achieved": self.goal_achieved,
            "goal_reached_status": self.goal_reached_status,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        actual_goal_pos = self.goal_region.pose.p
        actual_goal_pos[actual_goal_pos[:, 2] > 10, 2] = self.changed_goal_region.pose.p[actual_goal_pos[:, 2] > 10, 2]

        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=actual_goal_pos,
                cube_pose=self.cube.pose.raw_pose,
                oracle_info=self.oracle_info,
                goal_reached_status=self.goal_reached_status,
            )
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
        tcp_push_pose = Pose.create_from_pq(
            p=self.cube.pose.p + torch.tensor([-self.CUBE_HALFSIZE - 0.005, 0, 0], device=self.device)
        )

        tcp_to_cube_pose_push = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_cube_pose_push_dist = torch.linalg.norm(tcp_to_cube_pose_push, axis=1)
        reaching_reward_push = 1 - torch.tanh(5.0 * tcp_to_cube_pose_push_dist)

        cube_to_goal_dist = torch.linalg.norm(self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)
        place_in_goal_reward = 1 - torch.tanh(5.0 * cube_to_goal_dist)

        cube_to_initial_dist = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.initial_region.pose.p[..., :2], axis=1
        )
        place_back_reward = 1 - torch.tanh(5.0 * cube_to_initial_dist)

        goal_achieved_status = info["goal_reached_status"]
        reached_push = tcp_to_cube_pose_push_dist < 0.03

        tcp_pull_pos = self.cube.pose.p + torch.tensor([self.CUBE_HALFSIZE + 2 * 0.005, 0, 0], device=self.device)
        tcp_to_cube_pose_pull = tcp_pull_pos - self.agent.tcp.pose.p
        tcp_to_cube_pose_pull_dist = torch.linalg.norm(tcp_to_cube_pose_pull, axis=1)
        reaching_reward_pull = 1 - torch.tanh(5.0 * tcp_to_cube_pose_pull_dist)
        tcp_to_cube_dist = torch.linalg.norm(self.cube.pose.p - self.agent.tcp.pose.p, axis=1)
        keep_contact_reward = 1 - torch.tanh(6.0 * tcp_to_cube_dist)

        reached_pull = tcp_to_cube_pose_pull_dist < 0.03

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

        qvel = self.agent.robot.get_qvel()[..., :-2]
        qvel_l2 = torch.linalg.norm(qvel, axis=1)
        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        reward = torch.zeros_like(reaching_reward_push)
        reward[~goal_achieved_status] += 2.0 * reaching_reward_push[~goal_achieved_status]
        reward[~goal_achieved_status] += 1.5 * place_in_goal_reward[~goal_achieved_status]
        reward[~goal_achieved_status & reached_push] += 5.0 * place_in_goal_reward[~goal_achieved_status & reached_push]

        reward[goal_achieved_status] += 10.0 * reaching_reward_pull[goal_achieved_status]
        reward[goal_achieved_status] += 4.0 * keep_contact_reward[goal_achieved_status]
        reward[goal_achieved_status & reached_pull] += 15.0 * place_back_reward[goal_achieved_status & reached_pull]
        reward -= smooth_penalty

        reward[info["success"]] = 30.0

        self.reward_dict = {
            "reached_push": reached_push,
            "reached_pull": reached_pull,
            "tcp_push_dist": tcp_to_cube_pose_push_dist,
            "tcp_pull_dist": tcp_to_cube_pose_pull_dist,
            "reaching_reward_push": reaching_reward_push,
            "reaching_reward_pull": reaching_reward_pull,
            "keep_contact_reward": keep_contact_reward,
            "goal_achieved": info["goal_achieved"],
            "goal_reached_status": info["goal_reached_status"],
            "place_in_goal_reward": place_in_goal_reward,
            "place_back_reward": place_back_reward,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
        }

        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 30.0


@register_env("TakeItBack-VLA-v0", max_episode_steps=60)
class TakeItBackVLAEnv(TakeItBackVLABaseEnv):
    pass
