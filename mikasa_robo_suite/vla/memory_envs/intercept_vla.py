"""Interception-and-push tasks for the VLA benchmark."""

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


class InterceptVLABaseEnv(BaseEnv):
    """Intercept a moving ball and push it into a target region.

    A ball starts with random motion, and the robot must make contact at the
    right time and direction so that the ball rolls into the goal area. Unlike
    the grasping variant, the task ends with successful redirection rather than
    object pickup.

    Episode flow:
    - The ball is launched with sampled initial velocity.
    - The robot moves to an effective hitting position behind the ball.
    - The ball is redirected toward the goal region.

    Success (`success=True`):
    - The ball center must end up inside the goal radius.

    How to customize:
    - `VELOCITY_RANGE` changes how difficult interception timing becomes.
    - `BALL_RADIUS` changes contact geometry and how easy the ball is to push.
    - `GOAL_RADIUS` changes how forgiving the final placement criterion is.
    - `ACTION_L2_COEF`, `ACTION_DELTA_L2_COEF`, and `QVEL_L2_COEF` can be used
      to penalize unstable motions if needed.
    """

    LANGUAGE_INSTRUCTION = "Intercept the rolling ball by moving to its path and deflecting it toward the target."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    VELOCITY_RANGE = (0.0, 0.0)
    GOAL_RADIUS: float = 0.1
    BALL_RADIUS: float = 0.02

    ACTION_L2_COEF = 0.0
    ACTION_DELTA_L2_COEF = 0.0
    QVEL_L2_COEF = 0.0

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

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([1, 0, 0, 1]),
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.GOAL_RADIUS,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b,)) * 2 - 1) * 0.15 - 0.2
            xyz[..., 1] = torch.rand((b,)) * 0.25 - 1.0 + self.GOAL_RADIUS
            xyz[..., 2] = 2 * self.BALL_RADIUS
            self.ball.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            initial_velocity = torch.zeros((b, 3))
            initial_velocity[..., 0] = torch.rand((b,)) * 0.05
            min_vel, max_vel = self.VELOCITY_RANGE
            initial_velocity[..., 1] = torch.rand((b,)) * (max_vel - min_vel) + min_vel
            self.ball.set_linear_velocity(initial_velocity)

            self.oracle_info = initial_velocity

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = xyz[..., 0].clone() + 0.2
            xyz_goal[..., 1] = torch.rand((b,)) * 0.4 - 1.0 + self.GOAL_RADIUS + 0.7
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(p=xyz_goal, q=euler2quat(0, np.pi / 2, 0)),
            )

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

        self.reached_status[env_idx] = 0.0

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2],
                axis=1,
            )
            < self.GOAL_RADIUS
        )

        return {
            "success": is_obj_placed,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                ball_pose=self.ball.pose.raw_pose,
                oracle_info=self.oracle_info,
                ball_linear_vel=self.ball.linear_velocity,
                ball_angular_vel=self.ball.angular_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
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
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)

        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.BALL_RADIUS + 0.05),
        )

        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)

        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0

        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

        obj_to_goal_dist = torch.linalg.norm(self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1)

        place_reward = 1 - torch.tanh(obj_to_goal_dist)

        reward = 5.0 * place_reward * self.reached_status + 2.0 * reaching_reward * torch.logical_not(
            self.reached_status
        )

        reward[info["success"]] = 30.0

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "reached_status": self.reached_status,
            "place_reward": place_reward,
            "tcp_to_hit_pose_dist": tcp_to_hit_pose_dist,
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("InterceptSlow-VLA-v0", max_episode_steps=60)
class InterceptSlowVLAEnv(InterceptVLABaseEnv):
    VELOCITY_RANGE = (0.25, 0.5)


@register_env("InterceptMedium-VLA-v0", max_episode_steps=60)
class InterceptMediumVLAEnv(InterceptVLABaseEnv):
    VELOCITY_RANGE = (0.5, 0.75)


@register_env("InterceptFast-VLA-v0", max_episode_steps=60)
class InterceptFastVLAEnv(InterceptVLABaseEnv):
    VELOCITY_RANGE = (0.75, 1.0)
