"""Interception-and-grasp tasks for the VLA benchmark."""

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


class InterceptGrabVLABaseEnv(BaseEnv):
    """Intercept a moving ball and finish with a stable grasp.

    A ball is launched across the table with random velocity. The robot has no
    separate observation phase: it must react immediately, intercept the ball,
    close the gripper around it, and settle into a stable final state.

    Episode flow:
    - The ball is spawned with a sampled initial velocity.
    - The robot moves to intercept its trajectory.
    - The robot grasps the ball and stabilizes.

    Success (`success=True`):
    - The ball must be grasped and the robot must be static at the end of the
      episode step.

    How to customize:
    - `VELOCITY_RANGE` changes how fast the ball moves and therefore how much
      anticipation the policy needs.
    - `BALL_RADIUS` changes grasp geometry and contact difficulty.
    - `ACTION_L2_COEF`, `ACTION_DELTA_L2_COEF`, and `QVEL_L2_COEF` can be used
      to regularize aggressive or jerky behavior.
    """

    LANGUAGE_INSTRUCTION = "Intercept the rolling ball and grasp it to stop it."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    VELOCITY_RANGE = (0.0, 0.0)
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
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
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
            xyz[..., 1] = torch.rand((b,)) * 0.25 - 1.0 + 0.1
            xyz[..., 2] = self.BALL_RADIUS
            self.ball.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))

            initial_velocity = torch.zeros((b, 3))
            initial_velocity[..., 0] = torch.rand((b,)) * 0.05
            min_vel, max_vel = self.VELOCITY_RANGE
            initial_velocity[..., 1] = torch.rand((b,)) * (max_vel - min_vel) + min_vel
            self.ball.set_linear_velocity(initial_velocity)

            self.oracle_info = initial_velocity

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
        is_ball_grasped = self.agent.is_grasping(self.ball)
        is_robot_static = self.agent.is_static(0.2)

        return {
            "success": is_ball_grasped & is_robot_static,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "is_ball_grasped": is_ball_grasped,
            "is_robot_static": is_robot_static,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                ball_pose=self.ball.pose.raw_pose,
                oracle_info=self.oracle_info,
                is_ball_grasped=info["is_ball_grasped"],
                ball_linear_vel=self.ball.linear_velocity,
                ball_angular_vel=self.ball.angular_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                gripper_width=self.agent.robot.get_qpos()[:, -2:].sum(dim=1),
                relative_velocity=self.ball.linear_velocity - self.agent.tcp.linear_velocity,
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
        tcp_to_obj_dist = torch.linalg.norm(self.ball.pose.p - self.agent.tcp.pose.p, axis=1)

        reward = 10 * (1 - torch.tanh(3 * tcp_to_obj_dist))
        reward = 10 * (1 - torch.tanh(3 * tcp_to_obj_dist))

        current_gripper_width = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1)

        is_ball_grasped = info["is_ball_grasped"]

        dist_tcp_to_ball_x = torch.abs(self.agent.tcp.pose.p[:, 0] - self.ball.pose.p[:, 0])
        dist_tcp_to_ball_y = torch.abs(self.agent.tcp.pose.p[:, 1] - self.ball.pose.p[:, 1])
        dist_tcp_to_ball_z = torch.abs(self.agent.tcp.pose.p[:, 2] - self.ball.pose.p[:, 2])

        ball_in_gripper = (
            (dist_tcp_to_ball_x <= self.BALL_RADIUS * 2.0)
            & (dist_tcp_to_ball_y <= self.BALL_RADIUS * 2.0)
            & (dist_tcp_to_ball_z <= self.BALL_RADIUS * 1.5)
            & (tcp_to_obj_dist <= self.BALL_RADIUS * 2.0)
            & (current_gripper_width >= self.BALL_RADIUS * 2)
        )

        reward[ball_in_gripper] += 10.0

        very_close_mask = tcp_to_obj_dist <= self.BALL_RADIUS * 1.25
        reward[very_close_mask] += 30.0 * (1 - tcp_to_obj_dist[very_close_mask] / (self.BALL_RADIUS * 3))

        optimal_width = 2 * self.BALL_RADIUS
        width_error = torch.abs(current_gripper_width[ball_in_gripper] - optimal_width)
        width_error = torch.clamp(width_error - 0.02, min=0.0)
        closing_reward = 90.0 * torch.exp(-5.0 * width_error)
        reward[ball_in_gripper] += closing_reward

        gripper_attempt_mask = ball_in_gripper & (current_gripper_width <= optimal_width + 0.02)
        reward[gripper_attempt_mask] += 90.0

        reward[is_ball_grasped] += 200

        v = torch.linalg.norm(self.ball.linear_velocity, axis=1)
        av = torch.linalg.norm(self.ball.angular_velocity, axis=1)

        static_reward = 1 - torch.tanh(v * 10 + av)

        robot_static_reward = self.agent.is_static(0.2)

        reward[info["success"]] = 300

        self.reward_dict = {
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "is_grasped_reward": is_ball_grasped,
            "ball_in_gripper": ball_in_gripper,
            "dist_tcp_to_ball_x": dist_tcp_to_ball_x,
            "dist_tcp_to_ball_y": dist_tcp_to_ball_y,
            "dist_tcp_to_ball_z": dist_tcp_to_ball_z,
            "current_gripper_width": current_gripper_width,
            "very_close_mask": very_close_mask,
            "static_reward": static_reward,
            "robot_static_reward": robot_static_reward,
            "success": info["success"],
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 300.0


@register_env("InterceptGrabSlow-VLA-v0", max_episode_steps=60)
class InterceptGrabSlowVLAEnv(InterceptGrabVLABaseEnv):
    VELOCITY_RANGE = (0.25, 0.5)


@register_env("InterceptGrabMedium-VLA-v0", max_episode_steps=60)
class InterceptGrabMediumVLAEnv(InterceptGrabVLABaseEnv):
    VELOCITY_RANGE = (0.5, 0.75)


@register_env("InterceptGrabFast-VLA-v0", max_episode_steps=60)
class InterceptGrabFastVLAEnv(InterceptGrabVLABaseEnv):
    VELOCITY_RANGE = (0.75, 1.0)
