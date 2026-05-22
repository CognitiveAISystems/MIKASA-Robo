"""Strict rotation-control tasks for the VLA benchmark."""

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from transforms3d.euler import euler2quat


class RotateStrictVLABaseEnv(BaseEnv):
    """Rotate a peg to a target angle while keeping it in place.

    This is the stricter rotation variant. The robot must not only match the
    target angle but also avoid pushing the peg far from its original position.
    The task therefore tests controlled in-place rotation rather than just any
    successful reorientation.

    Episode flow:
    - The environment samples an initial peg orientation and a target angle.
    - The robot contacts the peg and rotates it.
    - The final state is checked for both angle accuracy and position drift.

    Success (`success=True`):
    - The peg angle must be within `angle_threshold`, the peg must stay close to
      its initial XY position, and the robot must be static.

    How to customize:
    - `MODE` controls whether only positive or both positive and negative target
      angles can be sampled.
    - `angle_threshold` changes how precise the final rotation must be.
    - `PEG_HALF_WIDTH` and `PEG_HALF_LENGTH` change the peg geometry and the
      leverage available during contact.
    """

    LANGUAGE_INSTRUCTION_TEMPLATE = (
        "Rotate the peg by {angle_deg} degrees to match the target angle while keeping the center of the peg in place."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    MODE = "pos_angle"

    PEG_HALF_WIDTH = 0.025
    PEG_HALF_LENGTH = 0.12

    ACTION_L2_COEF = 0.0
    ACTION_DELTA_L2_COEF = 0.0
    QVEL_L2_COEF = 0.0

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, angle_threshold=0.1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.angle_threshold = angle_threshold
        self.target_angle = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.PEG_HALF_LENGTH,
            width=self.PEG_HALF_WIDTH,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(
                p=[0, 0, self.PEG_HALF_LENGTH],
                q=euler2quat(0, np.pi / 2, 0),
            ),
        )

        self.initial_rotations = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            random_values = self._batched_episode_rng.rand()
            random_values = torch.from_numpy(random_values).to(
                device=self.device,
                dtype=torch.float32,
            )

            if self.MODE == "pos_angle":
                self.target_angle = self.angle_threshold + random_values * (0.5 * np.pi - self.angle_threshold)
            elif self.MODE == "pos_neg_angle":
                is_positive = random_values > 0.5
                angle_magnitude = self.angle_threshold + (random_values * (0.25 * np.pi - self.angle_threshold))
                self.target_angle = torch.where(
                    is_positive,
                    angle_magnitude,
                    -angle_magnitude,
                )
            self.target_angle = self.target_angle.to(torch.float16)

            self.task_cue = self.target_angle
            self.reward_dict = None

            initial_z_rotation = self._batched_episode_rng.rand() * 2 * np.pi
            self.initial_rotations = torch.from_numpy(initial_z_rotation).to(
                dtype=torch.float32,
                device=self.device,
            )

            qz = torch.zeros((b, 4), device=self.device)
            qz[:, 0] = torch.cos(torch.from_numpy(initial_z_rotation / 2))
            qz[:, 3] = torch.sin(torch.from_numpy(initial_z_rotation / 2))

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.PEG_HALF_WIDTH

            self.initial_peg_position = xyz.clone()

            self.peg.set_pose(Pose.create_from_pq(p=xyz, q=qz))

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
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")

        y_angle = euler[:, 2]
        relative_angle = y_angle - self.initial_rotations
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        y_angle_diff = self.target_angle - relative_angle
        y_angle_diff = (y_angle_diff + np.pi) % (2 * np.pi) - np.pi

        self.angle_diff = y_angle_diff
        self.oracle_info = y_angle_diff

        peg_position = self.peg.pose.p
        x_pos_diff = peg_position[:, 0] - self.initial_peg_position[:, 0]
        y_pos_diff = peg_position[:, 1] - self.initial_peg_position[:, 1]

        pos_threshold = 0.05
        correct_x_pos = torch.abs(x_pos_diff) < pos_threshold
        correct_y_angle = torch.abs(y_angle_diff) < self.angle_threshold
        correct_y_pos = torch.abs(y_pos_diff) < pos_threshold

        self.correct_angle = correct_y_angle
        is_stable = self.agent.is_static(0.2)

        angle_deg = torch.round(self.target_angle.to(torch.float32) * (180.0 / np.pi)).to(torch.int32)
        language_instruction = [
            self.LANGUAGE_INSTRUCTION_TEMPLATE.format(angle_deg=int(angle))
            for angle in angle_deg.detach().cpu().view(-1).tolist()
        ]

        return {
            "success": correct_y_angle & correct_x_pos & correct_y_pos & is_stable,
            "task_cue": self.target_angle,
            "language_instruction": language_instruction,
            "oracle_info": self.oracle_info,
            "relative_angle": relative_angle,
            "x_pos_error": x_pos_diff,
            "y_angle_error": y_angle_diff,
            "y_pos_error": y_pos_diff,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                oracle_info=self.oracle_info,
                peg_pose=self.peg.pose.raw_pose,
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
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_reward = torch.exp(-3.0 * to_grip_dist)

        reach_threshold = 0.04
        reached_status = to_grip_dist < reach_threshold

        x_pos_reward = torch.exp(-5.0 * torch.abs(info["x_pos_error"]))
        y_pos_reward = torch.exp(-5.0 * torch.abs(info["y_pos_error"]))

        y_angle_diff = info["y_angle_error"]
        y_angle_reward = torch.exp(-1.0 * torch.abs(y_angle_diff))

        agent_is_static = self.agent.is_static(0.2)

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

        reward = 0.5 * reaching_reward - smooth_penalty

        reward = torch.where(reached_status, reward + 5.0 * x_pos_reward, reward)
        reward = torch.where(reached_status, reward + 5.0 * y_pos_reward, reward)

        position_ok = (torch.abs(info["x_pos_error"]) < 0.05) & (torch.abs(info["y_pos_error"]) < 0.05)
        rotation_mask = reached_status & position_ok
        reward = torch.where(rotation_mask, reward + 35.0 * y_angle_reward, reward)

        reward = torch.where(
            self.correct_angle & position_ok & reached_status,
            reward + 10.0,
            reward,
        )
        reward = torch.where(
            self.correct_angle & agent_is_static & position_ok & reached_status,
            reward + 5.0,
            reward,
        )

        reward = torch.where(info["success"], torch.tensor(100.0, device=reward.device), reward)

        self.reward_dict = {
            "to_grip_dist": to_grip_dist,
            "reaching_reward": reaching_reward,
            "reached_status": reached_status,
            "y_angle_reward": y_angle_reward,
            "correct_angle": self.correct_angle,
            "y_angle_diff_deg": y_angle_diff * (180 / np.pi),
            "agent_is_static": agent_is_static,
            "success": info["success"],
            "x_pos_error": info["x_pos_error"],
            "y_pos_error": info["y_pos_error"],
            "x_pos_reward": x_pos_reward,
            "y_pos_reward": y_pos_reward,
            "position_ok": position_ok,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
            "smooth_penalty": smooth_penalty,
        }

        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 100.0


@register_env("RotateStrictPos-VLA-v0", max_episode_steps=90)
class RotateStrictPosVLAEnv(RotateStrictVLABaseEnv):
    MODE = "pos_angle"


@register_env("RotateStrictPosNeg-VLA-v0", max_episode_steps=90)
class RotateStrictPosNegVLAEnv(RotateStrictVLABaseEnv):
    MODE = "pos_neg_angle"
