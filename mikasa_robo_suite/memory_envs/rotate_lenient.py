from typing import Any, Dict, Union
import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat
from mani_skill.agents.robots import Fetch, Panda
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


class RotateLanientEnv(BaseEnv):
    """

    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    MODE = "pos_angle" # "pos_angle" or "pos_neg_angle" 

    PEG_HALF_WIDTH = 0.025
    PEG_HALF_LENGTH = 0.12

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, angle_threshold=0.1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.angle_threshold = angle_threshold  # permissible deviation from the target angle in radians
        self.target_angle = None  # will be set in _initialize_episode
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
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.PEG_HALF_LENGTH,
            width=self.PEG_HALF_WIDTH,
            color_1=np.array([12, 42, 160, 255]) / 255, # np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255, # np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.PEG_HALF_LENGTH], q=euler2quat(0, np.pi/2, 0)),
        )

        # Store initial rotation for each environment
        self.initial_rotations = torch.zeros(self.num_envs, dtype=torch.float32)
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            random_values = self._batched_episode_rng.rand()
            random_values = torch.from_numpy(random_values).to(device=self.device, dtype=torch.float32)
            
            # Calculate target rotation (how much we need to rotate from initial position)
            if self.MODE == "pos_angle":
                self.target_angle = 1*self.angle_threshold + random_values * (0.5 * np.pi - 1*self.angle_threshold)
            elif self.MODE == "pos_neg_angle":
                is_positive = random_values > 0.5
                angle_magnitude = self.angle_threshold + (random_values * (0.25 * np.pi - self.angle_threshold))
                self.target_angle = torch.where(is_positive, angle_magnitude, -angle_magnitude)
            self.target_angle = self.target_angle.to(torch.float16)

            self.prompt = self.target_angle
            self.reward_dict = None

            # Generate peg random initial rotation around Z axis
            initial_z_rotation = self._batched_episode_rng.rand() * 2 * np.pi
            self.initial_rotations = torch.from_numpy(initial_z_rotation).to(dtype=torch.float32, device=self.device)

            # Create batched quaternions
            # For Z rotation (different for each in batch)
            qz = torch.zeros((b, 4), device=self.device)
            qz[:, 0] = torch.cos(torch.from_numpy(initial_z_rotation/2))  # real part
            qz[:, 3] = torch.sin(torch.from_numpy(initial_z_rotation/2))  # z component

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 0] -= 0
            xyz[..., 2] = self.PEG_HALF_WIDTH
            
            obj_pose = Pose.create_from_pq(
                p=xyz,
                q=qz  # Random rotation around Z
            )
            self.peg.set_pose(obj_pose)

            # Initialize robot arm to a higher position above the table than the default typically used for other table top tasks
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
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
        
        # Get current rotation angle on the Y axis
        y_angle = euler[:, 2]

        # Calculate relative rotation from initial position
        relative_angle = y_angle - self.initial_rotations        

        # Convert to range [-π, π]
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

        # Calculate the difference between current relative rotation and target rotation
        y_angle_diff = self.target_angle - relative_angle
        y_angle_diff = (y_angle_diff + np.pi) % (2 * np.pi) - np.pi
        
        self.angle_diff = y_angle_diff
        self.oracle_info = y_angle_diff

        correct_y_angle = torch.abs(y_angle_diff) < self.angle_threshold
        
        self.correct_angle = correct_y_angle

        is_stable = self.agent.is_static(0.2)

        return {
            "success": self.correct_angle & is_stable,
            "prompt": self.target_angle,
            "oracle_info": self.oracle_info,
            "relative_angle": relative_angle,
            "y_angle_error": y_angle_diff,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                oracle_info=self.oracle_info,
                peg_pose=self.peg.pose.raw_pose,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        info['success'] *= (info['elapsed_steps'] > 5)

        # * Reward for reaching the peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_reward = torch.exp(-5.0 * to_grip_dist)

        # * Reward for Y axis rotation
        y_angle_diff = info["y_angle_error"]
        y_angle_reward = torch.exp(-1.5 * torch.abs(y_angle_diff))

        # * Reward for stability
        agent_is_static = self.agent.is_static(0.2)

        # * Update reached status
        reach_threshold = 0.1
        reached_status = to_grip_dist < reach_threshold

        reward = reaching_reward
        reward[reached_status] += 15.0 * y_angle_reward[reached_status]
        reward[self.correct_angle] += 5.0
        reward[self.correct_angle & agent_is_static] += 2.0
        
        reward[info["success"]] = 25.0

        self.reward_dict = {
            "to_grip_dist": to_grip_dist,
            "reaching_reward": reaching_reward,
            "reached_status": reached_status,
            "y_angle_reward": y_angle_reward,
            "self.correct_angle": self.correct_angle,
            "self.angle_diff": self.angle_diff * (180 / np.pi),
            "agent_is_static": agent_is_static,
            "success": info["success"]
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 25.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    

@register_env("RotateLenientPos-v0", max_episode_steps=90)
class RotateLenientEnvPos(RotateLanientEnv):
    MODE = "pos_angle"

@register_env("RotateLenientPosNeg-v0", max_episode_steps=90)
class RotateLenientEnvPosNeg(RotateLanientEnv):
    MODE = "pos_neg_angle"