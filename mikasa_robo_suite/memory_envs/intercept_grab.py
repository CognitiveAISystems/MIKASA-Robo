from typing import Any, Dict
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from typing import Optional, Union
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actors.common import _build_by_type


class InterceptGrabBaseEnv(BaseEnv):
    """
    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    VELOCITY_RANGE = (0.0, 0.0)  # Will be overridden by child classes

    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS:    float = 0.02  # radius of the ball

    def __init__(self, *args, robot_uids="panda_wristcam", # "panda"
                  robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
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
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
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

            self.prompt = None
            self.reward_dict = None

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.15 - 0.2
            xyz[..., 1] = torch.rand((b)) * 0.25 - 1.0 + 0.1
            xyz[..., 2] = self.BALL_RADIUS
            q = [1, 0, 0, 0]

            obj_pose_ball = Pose.create_from_pq(p=xyz, q=q)
            self.ball.set_pose(obj_pose_ball)

            initial_velocity = torch.zeros((b, 3))
            initial_velocity[..., 0] = torch.rand((b)) * 0.05

            min_vel, max_vel = self.VELOCITY_RANGE
            initial_velocity[..., 1] = torch.rand((b)) * (max_vel - min_vel) + min_vel

            initial_velocity[..., 2] = torch.rand((b)) * 0
            self.ball.set_linear_velocity(initial_velocity)

            self.oracle_info = initial_velocity

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
        is_ball_grasped = self.agent.is_grasping(self.ball)
        is_robot_static = self.agent.is_static(0.2)
        success = is_ball_grasped & is_robot_static
        success = is_ball_grasped & is_robot_static

        return {
            "success": success,
            "prompt": self.prompt,
            "oracle_info": self.oracle_info,
            "is_ball_grasped": is_ball_grasped,
            "is_robot_static": is_robot_static,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
        
            obs.update(
                ball_pose=self.ball.pose.raw_pose,
                oracle_info=self.oracle_info,
                is_ball_grasped=info["is_ball_grasped"],
                ball_linear_vel=self.ball.linear_velocity,
                ball_angular_vel=self.ball.angular_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                
                # add info about open/close gripper
                gripper_width=self.agent.robot.get_qpos()[:, -2:].sum(dim=1),
                # add info about relative velocity between gripper and ball
                relative_velocity=self.ball.linear_velocity - self.agent.tcp.linear_velocity,
                
            )
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # Calculate distance between TCP and ball
        tcp_to_obj_dist = torch.linalg.norm(self.ball.pose.p - self.agent.tcp.pose.p, axis=1)

        # Base reward: the closer TCP to the ball, the higher reward (maximum 2)
        reward = 10 * (1 - torch.tanh(3 * tcp_to_obj_dist)) # TODO: 5
        reward = 10 * (1 - torch.tanh(3 * tcp_to_obj_dist)) # TODO: 5

        current_gripper_width = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1)

        # Significant reward for the ball being between the gripper parts
        is_ball_grasped = info["is_ball_grasped"]

        # Check if the ball is between the gripper parts
        dist_tcp_to_ball_x = torch.abs(self.agent.tcp.pose.p[:, 0] - self.ball.pose.p[:, 0])
        dist_tcp_to_ball_y = torch.abs(self.agent.tcp.pose.p[:, 1] - self.ball.pose.p[:, 1])
        dist_tcp_to_ball_z = torch.abs(self.agent.tcp.pose.p[:, 2] - self.ball.pose.p[:, 2])

        ball_in_gripper = \
            (dist_tcp_to_ball_x <= self.BALL_RADIUS * 2.0) & \
            (dist_tcp_to_ball_y <= self.BALL_RADIUS * 2.0) & \
            (dist_tcp_to_ball_z <= self.BALL_RADIUS * 1.5) & \
            (tcp_to_obj_dist <= self.BALL_RADIUS * 2.0) & \
            (current_gripper_width >= self.BALL_RADIUS * 2)

        reward[ball_in_gripper] += 10.0

        # Add extra reward when very close to encourage final approach
        very_close_mask = tcp_to_obj_dist <= self.BALL_RADIUS * 1.25
        reward[very_close_mask] += 30.0 * (1 - tcp_to_obj_dist[very_close_mask] / (self.BALL_RADIUS * 3))

        optimal_width = 2 * self.BALL_RADIUS
        width_error = torch.abs(current_gripper_width[ball_in_gripper] - optimal_width)
        width_error = torch.clamp(width_error - 0.02, min=0.0)
        closing_reward = 90.0 * torch.exp(-5.0 * width_error)  # Gaussian reward with peak at optimal_width
        reward[ball_in_gripper] += closing_reward

        # Add reward for trying to grasp when the ball is close
        gripper_attempt_mask = ball_in_gripper & (current_gripper_width <= optimal_width + 0.02)  # Mask for cases when the gripper is closing
        reward[gripper_attempt_mask] += 45.0  # Additional reward for trying to grasp
        
        reward[is_ball_grasped] += 100

        # Calculate linear and angular velocities of the ball
        v = torch.linalg.norm(self.ball.linear_velocity, axis=1)
        av = torch.linalg.norm(self.ball.angular_velocity, axis=1)

        # Reward for staticness: higher when the ball is not moving
        static_reward = 1 - torch.tanh(v * 10 + av)

        # Check how static the robot is
        robot_static_reward = self.agent.is_static(0.2)  # keep the robot static at the end state, since the sphere may spin when being placed on top

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
        max_reward = 300.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("InterceptGrabSlow-v0", max_episode_steps=90)
class InterceptGrabSlowEnv(InterceptGrabBaseEnv):
    VELOCITY_RANGE = (0.25, 0.5)  # Slow speed range

@register_env("InterceptGrabMedium-v0", max_episode_steps=90)
class InterceptGrabMediumEnv(InterceptGrabBaseEnv):
    VELOCITY_RANGE = (0.5, 0.75)  # Medium speed range

@register_env("InterceptGrabFast-v0", max_episode_steps=90)
class InterceptGrabFastEnv(InterceptGrabBaseEnv):
    VELOCITY_RANGE = (0.75, 1.0)  # Fast speed range