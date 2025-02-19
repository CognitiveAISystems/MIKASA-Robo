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


class InterceptBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    VELOCITY_RANGE = (0.0, 0.0)  # Will be overridden by child classes

    agent: Union[Panda, PandaWristCam]

    GOAL_RADIUS:    float = 0.1  # radius of the goal region
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

            self.prompt = None
            self.reward_dict = None

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.15 - 0.2
            xyz[..., 1] = torch.rand((b)) * 0.25 - 1.0 + self.GOAL_RADIUS
            xyz[..., 2] = 2 * self.BALL_RADIUS
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

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = xyz[..., 0].clone() + 0.2
            xyz_goal[..., 1] = torch.rand((b)) * 0.4 - 1.0 + self.GOAL_RADIUS + 0.7
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

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
        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.GOAL_RADIUS
        )

        return {
            "success": is_obj_placed,
            "prompt": self.prompt,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
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

        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # Calculate unit vector from ball to goal
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
        
        # Calculate ideal hitting position slightly behind the ball (BALL_RADIUS + 0.05)
        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.BALL_RADIUS + 0.05),
        )
        
        # Calculate distance between robot end-effector and ideal hitting position
        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
        
        # Update reached status - marks if robot has gotten close enough to hit the ball
        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
        
        # Reaching reward: encourages robot to move to ideal hitting position
        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

        # Distance between ball and goal (only considering x-y plane)
        obj_to_goal_dist = torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )

        place_reward = 1 - torch.tanh(obj_to_goal_dist)

        reward = (
            5.0 * place_reward * self.reached_status + 
            2.0 * reaching_reward * torch.logical_not(self.reached_status)
        )

        # Maximum reward for successful task completion
        reward[info["success"]] = 30.0

        self.reward_dict = {
            'reaching_reward': reaching_reward,
            'reached_status': self.reached_status,
            'place_reward': place_reward,
            'tcp_to_hit_pose_dist': tcp_to_hit_pose_dist
        }

        return reward
    
    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    

@register_env("InterceptSlow-v0", max_episode_steps=90)
class InterceptSlowEnv(InterceptBaseEnv):
    VELOCITY_RANGE = (0.25, 0.5)  # Slow speed range

@register_env("InterceptMedium-v0", max_episode_steps=90)
class InterceptMediumEnv(InterceptBaseEnv):
    VELOCITY_RANGE = (0.5, 0.75)  # Medium speed range

@register_env("InterceptFast-v0", max_episode_steps=90)
class InterceptFastEnv(InterceptBaseEnv):
    VELOCITY_RANGE = (0.75, 1.0)  # Fast speed range