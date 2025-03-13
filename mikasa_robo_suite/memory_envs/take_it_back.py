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
from mikasa_robo_suite.utils import shapes


@register_env("TakeItBack-v0", max_episode_steps=180)
class TakeItBackEnv(BaseEnv):
    """

    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    GOAL_RADIUS:    float = 0.08 # 0.1  # radius of the goal region
    CUBE_HALFSIZE:  float = 0.02  # radius of the cube

    def __init__(self, *args, robot_uids="panda_wristcam", # "panda"
                  robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, 
                max_rigid_patch_count=2**18
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

            self.prompt = None
            self.reward_dict = None

            initial_positions = torch.from_numpy(
                self._batched_episode_rng.choice([-1, 0, 1])
                ).to(self.device)

            # initial_positions = np.random.choice([-1.0, 0.0, 1.0], size=(b))
            # initial_positions = torch.from_numpy(initial_positions).to(self.device)

            # * Initial position
            xyz_initial = torch.zeros((b, 3))
            xyz_initial[..., 0] = (torch.rand((b)) - 0.5) * 0.1  # Random x in [-0.05, 0.05]
            xyz_initial[..., 1] = initial_positions * 0.1 + torch.rand((b)) * 0.05 # Random y in [-0.125, 0.125]
            xyz_initial[..., 2] = 1e-3

            # * Goal position
            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = torch.rand((b)) * 0.05 + 0.2
            xyz_goal[..., 1] = (torch.rand((b)) - 0.5) * 0.05
            xyz_goal[..., 2] = 1e-3

            # * Changed Goal
            xyz_changed_goal = xyz_goal.clone()
            xyz_changed_goal[..., 2] = 1000
            
            self.initial_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_initial,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            self.changed_goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_changed_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

            # * Cube
            xyz_cube = xyz_initial.clone()
            xyz_cube[..., 2] = self.CUBE_HALFSIZE
            q = [1, 0, 0, 0]
            obj_pose_cube = Pose.create_from_pq(p=xyz_cube, q=q)
            self.cube.set_pose(obj_pose_cube)

            self.oracle_info = xyz_initial.clone()

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
            
        self.goal_reached_status[env_idx] = False
        self.goal_achieved[env_idx] = False

    def evaluate(self):

        self.original_poses = {
            'goal_region': self.goal_region.pose.raw_pose.clone(),
            'changed_goal_region': self.changed_goal_region.pose.raw_pose.clone(),
            'cube': self.cube.pose.raw_pose.clone()
        }

        new_goal_region_pose = self.original_poses['goal_region']
        new_goal_region_pose[self.goal_achieved, 2] = 1000

        new_changed_goal_region_pose = self.original_poses['changed_goal_region']
        new_changed_goal_region_pose[self.goal_achieved, 2] = 1e-3

        self.goal_region.pose = new_goal_region_pose
        self.changed_goal_region.pose = new_changed_goal_region_pose
        
        self.goal_achieved = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.GOAL_RADIUS
        )

        self.goal_reached_status = torch.logical_or(self.goal_reached_status, self.goal_achieved)

        is_cube_returned = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.initial_region.pose.p[..., :2], axis=1
            )
            < self.GOAL_RADIUS
        )

        return {
            "success": is_cube_returned & self.goal_reached_status,
            "prompt": self.prompt,
            "oracle_info": self.oracle_info,
            "goal_achieved": self.goal_achieved,
            "goal_reached_status": self.goal_reached_status,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        actual_goal_pos = self.goal_region.pose.p
        actual_goal_pos[actual_goal_pos[:, 2] > 10, 2] \
            = self.changed_goal_region.pose.p[actual_goal_pos[:, 2] > 10, 2]   

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
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

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # * Reward for reaching the cube
        tcp_push_pose = Pose.create_from_pq(
            p=self.cube.pose.p
            + torch.tensor([-self.CUBE_HALFSIZE - 0.005, 0, 0], device=self.device)
        )
        
        tcp_to_cube_pose_push = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_cube_pose_push_dist = torch.linalg.norm(tcp_to_cube_pose_push, axis=1)
        reaching_reward_push = 1 - torch.tanh(5.0 * tcp_to_cube_pose_push_dist)

        # * Reward for pushing cube to the goal region
        cube_to_goal_dist = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_in_goal_reward = 1 - torch.tanh(5.0 * cube_to_goal_dist)

        # * Reward for pushing cube back to the initial region
        cube_to_initial_dist = torch.linalg.norm(
            self.cube.pose.p[..., :2] - self.initial_region.pose.p[..., :2], axis=1
        )
        place_back_reward = 1 - torch.tanh(5.0 * cube_to_initial_dist)

        # * Reward for removing cube from the initial region
        goal_achieved_status = info['goal_reached_status']
        reached_push = tcp_to_cube_pose_push_dist < 0.01

        tcp_pull_pos = self.cube.pose.p + torch.tensor(
            [self.CUBE_HALFSIZE + 2 * 0.005, 0, 0], device=self.device
        )
        tcp_to_cube_pose_pull = tcp_pull_pos - self.agent.tcp.pose.p
        tcp_to_cube_pose_pull_dist = torch.linalg.norm(tcp_to_cube_pose_pull, axis=1)
        reaching_reward_pull = 1 - torch.tanh(5.0 * tcp_to_cube_pose_pull_dist)

        reached_pull = tcp_to_cube_pose_pull_dist < 0.01

        # ? PUSH STAGE
        reward = torch.zeros_like(reaching_reward_push)
        reward[~goal_achieved_status] += 2.0 * reaching_reward_push[~goal_achieved_status]
        reward[~goal_achieved_status & reached_push] += 5.0 * place_in_goal_reward[~goal_achieved_status & reached_push]

        # ? PULL STAGE
        reward[goal_achieved_status] += 10.0 * reaching_reward_pull[goal_achieved_status]
        reward[goal_achieved_status & reached_pull] += 15.0 * place_back_reward[goal_achieved_status & reached_pull]
        
        reward[info["success"]] = 30.0

        self.reward_dict = {
            'reached_push': reached_push,
            'reached_pull': reached_pull,
            'tcp_push_dist': tcp_to_cube_pose_push_dist,
            'tcp_pull_dist': tcp_to_cube_pose_pull_dist,
            'reaching_reward_push': reaching_reward_push,
            'reaching_reward_pull': reaching_reward_pull,
            'goal_achieved': info['goal_achieved'],
            'goal_reached_status': info['goal_reached_status'],
            'place_in_goal_reward': place_in_goal_reward,
            'place_back_reward': place_back_reward
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward