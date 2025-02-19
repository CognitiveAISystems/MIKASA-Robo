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


class RememberColorBaseEnv(BaseEnv):
    """

    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    COLORS = 3  # Will be overridden by child classes

    # Environment constants
    GOAL_THRESH = 0.05    # Radius of the goal region № 0.05
    CUBE_HALFSIZE = 0.02  # Radius of the cube
    TIME_OFFSET = 5       # Time to observe the goal cube

    # Color definitions (RGBA format)
    COLOR_MAPPING = {
        0: ("Red",     [255, 0, 0, 255]),
        1: ("Lime",    [0, 255, 0, 255]),
        2: ("Blue",    [0, 0, 255, 255]),
        3: ("Yellow",  [255, 255, 0, 255]),
        4: ("Magenta", [255, 0, 255, 255]), 
        5: ("Cyan",    [0, 255, 255, 255]),
        6: ("Maroon",  [128, 0, 0, 255]),
        7: ("Olive",   [255, 128, 0, 255]),
        8: ("Teal",    [0, 128, 128, 255])
    }

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, delta_time=5, **kwargs):

        self.DELTA_TIME = delta_time
        # Initialize color dictionary with specified number of colors
        self.color_dict = {
            k: np.array(v[1]) / 255.0 
            for k, v in list(self.COLOR_MAPPING.items())[:self.COLORS]
        }

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.initial_poses = {}
        
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

        self.cubes = {}
        for key, color in self.color_dict.items():
            self.cubes[key] = actors.build_cube(
                self.scene,
                half_size=self.CUBE_HALFSIZE,
                color=color,
                name=f"cube_{key}",
                body_type="dynamic",
                initial_pose=sapien.Pose(p=[0, 0, self.CUBE_HALFSIZE]),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.prompt = None
            self.reward_dict = None

            self.true_color_indices = self._batched_episode_rng.choice(list(self.color_dict.keys()))
            self.true_color_indices = torch.from_numpy(self.true_color_indices).to(device=self.device, dtype=torch.uint8)

            # * Initial position
            xyz_initial = torch.zeros((b, 3))
            self.center_pose = xyz_initial.clone()
            self.center_pose[..., 2] = self.CUBE_HALFSIZE
            self.center_pose = self.center_pose[0].unsqueeze(0)

            # * Cubes
            for key, color in self.color_dict.items():
                xyz_cube = xyz_initial.clone()
                if self.COLORS != 3:
                    angle = np.pi * (key - (len(self.color_dict) // 2)) / len(self.color_dict)
                    radius = 0.3

                    xyz_cube[..., 0] = radius * np.cos(angle) - 0.25
                    xyz_cube[..., 1] = radius * np.sin(angle)

                    if self.COLORS in [5, 9]:
                        xyz_cube[..., 1] -= (key - (len(self.color_dict) // 2)) * 0.025
                else:
                    xyz_cube[..., 1] -= (key - (len(self.color_dict) // 2)) * 0.1 
                xyz_cube[..., 2] = self.CUBE_HALFSIZE
                q = [1, 0, 0, 0]
                obj_pose_cube = Pose.create_from_pq(p=xyz_cube, q=q)
                self.cubes[key].set_pose(obj_pose_cube)
                self.initial_poses[key] = xyz_cube.clone()


            # After calculating all initial poses, but before setting them:
            with torch.device(self.device):
                min_distance = self.CUBE_HALFSIZE * 3  # Min distance between objects
                max_attempts = 50  # Max attempts to find a valid position
                
                # Create a permutation for each environment
                for env_i in range(b):
                    # Get list of positions for this environment
                    positions = [self.initial_poses[key][env_i].clone() for key in self.initial_poses.keys()]
                    
                    # For each position
                    for i in range(len(positions)):
                        attempt = 0
                        while attempt < max_attempts:
                            # Add random offset to current position
                            noise = torch.randn(2, device=self.device) * self.CUBE_HALFSIZE * 0.5
                            new_pos = positions[i].clone()
                            new_pos[:2] += noise
                            
                            # Check distance to all previously placed objects
                            valid_position = True
                            for j in range(i):
                                distance = torch.norm(new_pos[:2] - positions[j][:2])
                                if distance < min_distance:
                                    valid_position = False
                                    break
                            
                            if valid_position:
                                positions[i] = new_pos
                                break
                            attempt += 1
                    
                    # Shuffle positions
                    shuffled_indices = torch.randperm(len(positions))
                    shuffled_positions = [positions[i] for i in shuffled_indices]
                    
                    # Assign shuffled positions back
                    for key, new_pos in zip(self.initial_poses.keys(), shuffled_positions):
                        self.initial_poses[key][env_i] = new_pos
                        # Update actual object poses as well
                        current_pose = self.cubes[key].pose.raw_pose.clone()
                        current_pose[env_i, :3] = new_pos
                        self.cubes[key].pose = current_pose

            self.oracle_info = self.true_color_indices

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

    def evaluate(self):
        self.original_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes.keys()}
        
        hidden_shapes_poses = {}
        for key, shape in self.color_dict.items():
            hidden_shapes_poses[key] = self.cubes[key].pose.raw_pose.clone()
            hidden_shapes_poses[key][(self.elapsed_steps < self.TIME_OFFSET+self.DELTA_TIME), 2] = 1000
            self.cubes[key].pose = hidden_shapes_poses[key]

        # Update other timing-dependent logic similarly
        for key, shape in self.color_dict.items():
            true_shape_mask = self.true_color_indices == key
            b_ = hidden_shapes_poses[key].shape[0]

            hidden_shapes_poses[key][true_shape_mask, :3] = self.center_pose.repeat(b_, 1)[true_shape_mask, :3]

            hidden_shapes_poses[key][
                true_shape_mask \
                & (self.TIME_OFFSET + self.DELTA_TIME >= self.elapsed_steps) \
                & (self.elapsed_steps >= self.TIME_OFFSET), 
                2
            ] = 1000

            self.cubes[key].pose = hidden_shapes_poses[key]

        for key, shape in self.color_dict.items():
            mask = self.elapsed_steps >= self.TIME_OFFSET + self.DELTA_TIME
            # TODO: (if uncomment) in this mode, objects will rotate around their axis when interacting with the manipulator, but will not move from their place
            # hidden_shapes_poses[key][mask, :3] = self.initial_poses[key][mask, :3] 
            hidden_shapes_poses[key][mask, :3] = self.original_poses[key][mask, :3]
            self.cubes[key].pose = hidden_shapes_poses[key]

        self.masks = {}
        for key, color in self.color_dict.items():
            self.masks[key] = (self.true_color_indices == key).unsqueeze(-1)
        
        self.obj_to_goal_pos = torch.zeros_like(self.cubes[0].pose.p, 
                                           device=self.cubes[0].pose.p.device, 
                                           dtype=self.cubes[0].pose.p.dtype)
        
        for key, color in self.color_dict.items():
            self.obj_to_goal_pos += (
                (self.cubes[key].pose.p - self.agent.tcp.pose.p) * self.masks[key]
            )

        is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "success": is_obj_placed & is_robot_static,
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
                # obj_to_goal_pos=self.obj_to_goal_pos,
                oracle_info=self.oracle_info
            )

            # for key in self.cubes:
            #     obs[f'cube_{key}_pose'] = self.cubes[key].pose.p

            for key in self.cubes:
                obs[f'goal_{key}_pose'] = self.cubes[key].pose.p * self.masks[key]
            
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_obj_dist)

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )

        reached = tcp_to_obj_dist < self.GOAL_THRESH

        reward = (
            1.0 * reaching_reward + 
            0.5 * static_reward + 
            0.5 * info['is_robot_static'] * info['is_obj_placed']
        )
        
        reward[info["success"]] = 3.0

        self.reward_dict = {
            'tcp_to_obj_dist': tcp_to_obj_dist,
            'reaching_reward': reaching_reward,
            'is_robot_static': info['is_robot_static'],
            'reached': reached,
            'success': info['success'],
            'static_reward': static_reward,
            'obj_to_goal_pos_y': info['obj_to_goal_pos'][:, 1],
            'obj_to_goal_pos_x': info['obj_to_goal_pos'][:, 0],
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
    
@register_env("RememberColor3-v0", max_episode_steps=60)
class RememberColor3Env(RememberColorBaseEnv):
    COLORS = 3

@register_env("RememberColor5-v0", max_episode_steps=60)
class RememberColor5Env(RememberColorBaseEnv):
    COLORS = 5

@register_env("RememberColor9-v0", max_episode_steps=60)
class RememberColor9Env(RememberColorBaseEnv):
    COLORS = 9