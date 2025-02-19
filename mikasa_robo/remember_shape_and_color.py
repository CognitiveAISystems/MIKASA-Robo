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
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig, SceneConfig
from mani_skill.utils.structs import Articulation
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from typing import Optional, Union
from mani_skill.envs.scene import ManiSkillScene

from utils import shapes


class RememberShapeAndColorBaseEnv(BaseEnv):
    """

    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    SHAPES = 3  # Will be overridden by child classes

    # Environment constants
    GOAL_THRESH = 0.05    # Radius of the goal region
    SHAPE_SCALE = 0.02  # Radius of the cube
    TIME_OFFSET = 5       # Time to observe the goal cube
    # DELTA_TIME = 5        # Time after observation when no shapes are visible
    COLOR = [0, 0, 255, 255] # blue


    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, delta_time=5, **kwargs):

        self.DELTA_TIME = delta_time

        # Define base shapes and colors
        self.base_shapes = {
            0: "cube",
            1: "sphere", 
            2: "t_shape",
            3: "cross",
            4: "torus",
        }
        
        self.colors = {
            0: np.array([255, 0, 0, 255]) / 255.0,    # Red
            1: np.array([0, 255, 0, 255]) / 255.0,    # Green
            2: np.array([0, 0, 255, 255]) / 255.0,    # Blue
        }
        
        # Create combinations of shapes and colors
        self.shape_color_dict = {}
        idx = 0
        for shape_id, shape in self.base_shapes.items():
            for color_id, color in self.colors.items():
                if idx >= self.SHAPES:
                    break
                self.shape_color_dict[idx] = {
                    "shape": shape,
                    "color": color,
                    "shape_id": shape_id,
                    "color_id": color_id
                }
                idx += 1


        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.initial_poses = {}
        
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, 
                max_rigid_patch_count=2**21, # 18
                max_rigid_contact_count=2**22 # 19
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

        self.shapes = {}
        for key, shape_info in self.shape_color_dict.items():
            shape = shape_info["shape"]
            color = shape_info["color"]
            
            if shape == "cube":
                self.shapes[key] = actors.build_cube(
                    self.scene,
                    half_size=self.SHAPE_SCALE,
                    color=color,
                    name=f"cube_{key}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
                )
            elif shape == "sphere":
                self.shapes[key] = actors.build_sphere(
                    self.scene,
                    radius=self.SHAPE_SCALE,
                    color=color,
                    name=f"sphere_{key}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
                )
            elif shape == "cross":
                self.shapes[key] = shapes.build_cross(
                    self.scene,
                    arm_length=self.SHAPE_SCALE*1.5,
                    width=self.SHAPE_SCALE*0.75,
                    color=color,
                    name=f"cross_{key}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
                )
            elif shape == "torus":
                self.shapes[key] = shapes.build_torus(
                    self.scene,
                    radius=self.SHAPE_SCALE,
                    tube_radius=self.SHAPE_SCALE/2,
                    color=color,
                    name=f"torus_{key}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
                )
            elif shape == "t_shape":
                self.shapes[key] = shapes.build_t_shape(
                    self.scene,
                    width=self.SHAPE_SCALE*2,
                    height=self.SHAPE_SCALE*2,
                    thickness=self.SHAPE_SCALE*0.75,
                    color=color,
                    name=f"t_shape_{key}",
                    body_type="dynamic",
                    initial_pose=sapien.Pose(p=[0, 0, self.SHAPE_SCALE]),
                )
            else:
                raise NotImplementedError(shape)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.prompt = None
            self.reward_dict = None

            self.true_shape_indices = self._batched_episode_rng.choice(list(self.shape_color_dict.keys()))
            self.true_shape_indices = torch.from_numpy(self.true_shape_indices).to(device=self.device, dtype=torch.uint8)

            # Extract shape and color info separately
            self.true_shapes_info = torch.tensor([self.shape_color_dict[idx.item()]["shape_id"] 
                                                for idx in self.true_shape_indices], 
                                            device=self.device, dtype=torch.uint8)
            self.true_colors_info = torch.tensor([self.shape_color_dict[idx.item()]["color_id"] 
                                                for idx in self.true_shape_indices],
                                            device=self.device, dtype=torch.uint8)

            # * Initial position
            xyz_initial = torch.zeros((b, 3))
            self.center_pose = xyz_initial.clone()
            self.center_pose[..., 2] = self.SHAPE_SCALE
            self.center_pose = self.center_pose[0].unsqueeze(0)

            # * shapes
            for key, shape in self.shape_color_dict.items():
                xyz_cube = xyz_initial.clone()

                if len(self.shape_color_dict) % 2 == 0:
                    angle = np.pi * (key + 0.5 - (len(self.shape_color_dict) / 2)) / len(self.shape_color_dict)
                    radius = 0.3
                    xyz_cube[..., 0] = radius * np.cos(angle) - 0.25
                    xyz_cube[..., 1] = radius * np.sin(angle)
                else:
                    angle = np.pi * (key - (len(self.shape_color_dict) // 2)) / len(self.shape_color_dict)
                    radius = 0.3
                    xyz_cube[..., 0] = radius * np.cos(angle) - 0.25
                    xyz_cube[..., 1] = radius * np.sin(angle)

                    if self.SHAPES in [5, 9]:
                        xyz_cube[..., 1] -= (key - (len(self.shape_color_dict) // 2)) * 0.025

                xyz_cube[..., 2] = self.SHAPE_SCALE # 1e-3
                q = [1, 0, 0, 0]
                obj_pose_cube = Pose.create_from_pq(p=xyz_cube, q=q)
                self.shapes[key].set_pose(obj_pose_cube)
                self.initial_poses[key] = xyz_cube.clone()

            # Replace the position initialization code with vectorized version
            with torch.device(self.device):
                min_distance = self.SHAPE_SCALE * 3
                max_attempts = 20
                n_objects = len(self.initial_poses)
                
                # Get all initial positions in a single tensor [batch, n_objects, 3]
                positions = torch.stack([self.initial_poses[key] for key in self.initial_poses.keys()], dim=1)
                
                # Process each environment in parallel
                for env_i in range(b):
                    # Add random noise to all positions at once
                    best_positions = positions[env_i].clone()
                    best_min_distance = 0
                    
                    for _ in range(max_attempts):
                        # Generate noise for all objects at once [n_objects, 2]
                        noise = torch.randn(n_objects, 2, device=self.device) * self.SHAPE_SCALE * 0.5
                        candidate_positions = positions[env_i].clone()
                        candidate_positions[:, :2] += noise
                        
                        # Compute pairwise distances between all objects [n_objects, n_objects]
                        diffs = candidate_positions[:, None, :2] - candidate_positions[None, :, :2]
                        distances = torch.norm(diffs, dim=2)
                        
                        # Set diagonal to large value to ignore self-distances
                        distances.fill_diagonal_(float('inf'))
                        
                        # Find minimum distance between any pair of objects
                        min_dist = distances.min()
                        
                        # Update best positions if this arrangement is better
                        if min_dist > best_min_distance:
                            best_positions = candidate_positions
                            best_min_distance = min_dist
                            
                            if min_dist >= min_distance:
                                break
                    
                    # Shuffle positions
                    perm = torch.randperm(n_objects)
                    best_positions = best_positions[perm]
                    
                    # Update positions and object poses
                    for idx, (key, new_pos) in enumerate(zip(self.initial_poses.keys(), best_positions)):
                        self.initial_poses[key][env_i] = new_pos
                        current_pose = self.shapes[key].pose.raw_pose.clone()
                        current_pose[env_i, :3] = new_pos
                        self.shapes[key].pose = current_pose

            self.oracle_info = torch.stack([self.true_shapes_info, self.true_colors_info], dim=1)

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
        self.original_poses = {key: self.shapes[key].pose.raw_pose.clone() for key in self.shapes.keys()}

        hidden_shapes_poses = {}
        for key, shape in self.shape_color_dict.items():
            hidden_shapes_poses[key] = self.shapes[key].pose.raw_pose.clone()
            hidden_shapes_poses[key][(self.elapsed_steps < self.TIME_OFFSET+self.DELTA_TIME), 2] = 1000
            self.shapes[key].pose = hidden_shapes_poses[key]

        # Update other timing-dependent logic similarly
        for key, shape in self.shape_color_dict.items():
            true_shape_mask = self.true_shape_indices == key
            b_ = hidden_shapes_poses[key].shape[0]

            hidden_shapes_poses[key][true_shape_mask, :3] = self.center_pose.repeat(b_, 1)[true_shape_mask, :3]

            hidden_shapes_poses[key][
                true_shape_mask \
                & (self.TIME_OFFSET + self.DELTA_TIME >= self.elapsed_steps) \
                & (self.elapsed_steps >= self.TIME_OFFSET), 
                2
            ] = 1000

            self.shapes[key].pose = hidden_shapes_poses[key]

        for key, shape in self.shape_color_dict.items():
            mask = self.elapsed_steps >= self.TIME_OFFSET + self.DELTA_TIME
            # TODO: (if uncomment) in this mode, objects will rotate around their axis when interacting with the manipulator, but will not move from their place
            # hidden_shapes_poses[key][mask, :3] = self.initial_poses[key][mask, :3] 
            hidden_shapes_poses[key][mask, :3] = self.original_poses[key][mask, :3]
            self.shapes[key].pose = hidden_shapes_poses[key]
            
        self.masks = {}
        for key, shape in self.shape_color_dict.items():
            self.masks[key] = (self.true_shape_indices == key).unsqueeze(-1)
        
        self.obj_to_goal_pos = torch.zeros_like(self.shapes[0].pose.p, 
                                           device=self.shapes[0].pose.p.device, 
                                           dtype=self.shapes[0].pose.p.dtype)
        
        self.shifted_obj_to_goal_pos = self.obj_to_goal_pos.clone()
        for key, shape in self.shape_color_dict.items():
            self.obj_to_goal_pos += (
                (self.shapes[key].pose.p - self.agent.tcp.pose.p) * self.masks[key]
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
            # for key in self.shapes:
            #     obs[f'cube_{key}_pose'] = self.shapes[key].pose.p

            for key in self.shapes:
                obs[f'goal_{key}_pose'] = self.shapes[key].pose.p * self.masks[key]
            
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
    

@register_env("RememberShapeAndColor3x2-v0", max_episode_steps=60)
class RememberShapeAndColor3x2Env(RememberShapeAndColorBaseEnv):
    SHAPES = 3 * 2

@register_env("RememberShapeAndColor3x3-v0", max_episode_steps=60)
class RememberShapeAndColor3x3Env(RememberShapeAndColorBaseEnv):
    SHAPES = 3 * 3

@register_env("RememberShapeAndColor5x3-v0", max_episode_steps=60)
class RememberShapeAndColor5x3Env(RememberShapeAndColorBaseEnv):
    SHAPES = 5 * 3