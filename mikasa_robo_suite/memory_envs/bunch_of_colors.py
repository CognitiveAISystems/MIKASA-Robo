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
import random
from collections import defaultdict
import gymnasium as gym
import torch
from tqdm.notebook import tqdm
from IPython.display import Video


class BunchOfColorsEnv(BaseEnv):
    """

    """
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    COLORS = 9  # Using all 9 colors
    
    # Environment constants
    GOAL_THRESH = 0.05
    CUBE_HALFSIZE = 0.02
    SEQUENCE_LENGTH = 5   # Number of cubes to show in sequence
    STEP_DURATION = 5    # Duration to show each cube
    EMPTY_DURATION = 5   # Duration of empty table

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
                found_lost_pairs_capacity=2**25, 
                max_rigid_contact_count=2**21,
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

            # Reset touched cubes tracking
            self.touched_cubes = torch.zeros(
                (b, len(self.color_dict)), 
                dtype=torch.bool, 
                device=self.device
            )

            self.initial_poses = {}

            # Select random sequence of unique colors
            all_colors = list(self.color_dict.keys())
            sequence_indices = self._batched_episode_rng.choice(
                all_colors, 
                size=self.SEQUENCE_LENGTH, 
                replace=False
            )
            self.true_color_indices = torch.from_numpy(sequence_indices).to(
                device=self.device, 
                dtype=torch.uint8
            )

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

            self.initial_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes.keys()}

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
        # ! attempt to speed up the code
        self.original_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.cubes.keys()}
        
        show_initial_cubes = self.elapsed_steps < self.STEP_DURATION
        empty_table = (self.elapsed_steps >= self.STEP_DURATION) & (self.elapsed_steps < self.STEP_DURATION + self.EMPTY_DURATION)
        show_all_cubes = self.elapsed_steps >= (self.STEP_DURATION + self.EMPTY_DURATION)
        self.active_phase = show_all_cubes
        
        hidden_shapes_poses = {key: self.cubes[key].pose.raw_pose.clone() for key in self.color_dict.keys()}
        b_ = next(iter(hidden_shapes_poses.values())).shape[0]


        # Prepare indices and angles for all cubes at once
        keys = torch.tensor(list(self.color_dict.keys()), device=self.device)
        angles = torch.pi * (keys - (len(self.color_dict) // 2)) / len(self.color_dict)
        radius = 0.22

        cos_angles = torch.cos(angles).to(device=self.device)
        sin_angles = torch.sin(angles).to(device=self.device)
        
        offsets = torch.stack([
            radius * cos_angles,
            radius * sin_angles,
            torch.zeros_like(keys, device=self.device)
        ], dim=1)  # Shape: [num_cubes, 3]

        # Calculate y-axis adjustments
        y_adjustments = (keys - (len(self.color_dict) // 2)) * 0.015
        offsets[:, 1] -= y_adjustments

        # Create target cube masks for all cubes at once
        is_target_cubes = torch.stack([
            torch.tensor([key in self.true_color_indices[i] for i in range(self.true_color_indices.shape[0])], 
                        device=self.device)
            for key in self.color_dict.keys()
        ])

        # Update poses in a vectorized way
        center_pose_expanded = self.center_pose.repeat(b_, 1)
        show_initial_expanded = show_initial_cubes.unsqueeze(-1)
        empty_table_expanded = empty_table.unsqueeze(-1)
        show_all_expanded = show_all_cubes.unsqueeze(-1)

        for key, shape in self.color_dict.items():
            mask_target = is_target_cubes[key]
            
            # Combine all position updates using masks
            new_pos = torch.where(
                mask_target.unsqueeze(-1) & show_initial_expanded,
                center_pose_expanded + offsets[key],
                torch.where(
                    empty_table_expanded | (~mask_target.unsqueeze(-1) & show_initial_expanded),
                    torch.tensor([0, 0, 1000.0], device=self.device),
                    self.original_poses[key][..., :3]
                )
            )
            
            hidden_shapes_poses[key][..., :3] = new_pos
            self.cubes[key].pose = hidden_shapes_poses[key]
        
        sequence_cubes_mask = torch.zeros((b_, len(self.color_dict)), dtype=torch.bool, device=self.device)
        sequence_cubes_mask.scatter_(1, self.true_color_indices.long(), True)
        self.sequence_cubes_mask = sequence_cubes_mask

        tcp_pos = self.agent.tcp.pose.p

        self.current_touches = {}
        for key in self.color_dict.keys():
            cube_pos = self.cubes[key].pose.p
            cube_touch_pos = Pose.create_from_pq(p=cube_pos+torch.tensor([0, 0, self.CUBE_HALFSIZE+0.005], device=self.device))
            distance = torch.norm(tcp_pos - cube_touch_pos.p, dim=-1)
            touch_mask = distance < (self.CUBE_HALFSIZE) * show_all_cubes
            touch_mask *= self.agent.is_static(0.2)
            self.current_touches[key] = touch_mask
            self.touched_cubes[:, key] |= touch_mask


        all_cubes_from_sequence_is_touched = torch.eq(self.touched_cubes, sequence_cubes_mask).all(1)
        self.all_cubes_from_sequence_is_touched = all_cubes_from_sequence_is_touched

        no_one_cube_not_from_sequence_is_touched = (~(self.touched_cubes & ~sequence_cubes_mask)).any(1)
        self.no_one_cube_not_from_sequence_is_touched = no_one_cube_not_from_sequence_is_touched

        success = (
            all_cubes_from_sequence_is_touched & 
            no_one_cube_not_from_sequence_is_touched & 
            self.agent.is_static(0.2)
        )

        success *= show_all_cubes

        next_target_mask = torch.zeros_like(sequence_cubes_mask, dtype=torch.bool, device=self.device)

        untouched_sequence_cubes = sequence_cubes_mask & ~self.touched_cubes

        first_untouched_indices = torch.argmax(untouched_sequence_cubes.float(), dim=1)

        batch_indices = torch.arange(sequence_cubes_mask.shape[0], device=self.device)
        next_target_mask[batch_indices, first_untouched_indices] = True

        next_target_mask *= show_all_cubes.unsqueeze(1)

        next_target_mask *= untouched_sequence_cubes.any(dim=1).unsqueeze(1)

        self.obj_to_goal_pos = torch.zeros_like(
            cube_touch_pos.p, 
            device=cube_touch_pos.p.device, 
            dtype=cube_touch_pos.p.dtype
        )

        for key in self.color_dict.keys():
            cube_pos = self.cubes[key].pose.p
            cube_touch_pos = Pose.create_from_pq(p=cube_pos+torch.tensor([0, 0, self.CUBE_HALFSIZE+0.005], device=self.device))
            self.obj_to_goal_pos += (
                (cube_touch_pos.p - self.agent.tcp.pose.p) * next_target_mask[:, key].unsqueeze(-1)
            )

        self.next_target_mask = next_target_mask
        is_robot_static = self.agent.is_static(0.2)

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_robot_static": is_robot_static,
            "success": success,
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
                obj_to_goal_pos=self.obj_to_goal_pos,
                oracle_info=self.oracle_info
            )
            
            for key in self.cubes:
                obs[f'cube_{key}_pose'] = self.cubes[key].pose.p
                obs[f'touched_{key}'] = self.touched_cubes[:, key]
                obs[f'cube_{key}_in_seq'] = self.sequence_cubes_mask[:, key]
                obs[f'next_target_{key}'] = self.next_target_mask[:, key]
            
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):

        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_obj_dist)

        correct_touches = (self.touched_cubes & self.sequence_cubes_mask).sum(1)
        correct_touch_reward = (correct_touches.float() / self.SEQUENCE_LENGTH) * 90.0 

        wrong_touches = (self.touched_cubes & ~self.sequence_cubes_mask).sum(1)
        wrong_touch_penalty = 10.0 * wrong_touches.float()

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )

        reward = (
            reaching_reward +
            0.5 * static_reward + 
            correct_touch_reward -
            wrong_touch_penalty
        )

        reward *= self.active_phase


        reward[info["success"]] = 100.0

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "correct_touches": correct_touches,
            "wrong_touches": wrong_touches,
            "is_robot_static": info["is_robot_static"],
            "touched_cubes": self.touched_cubes.sum(1),
            "sequence_cubes_mask": self.sequence_cubes_mask.sum(1),
            "all_seq_touched": self.all_cubes_from_sequence_is_touched,
            "no_extra_touched": self.no_one_cube_not_from_sequence_is_touched
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 100.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    

@register_env("BunchOfColors3-v0", max_episode_steps=120)
class BunchOfColors3Env(BunchOfColorsEnv):
    SEQUENCE_LENGTH = 3

@register_env("BunchOfColors5-v0", max_episode_steps=120)
class BunchOfColors5Env(BunchOfColorsEnv):
    SEQUENCE_LENGTH = 5

@register_env("BunchOfColors7-v0", max_episode_steps=120)
class BunchOfColors7Env(BunchOfColorsEnv):
    SEQUENCE_LENGTH = 7