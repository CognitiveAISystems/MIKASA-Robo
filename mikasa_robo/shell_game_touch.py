from typing import Any, Dict, List, Union, Literal
import numpy as np
import sapien
import torch
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import gymnasium as gym
import mani_skill.envs
import torch
import time

WARNED_ONCE = False


@register_env("ShellGameTouch-v0", max_episode_steps=90, asset_download_ids=["ycb"])
class ShellGameTouchEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]
    
    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000
    TIME_OFFSET = 5
    GOAL_THRESH = 0.08

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            list(
                load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()
            )
        )

        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
            
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _initialize_mug(self, model_ids, id_cup, name_suffix):
        objs: List[Actor] = []
        for i, _ in enumerate(model_ids):
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{id_cup}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            objs.append(builder.build(name=f"{id_cup}-{name_suffix}-{i}"))
            self.remove_from_state_dict_registry(objs[-1])
        mug = Actor.merge(objs, name=f"mug_{name_suffix}")
        self.add_to_state_dict_registry(mug)
        return mug, objs
    
    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        # ?: "024_bowl" "025_mug" "065-a_cups" (a..j) red: 065-g_cups

        id_cup = "025_mug"
        self.mug_left, self._objs_1 = self._initialize_mug(model_ids, id_cup, "left")
        self.mug_center, self._objs_2 = self._initialize_mug(model_ids, id_cup, "center") 
        self.mug_right, self._objs_3 = self._initialize_mug(model_ids, id_cup, "right")

        self.red_ball = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([255, 0, 0, 255]) / 255,  # Red color
            name="red_ball",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]), 
        )

    def _after_reconfigure(self, options: dict):
        # Pre-allocate list with known size
        num_objects = len(self._objs_1) + len(self._objs_2) + len(self._objs_3)
        self.object_zs = torch.empty(num_objects, device=self.device)
        
        # Use enumerate and flat iteration
        for idx, obj in enumerate((*self._objs_1, *self._objs_2, *self._objs_3)):
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs[idx] = -collision_mesh.bounding_box.bounds[0, 2]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.prompt = None
            self.reward_dict = None

            self.cup_with_ball_number = self._batched_episode_rng.choice([0, 1, 2])
            self.cup_with_ball_number = torch.from_numpy(self.cup_with_ball_number).to(device=self.device, dtype=torch.uint8)            

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = 2*self.BALL_RADIUS
            q = torch.tensor([0, 1, 0.5, 0]).repeat(b, 1)

            self.mug_left.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, -self.MIN_DIST, 0]).repeat(b, 1), q=q))
            self.mug_center.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, 0, 0]), q=q))
            self.mug_right.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, self.MIN_DIST, 0]).repeat(b, 1), q=q))

            # * Ball
            q = [1, 0, 0, 0]
            ball_xyz = xyz.clone()
            offsets = torch.zeros((b, 3), device=xyz.device)

            offsets[:, 1] = torch.where(self.cup_with_ball_number == 0, -self.MIN_DIST,
                           torch.where(self.cup_with_ball_number == 1, 0.0,
                           torch.where(self.cup_with_ball_number == 2, self.MIN_DIST, offsets[:, 1])))

            offsets[:, 2] = self.BALL_RADIUS - self.object_zs[env_idx]
            ball_xyz += offsets
            red_ball_pose = Pose.create_from_pq(p=ball_xyz, q=q)
            self.red_ball.set_pose(red_ball_pose)
            self.ball_initial_pose = ball_xyz

            self.oracle_info = self.cup_with_ball_number

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
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

    def evaluate(self):

        self.original_poses = {
            'mug_left': self.mug_left.pose.raw_pose.clone(),
            'mug_center': self.mug_center.pose.raw_pose.clone(),
            'mug_right': self.mug_right.pose.raw_pose.clone(),
            'ball': self.red_ball.pose.raw_pose.clone()
        }

        hide_mask = self.elapsed_steps < self.TIME_OFFSET
        
        # Update poses based on masks
        for mug, orig_pose in zip(
            [self.mug_left, self.mug_center, self.mug_right],
            [self.original_poses['mug_left'], self.original_poses['mug_center'], self.original_poses['mug_right']]
        ):
            new_pose = orig_pose.clone()
            new_pose[hide_mask & (new_pose[..., 2] < 100), 2] += self.HEIGHT_OFFSET
            new_pose[~hide_mask & (new_pose[..., 2] > 100), 2] -=self.HEIGHT_OFFSET
            mug.pose = new_pose

            ball_on_mug = (self.original_poses['ball'][..., 2] >= orig_pose[..., 2])
            ball_pose = self.original_poses['ball'].clone()
            ball_pose[ball_on_mug, :3] = self.ball_initial_pose[ball_on_mug, :3]

            # self.red_ball.pose = ball_pose


        # Create masks for each cup position
        self.left_mask = (self.cup_with_ball_number == 0).unsqueeze(-1)
        self.center_mask = (self.cup_with_ball_number == 1).unsqueeze(-1)
        self.right_mask = (self.cup_with_ball_number == 2).unsqueeze(-1)

        self.obj_to_goal_pos = torch.zeros_like(self.mug_left.pose.p, 
                                              device=self.mug_left.pose.p.device, 
                                              dtype=self.mug_left.pose.p.dtype)
        
        # Calculate obj_to_goal_pos using masks
        self.obj_to_goal_pos = (
                (self.mug_left.pose.p - self.agent.tcp.pose.p) * self.left_mask +
                (self.mug_center.pose.p - self.agent.tcp.pose.p) * self.center_mask +
                (self.mug_right.pose.p - self.agent.tcp.pose.p) * self.right_mask
            )
        
        self.is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        self.is_robot_static = self.agent.is_static(0.2)

        return dict(
            obj_to_goal_pos=self.obj_to_goal_pos,
            is_obj_placed=self.is_obj_placed,
            is_robot_static=self.is_robot_static,
            success=self.is_obj_placed & self.is_robot_static,
            prompt=self.prompt,
            oracle_info=self.oracle_info,
            reward_dict=self.reward_dict,
        )

    def _get_obs_extra(self, info: Dict):
        # Use masks to assign poses
        self.obj_pose = (
            self.mug_left.pose.raw_pose * self.left_mask + 
            self.mug_center.pose.raw_pose * self.center_mask + 
            self.mug_right.pose.raw_pose * self.right_mask
        )

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.obj_pose,
                ball_pose=self.red_ball.pose.raw_pose,
                oracle_info=self.oracle_info,
                # tcp_to_obj_pos=self.obj_to_goal_pos,
            )
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info['success'] *= info['elapsed_steps'] >= 5

        # Reuse pre-calculated distance from evaluate()
        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        
        # Vectorized reward calculation
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_obj_dist)
        static_reward = 1 - torch.tanh(5.0 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))

        reward = (
            reaching_reward + 
            static_reward * info["is_obj_placed"] + 
            info['is_robot_static'] * info["is_obj_placed"]
        )

        reward *= info['elapsed_steps'] >= 5
        reward[info["success"]] = 3

        self.reward_dict = {
            'reaching_reward': reaching_reward,
            'static_reward': static_reward,
            'tcp_to_obj_dist': tcp_to_obj_dist,
            'info["is_obj_placed"]': info["is_obj_placed"],
        }
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3