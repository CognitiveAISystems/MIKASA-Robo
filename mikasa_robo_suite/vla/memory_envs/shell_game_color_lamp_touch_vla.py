"""Shell-game tasks with lamp color cues and no cup shuffling."""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from mani_skill import ASSET_DIR
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from mikasa_robo_suite.vla.utils import shapes

WARNED_ONCE = False


class ShellGameColorLampTouchVLABaseEnv(BaseEnv):
    """Select a fixed cup using a lamp color cue.

    Three cups hide three different colored balls in fixed left, center, and
    right slots. During the cue phase the balls are visible, so the agent can
    observe which color is in which slot. After that the cups cover the balls,
    the lamp turns to one target color, and the agent must touch the cup hiding
    the ball of that color.

    Episode flow:
    - Cue: the three colored balls are visible in fixed cup slots.
    - Manipulation: cups cover the balls and the lamp reveals the target color.

    Success (`success=True`):
    - The robot must touch the cup covering the ball whose color matches the
      lamp, and the robot must be static.

    How to customize:
    - `CUE_PHASE_STEPS` changes how long the color-to-slot mapping is visible.
    - `MIN_DIST` changes spacing between the three cup positions.
    - `BALL_RADIUS` changes the object size under each cup.
    - `GOAL_THRESH` changes how close the TCP must get for the touch to count.
    - `MUG_SCALE` changes the size of the mug assets used as cups.
    """

    LANGUAGE_INSTRUCTION = "Observe which color is under each cup, then touch the cup matching the lamp color."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000.0
    MUG_SCALE = 1.3
    GOAL_THRESH = 0.08

    CUE_PHASE_STEPS: List[int] = [1, 5]

    LAMP_BEHIND_OFFSET_X = 0.25
    LAMP_HEIGHT = 0.06

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
        self.all_model_ids = np.array(list(load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys()))
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
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
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**22,
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
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{id_cup}")
            for record in builder.collision_records:
                if hasattr(record, "scale") and record.scale is not None:
                    record.scale = (np.asarray(record.scale, dtype=np.float32) * self.MUG_SCALE).tolist()
            for record in builder.visual_records:
                if hasattr(record, "scale") and record.scale is not None:
                    record.scale = (np.asarray(record.scale, dtype=np.float32) * self.MUG_SCALE).tolist()
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
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                "There are less parallel environments than total available models to sample. "
                "Not all models will be used during interaction even after resets unless you call "
                "env.reset(options=dict(reconfigure=True)) or set reconfiguration_freq >= 1."
            )

        id_cup = "025_mug"
        self.mug_left, self._objs_1 = self._initialize_mug(model_ids, id_cup, "left")
        self.mug_center, self._objs_2 = self._initialize_mug(model_ids, id_cup, "center")
        self.mug_right, self._objs_3 = self._initialize_mug(model_ids, id_cup, "right")

        self.ball_red = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([255, 0, 0, 255]) / 255.0,
            name="ball_red",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )
        self.ball_green = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([0, 255, 0, 255]) / 255.0,
            name="ball_green",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )
        self.ball_blue = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([0, 0, 255, 255]) / 255.0,
            name="ball_blue",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )

        lamp_white_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="color_lamp_white",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([255, 255, 255, 255]) / 255.0,
        )
        lamp_red_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="color_lamp_red",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([255, 0, 0, 255]) / 255.0,
        )
        lamp_green_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="color_lamp_green",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([0, 255, 0, 255]) / 255.0,
        )
        lamp_blue_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="color_lamp_blue",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([0, 0, 255, 255]) / 255.0,
        )
        self.lamp_body = lamp_white_parts["body"]
        self.lamp_white = lamp_white_parts["bulb_off"]
        self.lamp_red = lamp_red_parts["bulb_on"]
        self.lamp_green = lamp_green_parts["bulb_on"]
        self.lamp_blue = lamp_blue_parts["bulb_on"]
        self._lamp_aux_bodies = [
            lamp_red_parts["body"],
            lamp_green_parts["body"],
            lamp_blue_parts["body"],
            lamp_red_parts["bulb_off"],
            lamp_green_parts["bulb_off"],
            lamp_blue_parts["bulb_off"],
        ]

    def _after_reconfigure(self, options: dict):
        num_objects = len(self._objs_1) + len(self._objs_2) + len(self._objs_3)
        self.object_zs = torch.empty(num_objects, device=self.device)
        for idx, obj in enumerate((*self._objs_1, *self._objs_2, *self._objs_3)):
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs[idx] = -collision_mesh.bounding_box.bounds[0, 2]

    def _ensure_phase_buffers(self, env_idx: torch.Tensor):
        n = int(env_idx.max().item()) + 1
        if not hasattr(self, "cue_steps_per_env") or self.cue_steps_per_env is None:
            self.cue_steps_per_env = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.target_color = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.slot_positions = torch.zeros(n, 3, 3, device=self.device)
            self.lamp_on_pos = torch.zeros(n, 3, device=self.device)
            self.lamp_off_pos = torch.zeros(n, 3, device=self.device)
            return
        cur = self.cue_steps_per_env.shape[0]
        if n > cur:
            p = n - cur
            def z(*s, **kw):
                return torch.zeros(*s, device=self.device, **kw)
            self.cue_steps_per_env = torch.cat([self.cue_steps_per_env, z(p, dtype=torch.int64)])
            self.target_color = torch.cat([self.target_color, z(p, dtype=torch.int64)])
            self.slot_positions = torch.cat([self.slot_positions, z(p, 3, 3)])
            self.lamp_on_pos = torch.cat([self.lamp_on_pos, z(p, 3)])
            self.lamp_off_pos = torch.cat([self.lamp_off_pos, z(p, 3)])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.task_cue = None
            self.reward_dict = None

            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, :2] = torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            q = torch.tensor([0, 1, 0.5, 0], device=self.device).repeat(b, 1)

            left_pos = xyz + torch.tensor([0, -self.MIN_DIST, 0], device=self.device).repeat(b, 1)
            center_pos = xyz.clone()
            right_pos = xyz + torch.tensor([0, self.MIN_DIST, 0], device=self.device).repeat(b, 1)

            self.mug_left.set_pose(Pose.create_from_pq(p=left_pos, q=q))
            self.mug_center.set_pose(Pose.create_from_pq(p=center_pos, q=q))
            self.mug_right.set_pose(Pose.create_from_pq(p=right_pos, q=q))
            self.mug_quat = q / q.norm(dim=-1, keepdim=True)

            z_offset = self.BALL_RADIUS - self.object_zs[env_idx]
            red_xyz = left_pos.clone()
            green_xyz = center_pos.clone()
            blue_xyz = right_pos.clone()
            red_xyz[:, 2] += z_offset
            green_xyz[:, 2] += z_offset
            blue_xyz[:, 2] += z_offset

            q_ball = [1, 0, 0, 0]
            self.ball_red.set_pose(Pose.create_from_pq(p=red_xyz, q=q_ball))
            self.ball_green.set_pose(Pose.create_from_pq(p=green_xyz, q=q_ball))
            self.ball_blue.set_pose(Pose.create_from_pq(p=blue_xyz, q=q_ball))
            self.ball_initial_pos = torch.stack([red_xyz, green_xyz, blue_xyz], dim=1)

            self._ensure_phase_buffers(env_idx)
            cue_lo, cue_hi = self.CUE_PHASE_STEPS
            cue_steps = torch.randint(cue_lo, cue_hi + 1, (b,), device=self.device, dtype=torch.int64)
            self.cue_steps_per_env[env_idx] = cue_steps
            self.slot_positions[env_idx] = torch.stack([left_pos, center_pos, right_pos], dim=1)

            self.target_color[env_idx] = torch.randint(0, 3, (b,), device=self.device, dtype=torch.int64)

            lamp_pos = center_pos.clone()
            lamp_pos[:, 0] += self.LAMP_BEHIND_OFFSET_X
            lamp_pos[:, 2] = self.LAMP_HEIGHT
            lamp_off = lamp_pos.clone()
            lamp_off[:, 2] += self.HEIGHT_OFFSET
            self.lamp_on_pos[env_idx] = lamp_pos
            self.lamp_off_pos[env_idx] = lamp_off
            lamp_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            self.lamp_body.set_pose(Pose.create_from_pq(p=lamp_pos, q=lamp_q))
            for aux in self._lamp_aux_bodies:
                aux.set_pose(Pose.create_from_pq(p=lamp_off, q=lamp_q))

            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

            self.oracle_info = self.target_color[env_idx].to(torch.uint8)
            self.task_cue = self.target_color[env_idx].to(torch.uint8)

    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)
        b = elapsed.shape[0]
        cue_mask = elapsed < self.cue_steps_per_env
        manip_mask = ~cue_mask
        self.manip_mask = manip_mask

        hidden_pos = self.ball_initial_pos[:, 0].clone()
        hidden_pos[:, 2] += self.HEIGHT_OFFSET

        for color_idx, ball in enumerate([self.ball_red, self.ball_green, self.ball_blue]):
            pose = ball.pose.raw_pose.clone()
            pose[cue_mask, :3] = self.ball_initial_pos[cue_mask, color_idx]
            pose[~cue_mask, :3] = hidden_pos[~cue_mask]
            ball.pose = pose

        white_pose = self.lamp_white.pose.raw_pose.clone()
        white_pose[cue_mask, :3] = self.lamp_on_pos[cue_mask]
        white_pose[manip_mask, :3] = self.lamp_off_pos[manip_mask]
        self.lamp_white.pose = white_pose

        for color_idx, lamp in enumerate([self.lamp_red, self.lamp_green, self.lamp_blue]):
            pose = lamp.pose.raw_pose.clone()
            on_mask = manip_mask & (self.target_color == color_idx)
            pose[on_mask, :3] = self.lamp_on_pos[on_mask]
            pose[~on_mask, :3] = self.lamp_off_pos[~on_mask]
            lamp.pose = pose

        mugs = [self.mug_left, self.mug_center, self.mug_right]
        for m, mug in enumerate(mugs):
            final_pos = self.slot_positions[:, m]
            cue_pos = final_pos.clone()
            cue_pos[:, 2] += self.HEIGHT_OFFSET
            new_pose = mug.pose.raw_pose.clone()
            new_pose[cue_mask, :3] = cue_pos[cue_mask]
            new_pose[manip_mask, :3] = final_pos[manip_mask]
            new_pose[:, 3:7] = self.mug_quat
            mug.pose = new_pose

        target_mug_pos = self.slot_positions[torch.arange(b, device=self.device), self.target_color.long()]
        self.obj_to_goal_pos = target_mug_pos - self.agent.tcp.pose.p

        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        self.is_obj_placed = tcp_to_obj_dist <= self.GOAL_THRESH
        self.is_robot_static = self.agent.is_static(0.2)
        success = self.is_obj_placed & self.is_robot_static & manip_mask

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": self.is_obj_placed,
            "is_robot_static": self.is_robot_static,
            "success": success,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_to_goal_pos=self.obj_to_goal_pos,
                target_color=self.target_color.to(torch.uint8),
                oracle_info=self.oracle_info,
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

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info["success"] *= self.manip_mask
        tcp_to_obj_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5.0 * tcp_to_obj_dist)
        static_reward = 1 - torch.tanh(5.0 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))
        reward = (
            reaching_reward + static_reward * info["is_obj_placed"] + (info["is_robot_static"] * info["is_obj_placed"])
        )
        reward *= self.manip_mask
        reward[info["success"]] = 3.0

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "static_reward": static_reward,
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "is_obj_placed": info["is_obj_placed"],
        }
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0


@register_env(
    "ShellGameColorLampTouch-VLA-v0",
    max_episode_steps=30,
    asset_download_ids=["ycb"],
)
class ShellGameColorLampTouchVLAEnv(ShellGameColorLampTouchVLABaseEnv):
    CUE_PHASE_STEPS = [1, 5]
