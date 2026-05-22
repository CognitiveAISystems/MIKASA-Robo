"""Shell-game shuffle tasks with lamp color cues for VLA."""

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


class ShellGameShuffleColorLampTouchVLABaseEnv(BaseEnv):
    """Track shuffled cups, then use a color cue to choose the right one.

    Three cups initially hide three differently colored balls. After the cups
    shuffle, a lamp tells the robot which ball color is the target. The robot
    therefore has to solve two subproblems: track the shuffle, then map the
    lamp color to the cup that currently hides the matching ball.

    Episode flow:
    - The initial ball-to-cup mapping is shown.
    - Cups swap positions several times during the shuffle phase.
    - The lamp reveals the target color and the robot selects a cup.

    Success (`success=True`):
    - The robot must touch the cup that hides the ball whose color matches the
      final lamp cue.

    How to customize:
    - `CUE_PHASE_STEPS` changes how long the initial mapping is visible.
    - `SHUFFLE_PHASE_STEPS` changes how much time the swaps take in total.
    - `NUM_SWAPS` changes how many swaps the agent must track.
    - `SWAP_ARC_HEIGHT` changes how high cups lift while moving during swaps.
    - `MIN_DIST` changes the spacing between the three cup slots.
    - `BALL_RADIUS` and `GOAL_THRESH` change object size and touch tolerance.
    """

    LANGUAGE_INSTRUCTION = "Observe which color is under each cup, track the cups as they shuffle, then touch the cup matching the lamp color."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000.0
    MUG_SCALE = 1.3
    GOAL_THRESH = 0.08
    SWAP_ARC_HEIGHT = 0.06

    CUE_PHASE_STEPS: List[int] = [1, 5]
    SHUFFLE_PHASE_STEPS: List[int] = [20, 35]
    NUM_SWAPS: List[int] = [2, 4]

    LAMP_RADIUS = 0.018
    LAMP_BEHIND_OFFSET_X = 0.25
    LAMP_HEIGHT = 0.06

    COLOR_RGBA = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

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
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
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
            name="shuffle_lamp_white",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([255, 255, 255, 255]) / 255.0,
        )
        lamp_red_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="shuffle_lamp_red",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([255, 0, 0, 255]) / 255.0,
        )
        lamp_green_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="shuffle_lamp_green",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.HEIGHT_OFFSET]),
            bulb_off_color=np.array([255, 255, 255, 255]) / 255.0,
            bulb_on_color=np.array([0, 255, 0, 255]) / 255.0,
        )
        lamp_blue_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="shuffle_lamp_blue",
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
            self.empty_steps_per_env = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.shuffle_steps_per_env = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.num_swaps_per_env = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.steps_per_swap_per_env = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.swap_pairs = torch.zeros(n, self.NUM_SWAPS[1], 2, dtype=torch.long, device=self.device)
            self.slot_of_mug = torch.zeros(n, self.NUM_SWAPS[1] + 1, 3, dtype=torch.long, device=self.device)
            self.slot_positions = torch.zeros(n, 3, 3, device=self.device)
            self.target_color = torch.zeros(n, dtype=torch.int64, device=self.device)
            self.lamp_on_pos = torch.zeros(n, 3, device=self.device)
            self.lamp_off_pos = torch.zeros(n, 3, device=self.device)
            return

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

            q_norm = q / q.norm(dim=-1, keepdim=True)
            self.mug_quat = q_norm

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
            shuf_lo, shuf_hi = self.SHUFFLE_PHASE_STEPS
            swap_lo, swap_hi = self.NUM_SWAPS
            cue_steps = torch.randint(cue_lo, cue_hi + 1, (b,), device=self.device, dtype=torch.int64)
            shuffle_steps = torch.randint(shuf_lo, shuf_hi + 1, (b,), device=self.device, dtype=torch.int64)
            num_swaps = torch.randint(swap_lo, swap_hi + 1, (b,), device=self.device, dtype=torch.int64)
            steps_per_swap = shuffle_steps // torch.clamp(num_swaps, min=1)

            self.cue_steps_per_env[env_idx] = cue_steps
            self.shuffle_steps_per_env[env_idx] = shuffle_steps
            self.empty_steps_per_env[env_idx] = shuffle_steps
            self.num_swaps_per_env[env_idx] = num_swaps
            self.steps_per_swap_per_env[env_idx] = steps_per_swap
            self.slot_positions[env_idx] = torch.stack([left_pos, center_pos, right_pos], dim=1)

            ALL_PAIRS = torch.tensor([[0, 1], [0, 2], [1, 2]], device=self.device)
            pair_indices = torch.randint(0, 3, (b, self.NUM_SWAPS[1]), device=self.device)
            local_swap_pairs = ALL_PAIRS[pair_indices]
            self.swap_pairs[env_idx] = local_swap_pairs

            mug_at_slot = torch.arange(3, device=self.device).unsqueeze(0).expand(b, -1).clone()
            arange3 = torch.arange(3, device=self.device).unsqueeze(0).expand(b, -1)
            batch_r = torch.arange(b, device=self.device)
            slot_of_mug_all = torch.zeros(b, self.NUM_SWAPS[1] + 1, 3, dtype=torch.long, device=self.device)
            slot_of_mug_all[:, 0].scatter_(1, mug_at_slot, arange3)
            for s in range(self.NUM_SWAPS[1]):
                active = s < num_swaps
                slot_a = local_swap_pairs[:, s, 0]
                slot_b = local_swap_pairs[:, s, 1]
                mug_a = mug_at_slot[batch_r, slot_a]
                mug_b = mug_at_slot[batch_r, slot_b]
                new_mas = mug_at_slot.clone()
                new_mas[batch_r[active], slot_a[active]] = mug_b[active]
                new_mas[batch_r[active], slot_b[active]] = mug_a[active]
                mug_at_slot = new_mas
                slot_of_mug_all[:, s + 1].scatter_(1, mug_at_slot, arange3)
            self.slot_of_mug[env_idx] = slot_of_mug_all

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

            local_target_color = self.target_color[env_idx]
            final_target_slot = slot_of_mug_all[batch_r, num_swaps, local_target_color]
            self.oracle_info = final_target_slot.to(torch.uint8)
            self.task_cue = local_target_color.to(torch.uint8)

    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)
        b = elapsed.shape[0]
        bi = torch.arange(b, device=self.device)

        cue_end = self.cue_steps_per_env
        shuffle_end = cue_end + self.shuffle_steps_per_env
        cue_mask = elapsed < cue_end
        shuffle_mask = (elapsed >= cue_end) & (elapsed < shuffle_end)
        manip_mask = elapsed >= shuffle_end
        self.manip_mask = manip_mask

        hidden_pos = self.ball_initial_pos[:, 0].clone()
        hidden_pos[:, 2] += self.HEIGHT_OFFSET

        for color_idx, ball in enumerate([self.ball_red, self.ball_green, self.ball_blue]):
            pose = ball.pose.raw_pose.clone()
            pose[cue_mask, :3] = self.ball_initial_pos[cue_mask, color_idx]
            pose[~cue_mask, :3] = hidden_pos[~cue_mask]
            ball.pose = pose

        white_pose = self.lamp_white.pose.raw_pose.clone()
        white_pose[(cue_mask | shuffle_mask), :3] = self.lamp_on_pos[(cue_mask | shuffle_mask)]
        white_pose[manip_mask, :3] = self.lamp_off_pos[manip_mask]
        self.lamp_white.pose = white_pose

        for color_idx, lamp in enumerate([self.lamp_red, self.lamp_green, self.lamp_blue]):
            pose = lamp.pose.raw_pose.clone()
            on_mask = manip_mask & (self.target_color == color_idx)
            pose[on_mask, :3] = self.lamp_on_pos[on_mask]
            pose[~on_mask, :3] = self.lamp_off_pos[~on_mask]
            lamp.pose = pose

        steps_into_shuffle = torch.clamp(elapsed - cue_end, min=0)
        sps = torch.clamp(self.steps_per_swap_per_env, min=1)
        raw_idx = steps_into_shuffle // sps
        past_all = raw_idx >= self.num_swaps_per_env
        cur_swap = torch.where(
            past_all,
            torch.clamp(self.num_swaps_per_env - 1, min=0),
            torch.clamp(raw_idx, max=self.NUM_SWAPS[1] - 1),
        ).long()
        t = (steps_into_shuffle - cur_swap * sps).float() / sps.float()
        t = torch.clamp(t, 0.0, 1.0)
        t[past_all] = 1.0
        progress = t * t * (3.0 - 2.0 * t)
        arc_z = self.SWAP_ARC_HEIGHT * torch.sin(torch.pi * progress)
        arc_z[past_all] = 0.0
        cos_t = torch.cos(torch.pi * progress)
        sin_t = torch.sin(torch.pi * progress)

        mugs = [self.mug_left, self.mug_center, self.mug_right]
        next_swap = torch.clamp(cur_swap + 1, max=self.NUM_SWAPS[1])
        for m, mug in enumerate(mugs):
            before_slot = self.slot_of_mug[bi, cur_swap, m]
            after_slot = self.slot_of_mug[bi, next_swap, m]
            after_slot = torch.where(past_all, before_slot, after_slot)

            start_pos = self.slot_positions[bi, before_slot]
            end_pos = self.slot_positions[bi, after_slot]
            mid = (start_pos + end_pos) * 0.5
            half = (start_pos - end_pos) * 0.5
            perp_x = -half[:, 1]
            perp_y = half[:, 0]

            anim_pos = mid.clone()
            anim_pos[:, 0] += half[:, 0] * cos_t + perp_x * sin_t
            anim_pos[:, 1] += half[:, 1] * cos_t + perp_y * sin_t
            anim_pos[:, 2] = start_pos[:, 2] + arc_z

            final_slot = self.slot_of_mug[bi, self.num_swaps_per_env, m]
            final_pos = self.slot_positions[bi, final_slot]
            cue_pos = self.slot_positions[:, m].clone()
            cue_pos[:, 2] += self.HEIGHT_OFFSET

            new_pose = mug.pose.raw_pose.clone()
            new_pose[cue_mask, :3] = cue_pos[cue_mask]
            new_pose[shuffle_mask, :3] = anim_pos[shuffle_mask]
            new_pose[manip_mask, :3] = final_pos[manip_mask]
            new_pose[:, 3:7] = self.mug_quat
            mug.pose = new_pose

        final_target_slot = self.slot_of_mug[bi, self.num_swaps_per_env, self.target_color.long()]
        target_mug_pos = self.slot_positions[bi, final_target_slot]
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


# ----- Standard tasks -----
@register_env(
    "ShellGameShuffleColorLampTouch-VLA-v0",
    max_episode_steps=60,
    asset_download_ids=["ycb"],
)
class ShellGameShuffleColorLampTouchVLAEnv(ShellGameShuffleColorLampTouchVLABaseEnv):
    CUE_PHASE_STEPS = [1, 5]
    SHUFFLE_PHASE_STEPS = [20, 35]
    NUM_SWAPS = [2, 4]


# ----- Long-horizon tasks -----
@register_env(
    "ShellGameShuffleColorLampTouch-Long-VLA-v0",
    max_episode_steps=600,
    asset_download_ids=["ycb"],
)
class ShellGameShuffleColorLampTouchLongVLAEnv(ShellGameShuffleColorLampTouchVLABaseEnv):
    CUE_PHASE_STEPS = [10, 100]
    SHUFFLE_PHASE_STEPS = [100, 400]
    NUM_SWAPS = [5, 15]
