"""Shell-game shuffle-and-touch tasks for the VLA benchmark."""

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

WARNED_ONCE = False


class ShellGameShuffleTouchVLABaseEnv(BaseEnv):
    """Track one target cup through a shell-game shuffle.

    The robot first observes which cup hides the ball. The cups then swap places
    several times, and the robot must keep track of the target cup through the
    entire motion sequence before making its final selection.

    Episode flow:
    - The target cup is visible before the shuffle starts.
    - Cups swap positions multiple times.
    - The robot touches the cup it believes still hides the ball.

    Success (`success=True`):
    - The robot must touch the final cup position that contains the hidden ball.

    How to customize:
    - `CUE_PHASE_STEPS` changes the observation time before the shuffle begins.
    - `SHUFFLE_PHASE_STEPS` changes the overall duration of the shuffle.
    - `NUM_SWAPS` changes how many swaps the agent must track.
    - `SWAP_ARC_HEIGHT` changes the vertical arc used during swapping.
    - `MIN_DIST` changes spacing between cup slots.
    - `BALL_RADIUS` and `GOAL_THRESH` affect object geometry and touch tolerance.
    """

    LANGUAGE_INSTRUCTION = (
        "Observe which cup hides the ball, track the cups as they shuffle, then touch the correct cup."
    )
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000
    MUG_SCALE = 1.3
    GOAL_THRESH = 0.08
    MUG_DISPLACEMENT_PENALTY_COEF = 0.1
    MUG_DISPLACEMENT_SUCCESS_THRESH = 0.05

    CUE_PHASE_STEPS: List[int] = [1, 5]
    SHUFFLE_PHASE_STEPS: List[int] = [20, 35]
    NUM_SWAPS: List[int] = [2, 4]
    SWAP_ARC_HEIGHT = 0.06

    ACTION_L2_COEF = 0.0
    ACTION_DELTA_L2_COEF = 0.0
    QVEL_L2_COEF = 0.0

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

        self.red_ball = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([255, 0, 0, 255]) / 255,
            name="red_ball",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )

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
            return
        cur = self.cue_steps_per_env.shape[0]
        if n > cur:
            p = n - cur
            def z(*s, **kw):
                return torch.zeros(*s, device=self.device, **kw)
            self.cue_steps_per_env = torch.cat([self.cue_steps_per_env, z(p, dtype=torch.int64)])
            self.empty_steps_per_env = torch.cat([self.empty_steps_per_env, z(p, dtype=torch.int64)])
            self.shuffle_steps_per_env = torch.cat([self.shuffle_steps_per_env, z(p, dtype=torch.int64)])
            self.num_swaps_per_env = torch.cat([self.num_swaps_per_env, z(p, dtype=torch.int64)])
            self.steps_per_swap_per_env = torch.cat([self.steps_per_swap_per_env, z(p, dtype=torch.int64)])
            self.swap_pairs = torch.cat([self.swap_pairs, z(p, self.NUM_SWAPS[1], 2, dtype=torch.long)])
            self.slot_of_mug = torch.cat([self.slot_of_mug, z(p, self.NUM_SWAPS[1] + 1, 3, dtype=torch.long)])
            self.slot_positions = torch.cat([self.slot_positions, z(p, 3, 3)])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            self.cup_with_ball_number = self._batched_episode_rng.choice([0, 1, 2])
            self.cup_with_ball_number = torch.from_numpy(
                self.cup_with_ball_number,
            ).to(device=self.device, dtype=torch.uint8)

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            q = torch.tensor([0, 1, 0.5, 0]).repeat(b, 1)

            left_pos = xyz + torch.tensor([0, -self.MIN_DIST, 0]).repeat(b, 1)
            center_pos = xyz.clone()
            right_pos = xyz + torch.tensor([0, self.MIN_DIST, 0]).repeat(b, 1)

            self.mug_left.set_pose(Pose.create_from_pq(p=left_pos, q=q))
            self.mug_center.set_pose(Pose.create_from_pq(p=center_pos, q=q))
            self.mug_right.set_pose(Pose.create_from_pq(p=right_pos, q=q))

            for buf_name in (
                "_mug_left_manip_ref",
                "_mug_center_manip_ref",
                "_mug_right_manip_ref",
            ):
                if not hasattr(self, buf_name):
                    setattr(
                        self,
                        buf_name,
                        torch.zeros(self.num_envs, 3, device=self.device),
                    )
            if (
                not hasattr(self, "_mug_ref_ready")
                or self._mug_ref_ready is None
                or self._mug_ref_ready.shape[0] != self.num_envs
            ):
                self._mug_ref_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._mug_ref_ready[env_idx] = False

            q_norm = q / q.norm(dim=-1, keepdim=True)
            self.mug_quat = q_norm

            q_ball = [1, 0, 0, 0]
            ball_xyz = xyz.clone()
            offsets = torch.zeros((b, 3), device=xyz.device)
            offsets[:, 1] = torch.where(
                self.cup_with_ball_number == 0,
                -self.MIN_DIST,
                torch.where(
                    self.cup_with_ball_number == 1,
                    0.0,
                    torch.where(self.cup_with_ball_number == 2, self.MIN_DIST, offsets[:, 1]),
                ),
            )
            offsets[:, 2] = self.BALL_RADIUS - self.object_zs[env_idx]
            ball_xyz += offsets
            self.red_ball.set_pose(Pose.create_from_pq(p=ball_xyz, q=q_ball))
            self.ball_initial_pose = ball_xyz

            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(
                    0,
                    self.robot_init_qpos_noise,
                    len(qpos) - 2,
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(
                    0,
                    self.robot_init_qpos_noise,
                    len(qpos) - 2,
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

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

            self.slot_positions[env_idx] = torch.stack(
                [left_pos, center_pos, right_pos],
                dim=1,
            )

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

            final_ball_slot = slot_of_mug_all[
                batch_r,
                num_swaps,
                self.cup_with_ball_number.long(),
            ]
            self.oracle_info = final_ball_slot.to(torch.uint8)

    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        B = elapsed.shape[0]
        bi = torch.arange(B, device=self.device)

        cue_end = self.cue_steps_per_env
        shuffle_end = cue_end + self.shuffle_steps_per_env

        cue_mask = elapsed < cue_end
        shuffle_mask = (elapsed >= cue_end) & (elapsed < shuffle_end)
        manip_mask = elapsed >= shuffle_end
        self.manip_mask = manip_mask

        ball_pose = self.red_ball.pose.raw_pose.clone()

        ball_pose[cue_mask, :3] = self.ball_initial_pose[cue_mask]

        hidden_ball_pos = self.ball_initial_pose.clone()
        hidden_ball_pos[:, 2] += self.HEIGHT_OFFSET
        ball_pose[shuffle_mask, :3] = hidden_ball_pos[shuffle_mask]

        final_ball_slot = self.slot_of_mug[
            bi,
            self.num_swaps_per_env,
            self.cup_with_ball_number.long(),
        ]
        ball_final_pos = self.slot_positions[bi, final_ball_slot].clone()
        ball_final_pos[:, 2] = self.ball_initial_pose[:, 2]
        ball_pose[manip_mask, :3] = ball_final_pos[manip_mask]

        self.red_ball.pose = ball_pose

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

        self.left_mask = (self.cup_with_ball_number == 0).unsqueeze(-1)
        self.center_mask = (self.cup_with_ball_number == 1).unsqueeze(-1)
        self.right_mask = (self.cup_with_ball_number == 2).unsqueeze(-1)

        self.obj_to_goal_pos = (
            (self.mug_left.pose.p - self.agent.tcp.pose.p) * self.left_mask
            + (self.mug_center.pose.p - self.agent.tcp.pose.p) * self.center_mask
            + (self.mug_right.pose.p - self.agent.tcp.pose.p) * self.right_mask
        )

        self.is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH
        self.is_robot_static = self.agent.is_static(0.2)

        just_entered_manip = self.manip_mask & (~self._mug_ref_ready)
        if torch.any(just_entered_manip):
            self._mug_left_manip_ref[just_entered_manip] = self.mug_left.pose.p[just_entered_manip]
            self._mug_center_manip_ref[just_entered_manip] = self.mug_center.pose.p[just_entered_manip]
            self._mug_right_manip_ref[just_entered_manip] = self.mug_right.pose.p[just_entered_manip]
            self._mug_ref_ready[just_entered_manip] = True

        left_displacement = torch.linalg.norm(self.mug_left.pose.p[:, :2] - self._mug_left_manip_ref[:, :2], dim=-1)
        center_displacement = torch.linalg.norm(
            self.mug_center.pose.p[:, :2] - self._mug_center_manip_ref[:, :2], dim=-1
        )
        right_displacement = torch.linalg.norm(self.mug_right.pose.p[:, :2] - self._mug_right_manip_ref[:, :2], dim=-1)
        self.mug_max_displacement = (
            torch.maximum(torch.maximum(left_displacement, center_displacement), right_displacement)
            * self._mug_ref_ready.float()
        )
        self.is_mug_displacement_ok = self.mug_max_displacement <= self.MUG_DISPLACEMENT_SUCCESS_THRESH

        return {
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "is_obj_placed": self.is_obj_placed,
            "is_robot_static": self.is_robot_static,
            "mug_max_displacement": self.mug_max_displacement,
            "is_mug_displacement_ok": self.is_mug_displacement_ok,
            "success": self.is_obj_placed & self.is_robot_static & self.is_mug_displacement_ok,
            "task_cue": self.task_cue,
            "language_instruction": self.LANGUAGE_INSTRUCTION,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        self.obj_pose = (
            self.mug_left.pose.raw_pose * self.left_mask
            + self.mug_center.pose.raw_pose * self.center_mask
            + self.mug_right.pose.raw_pose * self.right_mask
        )
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.obj_pose,
                ball_pose=self.red_ball.pose.raw_pose,
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
            reaching_reward + static_reward * info["is_obj_placed"] + info["is_robot_static"] * info["is_obj_placed"]
        )
        mug_shift_penalty = self.MUG_DISPLACEMENT_PENALTY_COEF * torch.tanh(10.0 * info["mug_max_displacement"])
        reward -= mug_shift_penalty

        reward *= self.manip_mask
        reward[info["success"]] = 3.0

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "static_reward": static_reward,
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "mug_max_displacement": info["mug_max_displacement"],
            "mug_shift_penalty": mug_shift_penalty,
            "is_mug_displacement_ok": info["is_mug_displacement_ok"],
            'info["is_obj_placed"]': info["is_obj_placed"],
        }
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: Dict,
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3.0


@register_env("ShellGameShuffleTouch-VLA-v0", max_episode_steps=60, asset_download_ids=["ycb"])
class ShellGameShuffleTouchVLAEnv(ShellGameShuffleTouchVLABaseEnv):
    CUE_PHASE_STEPS = [1, 5]
    SHUFFLE_PHASE_STEPS = [20, 35]
    NUM_SWAPS = [2, 4]


@register_env("ShellGameShuffleTouch-Long-VLA-v0", max_episode_steps=600, asset_download_ids=["ycb"])
class ShellGameShuffleTouchLongVLAEnv(ShellGameShuffleTouchVLABaseEnv):
    CUE_PHASE_STEPS = [10, 100]
    SHUFFLE_PHASE_STEPS = [100, 400]
    NUM_SWAPS = [5, 15]
