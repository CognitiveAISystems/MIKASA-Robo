"""Shell-game push tasks for the VLA memory benchmark."""

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


class ShellGamePushVLABaseEnv(BaseEnv):
    """Remember where the hidden ball is and recover it by pushing.

    The cue reveals which cup position contains the ball. Once the cups are in
    place, the robot must use memory to move to the correct location and push
    the hidden ball into a nearby goal region. The task is about spatial memory
    plus directional contact, not grasping.

    Episode flow:
    - The ball is visible during the cue phase.
    - Cups cover the candidate locations.
    - The robot pushes the ball from the correct hidden location toward the goal.

    Success (`success=True`):
    - The ball center must finish inside the goal region.

    How to customize:
    - `CUE_PHASE_STEPS` changes how long the agent can observe the target location.
    - `MIN_DIST` changes spacing between the three cup positions.
    - `BALL_RADIUS` changes contact geometry and pushing difficulty.
    - `GOAL_THRESH` changes how forgiving the final goal region is.
    - `MUG_SCALE` changes the size of the cup assets used as shells.
    """

    LANGUAGE_INSTRUCTION = "Observe which cup hides the ball, wait, then push that cup forward."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000
    CUE_PHASE_STEPS = [1, 5]
    MUG_SCALE = 1.3
    GOAL_THRESH = 0.06  # +20 % vs. the original 0.05 — bigger goal region

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
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**21, max_rigid_contact_count=2**22
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
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        id_cup = "025_mug"
        self.mug_left, self._objs_1 = self._initialize_mug(model_ids, id_cup, "left")
        self.mug_center, self._objs_2 = self._initialize_mug(model_ids, id_cup, "center")
        self.mug_right, self._objs_3 = self._initialize_mug(model_ids, id_cup, "right")

        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.GOAL_THRESH,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.GOAL_THRESH]),
        )
        self._hidden_objects.append(self.goal_site)

        # The red ball is a purely visual cue: it tells the human/oracle
        # which cup hides the target during the cue phase, but the policy
        # never grasps or manipulates it (success in this task is measured
        # by where the cup ends up, not the ball). Making it `dynamic` with
        # collision enabled caused two artefacts: (a) when cups descend at
        # the cue→manip transition the physics solver ejected the ball
        # upward onto a cup lid, and (b) at the very first frame the
        # collision-mesh sync lagged the visual sync, making the cup look
        # sunken into the table. `body_type="kinematic" + add_collision=False`
        # mirrors the existing `goal_site` config — pose is authoritative,
        # physics can't move it.
        self.red_ball = actors.build_sphere(
            self.scene,
            radius=self.BALL_RADIUS,
            color=np.array([255, 0, 0, 255]) / 255,
            name="red_ball",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, self.BALL_RADIUS]),
        )

    def _after_reconfigure(self, options: dict):
        num_objects = len(self._objs_1) + len(self._objs_2) + len(self._objs_3)
        self.object_zs = torch.empty(num_objects, device=self.device)

        for idx, obj in enumerate((*self._objs_1, *self._objs_2, *self._objs_3)):
            collision_mesh = obj.get_first_collision_mesh()
            self.object_zs[idx] = -collision_mesh.bounding_box.bounds[0, 2]

    def _ensure_phase_buffers(self, env_idx: torch.Tensor):
        target_size = int(env_idx.max().item()) + 1
        if not hasattr(self, "cue_steps_per_env") or self.cue_steps_per_env is None:
            self.cue_steps_per_env = torch.zeros(target_size, dtype=torch.int64, device=self.device)
            return
        if target_size > self.cue_steps_per_env.shape[0]:
            pad = target_size - self.cue_steps_per_env.shape[0]
            self.cue_steps_per_env = torch.cat(
                [
                    self.cue_steps_per_env,
                    torch.zeros(pad, dtype=torch.int64, device=self.device),
                ]
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.task_cue = None
            self.reward_dict = None

            self.cup_with_ball_number = self._batched_episode_rng.choice([0, 1, 2])
            self.cup_with_ball_number = torch.from_numpy(self.cup_with_ball_number).to(
                device=self.device, dtype=torch.uint8
            )

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            q = torch.tensor([0, 1, 0.5, 0]).repeat(b, 1)

            self.mug_left.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, -self.MIN_DIST, 0]).repeat(b, 1), q=q))
            self.mug_center.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, 0, 0]), q=q))
            self.mug_right.set_pose(Pose.create_from_pq(p=xyz + torch.tensor([0, self.MIN_DIST, 0]).repeat(b, 1), q=q))

            q = [1, 0, 0, 0]
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
            red_ball_pose = Pose.create_from_pq(p=ball_xyz, q=q)
            self.red_ball.set_pose(red_ball_pose)
            self.ball_initial_pose = ball_xyz

            goal_xyz = ball_xyz.clone() + torch.tensor([0.1, 0, self.GOAL_THRESH / 2])
            goal_q = torch.tensor([0.707, 0.707, 0, 0]).repeat(b, 1)
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_xyz, q=goal_q))

            self.oracle_info = self.cup_with_ball_number

            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            elif self.robot_uids == "xmate3_robotiq":
                qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.562, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

            self._ensure_phase_buffers(env_idx)
            cue_lo, cue_hi = self.CUE_PHASE_STEPS
            self.cue_steps_per_env[env_idx] = torch.randint(
                low=cue_lo,
                high=cue_hi + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )

    def evaluate(self):

        self.original_poses = {
            "mug_left": self.mug_left.pose.raw_pose.clone(),
            "mug_center": self.mug_center.pose.raw_pose.clone(),
            "mug_right": self.mug_right.pose.raw_pose.clone(),
            "ball": self.red_ball.pose.raw_pose.clone(),
        }

        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)
        hide_mask = elapsed < self.cue_steps_per_env
        self.manip_mask = ~hide_mask

        for mug, orig_pose in zip(
            [self.mug_left, self.mug_center, self.mug_right],
            [self.original_poses["mug_left"], self.original_poses["mug_center"], self.original_poses["mug_right"]],
        ):
            new_pose = orig_pose.clone()
            new_pose[hide_mask & (new_pose[..., 2] < 100), 2] += self.HEIGHT_OFFSET
            new_pose[~hide_mask & (new_pose[..., 2] > 100), 2] -= self.HEIGHT_OFFSET
            mug.pose = new_pose

        self.left_mask = (self.cup_with_ball_number == 0).unsqueeze(-1)
        self.center_mask = (self.cup_with_ball_number == 1).unsqueeze(-1)
        self.right_mask = (self.cup_with_ball_number == 2).unsqueeze(-1)

        # Ball-follows-cup. The kinematic red_ball doesn't respond to
        # physics, so we drive its xy from the hiding cup explicitly:
        # while the cup is at table height the ball tracks it (so when the
        # robot pushes the cup, the ball is carried along, exactly as if
        # tucked underneath); when the cup is clearly off the table (cue
        # phase lift, or gripper pick-up in the Pick variant) we leave the
        # ball where it last was. Ball.z is pinned to BALL_RADIUS so it
        # always sits on the table surface.
        cup_with_ball_p = (
            self.mug_left.pose.p * self.left_mask
            + self.mug_center.pose.p * self.center_mask
            + self.mug_right.pose.p * self.right_mask
        )  # (B, 3) — pose of the cup hiding the ball, post-step
        # All three cups share the same model, so object_zs[:num_envs]
        # gives the cup's natural resting z per env.
        cup_resting_z = self.object_zs[: self.num_envs]
        cup_at_table = cup_with_ball_p[..., 2] <= cup_resting_z + 0.05  # (B,)
        if cup_at_table.any():
            ball_q = self.original_poses["ball"][..., 3:]
            new_ball_p = self.original_poses["ball"][..., :3].clone()
            new_ball_p[cup_at_table, 0] = cup_with_ball_p[cup_at_table, 0]
            new_ball_p[cup_at_table, 1] = cup_with_ball_p[cup_at_table, 1]
            new_ball_p[cup_at_table, 2] = self.BALL_RADIUS
            self.red_ball.set_pose(Pose.create_from_pq(p=new_ball_p, q=ball_q))

        self.obj_to_goal_pos = torch.zeros_like(
            self.mug_left.pose.p, device=self.mug_left.pose.p.device, dtype=self.mug_left.pose.p.dtype
        )

        self.obj_to_goal_pos = (
            (self.goal_site.pose.p - self.mug_left.pose.p) * self.left_mask
            + (self.goal_site.pose.p - self.mug_center.pose.p) * self.center_mask
            + (self.goal_site.pose.p - self.mug_right.pose.p) * self.right_mask
        )

        self.is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH * 1.6
        self.is_robot_static = self.agent.is_static(0.3)

        return dict(
            obj_to_goal_pos=self.obj_to_goal_pos,
            is_obj_placed=self.is_obj_placed,
            is_robot_static=self.is_robot_static,
            success=self.is_obj_placed & self.is_robot_static,
            task_cue=self.task_cue,
            language_instruction=self.LANGUAGE_INSTRUCTION,
            oracle_info=self.oracle_info,
            reward_dict=self.reward_dict,
        )

    def _get_obs_extra(self, info: Dict):
        self.obj_pose = (
            self.mug_left.pose.raw_pose * self.left_mask
            + self.mug_center.pose.raw_pose * self.center_mask
            + self.mug_right.pose.raw_pose * self.right_mask
        )

        self.tcp_to_obj_pos = torch.zeros_like(
            self.mug_left.pose.p, device=self.mug_left.pose.p.device, dtype=self.mug_left.pose.p.dtype
        )
        self.tcp_to_obj_pos = (
            (self.mug_left.pose.p - self.agent.tcp.pose.p) * self.left_mask
            + (self.mug_center.pose.p - self.agent.tcp.pose.p) * self.center_mask
            + (self.mug_right.pose.p - self.agent.tcp.pose.p) * self.right_mask
        )

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.goal_site.pose.p,
                obj_pose=self.obj_pose,
                ball_pose=self.red_ball.pose.raw_pose,
                oracle_info=self.oracle_info,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info["success"] *= self.manip_mask

        tcp_to_obj_dist = torch.linalg.norm(self.tcp_to_obj_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10 * tcp_to_obj_dist)

        obj_to_goal_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)

        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))

        reached_status = tcp_to_obj_dist <= 0.08

        reward = (
            reaching_reward
            + 2 * place_reward * reached_status
            + 0.2 * static_reward * info["is_obj_placed"]
            + 0.2 * info["is_robot_static"] * info["is_obj_placed"]
        )

        reward[info["success"]] = 4.0
        reward *= self.manip_mask

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "static_reward": static_reward,
            "is_robot_static": self.is_robot_static,
            "place_reward": place_reward,
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "obj_to_goal_dist": obj_to_goal_dist,
            'info["is_obj_placed"]': info["is_obj_placed"],
            "reached_status": reached_status,
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0


@register_env("ShellGamePush-VLA-v0", max_episode_steps=30, asset_download_ids=["ycb"])
class ShellGamePushVLAEnv(ShellGamePushVLABaseEnv):
    CUE_PHASE_STEPS = [1, 5]
