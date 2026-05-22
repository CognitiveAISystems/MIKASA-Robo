"""Shell-game pick tasks for the VLA memory benchmark."""

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


class ShellGamePickVLABaseEnv(BaseEnv):
    """Remember where the hidden ball is and retrieve it by grasping.

    The task begins by showing which cup position contains the ball. Then the
    cups cover the scene and the robot must act from memory: it has to recover
    the ball from the correct location and place it onto the goal marker.

    Episode flow:
    - The ball is visible during the cue phase.
    - Cups are positioned over the possible ball locations.
    - The robot picks the ball from the correct location and lifts it to the goal.

    Success (`success=True`):
    - The ball must be placed inside the goal region above the correct cup site.

    How to customize:
    - `CUE_PHASE_STEPS` changes how long the ball is visible before action starts.
    - `MIN_DIST` changes spacing between the cup positions.
    - `BALL_RADIUS` changes the size of the ball to be grasped.
    - `GOAL_THRESH` changes the size of the placement goal region.
    - `MUG_SCALE` changes the size of the mug assets used as cups.
    """

    LANGUAGE_INSTRUCTION = "Observe which cup hides the ball, wait, then pick up that cup and lift it."
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    BALL_RADIUS = 0.02
    MIN_DIST = 0.2
    HEIGHT_OFFSET = 1000
    CUE_PHASE_STEPS = [1, 5]
    MUG_SCALE = 1.3
    GOAL_THRESH = 0.05
    # World-frame offset from mug center to the handle grasp point.
    # With fixed q=[0,1,2.5,0] the handle faces [-0.724, +0.689, 0] in world frame.
    # 0.048 * MUG_SCALE ≈ 0.062m from center to handle body.
    HANDLE_OFFSET_X = 0.045  # toward robot (-X)
    HANDLE_OFFSET_Y = 0.043  # sideways (+Y)

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

        # See shell_game_push_vla.py for rationale. The red ball is a
        # purely visual cue (which cup hides the target). Making it
        # `dynamic`+collidable caused the cup-descent physics to eject
        # the ball onto a cup lid and made the very first frame look
        # sunken (collision-mesh sync lagging visual sync). Kinematic +
        # no collision matches the existing `goal_site` setup.
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

            # Fixed orientation for all mugs (no randomization).
            # q=[0, 1, 2.5, 0] normalized → mug upside-down, handle faces [-0.724, +0.689, 0].
            q = torch.zeros((b, 4), device=self.device)
            q[:, 1] = 1.0
            q[:, 2] = 2.5
            q = q / torch.norm(q, dim=1, keepdim=True)

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

            goal_xyz = ball_xyz.clone() + torch.tensor([0, 0, 0.1 + self.GOAL_THRESH / 2])
            goal_q = torch.tensor([0.707, 0, 0.707, 0]).repeat(b, 1)
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

            self._mug_render_components = {}
            for mug_name, mug in [("left", self.mug_left), ("center", self.mug_center), ("right", self.mug_right)]:
                self._mug_render_components[mug_name] = [
                    obj.find_component_by_type(sapien.render.RenderBodyComponent) for obj in mug._objs
                ]

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

        # Ball-follows-cup. See shell_game_push_vla.py for full rationale.
        # In the Pick variant the policy lifts the cup with the gripper:
        # the `cup_at_table` gate ensures we stop tracking once the cup
        # rises clearly above the table, leaving the ball exposed on the
        # table at the spot where the cup was sitting (which is the
        # natural "reveal" for this task).
        cup_with_ball_p = (
            self.mug_left.pose.p * self.left_mask
            + self.mug_center.pose.p * self.center_mask
            + self.mug_right.pose.p * self.right_mask
        )  # (B, 3)
        cup_resting_z = self.object_zs[: self.num_envs]
        cup_at_table = cup_with_ball_p[..., 2] <= cup_resting_z + 0.05  # (B,)
        if cup_at_table.any():
            ball_q = self.original_poses["ball"][..., 3:]
            new_ball_p = self.original_poses["ball"][..., :3].clone()
            new_ball_p[cup_at_table, 0] = cup_with_ball_p[cup_at_table, 0]
            new_ball_p[cup_at_table, 1] = cup_with_ball_p[cup_at_table, 1]
            new_ball_p[cup_at_table, 2] = self.BALL_RADIUS
            self.red_ball.set_pose(Pose.create_from_pq(p=new_ball_p, q=ball_q))
        self.obj_to_goal_pos = (
            (self.goal_site.pose.p - self.mug_left.pose.p) * self.left_mask
            + (self.goal_site.pose.p - self.mug_center.pose.p) * self.center_mask
            + (self.goal_site.pose.p - self.mug_right.pose.p) * self.right_mask
        )

        self.is_obj_placed = torch.linalg.norm(self.obj_to_goal_pos, axis=1) <= self.GOAL_THRESH * 1.6

        self.is_grasped = (
            self.agent.is_grasping(self.mug_left) * self.left_mask.squeeze(-1)
            + self.agent.is_grasping(self.mug_center) * self.center_mask.squeeze(-1)
            + self.agent.is_grasping(self.mug_right) * self.right_mask.squeeze(-1)
        )

        self.is_robot_static = self.agent.is_static(0.2)

        return dict(
            is_grasped=self.is_grasped,
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

        self.tcp_to_obj_pos = (
            (self.mug_left.pose.p - self.agent.tcp.pose.p) * self.left_mask
            + (self.mug_center.pose.p - self.agent.tcp.pose.p) * self.center_mask
            + (self.mug_right.pose.p - self.agent.tcp.pose.p) * self.right_mask
        )

        # Handle position: mug center shifted toward the handle grasp point.
        # The mug z (= object_zs) is already at half mug height, matching the handle height.
        obj_pos = self.obj_pose[:, :3]  # [batch, 3]
        handle_offset = torch.zeros_like(obj_pos)
        handle_offset[:, 0] = -self.HANDLE_OFFSET_X
        handle_offset[:, 1] = self.HANDLE_OFFSET_Y
        self.handle_pos = obj_pos + handle_offset

        self.tcp_to_handle_pos = self.handle_pos - self.agent.tcp.pose.p

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.goal_site.pose.p,
                obj_pose=self.obj_pose,
                handle_pos=self.handle_pos,
                ball_pose=self.red_ball.pose.raw_pose,
                oracle_info=self.oracle_info,
                is_grasped=self.is_grasped,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        info["success"] *= self.manip_mask

        # Phase 1: reach the handle.
        tcp_to_handle_dist = torch.linalg.norm(self.tcp_to_handle_pos, axis=1)
        reaching_reward = 1 - torch.tanh(10 * tcp_to_handle_dist)

        obj_to_goal_dist = torch.linalg.norm(self.obj_to_goal_pos, axis=1)
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)

        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1))

        # TCP is within grasping range of the handle.
        reached_status = (tcp_to_handle_dist <= 0.04).float()

        is_grasped = info["is_grasped"].float()

        # Phase 1.5: while approaching (not yet at handle), reward opening the gripper so
        # it can wrap around the handle when it arrives.
        gripper_width = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1)  # 0=closed, 0.08=open
        open_gripper_reward = (gripper_width / 0.08) * (1.0 - reached_status)

        # Phase 2: grasp the handle — gives clear signal to close the gripper once nearby.
        grasp_reward = is_grasped * reached_status

        # Phase 3: lift and place — conditioned on actually holding the mug.
        reward = (
            3.0 * reaching_reward
            + 1.5 * open_gripper_reward
            + 2.0 * reached_status
            + 10.0 * grasp_reward
            + 30.0 * place_reward * is_grasped
            + 5.0 * static_reward * info["is_obj_placed"] * is_grasped
            + 5.0 * info["is_robot_static"] * info["is_obj_placed"] * is_grasped
        )

        reward[info["success"]] = 75.0
        reward *= self.manip_mask

        tcp_to_obj_dist = torch.linalg.norm(self.tcp_to_obj_pos, axis=1)
        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "open_gripper_reward": open_gripper_reward,
            "grasp_reward": grasp_reward,
            "reached_status": reached_status,
            'info["is_grasped"]': info["is_grasped"],
            "place_reward": place_reward,
            "static_reward": static_reward,
            "tcp_to_handle_dist": tcp_to_handle_dist,
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "obj_to_goal_dist": obj_to_goal_dist,
            'info["is_obj_placed"]': info["is_obj_placed"],
            "gripper_width": gripper_width,
            "dx_handle": self.tcp_to_handle_pos[:, 0],
            "dy_handle": self.tcp_to_handle_pos[:, 1],
            "dz_handle": self.tcp_to_handle_pos[:, 2],
        }

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 75.0


@register_env("ShellGamePick-VLA-v0", max_episode_steps=30, asset_download_ids=["ycb"])
class ShellGamePickVLAEnv(ShellGamePickVLABaseEnv):
    CUE_PHASE_STEPS = [1, 5]
