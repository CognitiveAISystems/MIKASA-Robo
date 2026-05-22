"""Timed-transfer tasks for the VLA memory benchmark.

The agent must wait a precise number of steps after a visual signal,
then transfer a cube from one disc to another.
"""

from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat

from mikasa_robo_suite.vla.utils import shapes


class TimedTransferVLABaseEnv(BaseEnv):
    """Wait a precise number of steps after a signal, then move a cube.

    Scene: two flat discs (green and red) on the table, a blue cube resting
    on the green disc, and a white lamp.

    Episode flow:
    1. The lamp is off.  The blue cube sits on the green disc.
    2. After a brief random delay the lamp turns green (the signal).
    3. The agent must internally count DELAY_STEPS from the signal.
    4. The agent picks the blue cube and places it on the red disc.
    5. The cube must be on the red disc within +/-TOLERANCE_FRAC of the target
       step.  Placing too early or too late is a failure.

    Customize difficulty by changing DELAY_STEPS.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, PandaWristCam]

    HEIGHT_OFFSET = 1000.0
    DELAY_STEPS: int = 100
    TOLERANCE_FRAC: float = 0.05
    PRE_SIGNAL_STEPS: List[int] = [1, 3]

    DISC_RADIUS = 0.07
    DISC_HALF_HEIGHT = 0.003
    CUBE_HALF_SIZE = 0.02
    GOAL_THRESH = 0.05

    LAMP_BASE_RADIUS = 0.018
    LAMP_BASE_HALF_HEIGHT = 0.008
    LAMP_STEM_RADIUS = 0.004
    LAMP_STEM_HALF_HEIGHT = 0.020
    LAMP_BULB_RADIUS = 0.012
    LAMP_FORWARD_OFFSET = 0.22

    DISC_SEPARATION = 0.22

    ACTION_L2_COEF = 0.01
    ACTION_DELTA_L2_COEF = 0.03
    QVEL_L2_COEF = 0.01
    SUCCESS_BONUS = 30.0
    FAILURE_PENALTY = 25.0

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._lang_instruction = (
            "When the white lamp turns green, start counting steps from that exact moment. "
            f"Move the blue cube from the green disc to the red disc exactly on step {self.DELAY_STEPS} of that count."
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
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

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

        default_initial_pose = sapien.Pose(p=[0.0, 0.0, self.HEIGHT_OFFSET])

        # --- Green disc (kinematic) ---
        green_disc_builder = self.scene.create_actor_builder()
        green_disc_builder.add_cylinder_collision(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
        )
        green_disc_builder.add_cylinder_visual(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(
                base_color=np.array([30, 180, 30, 255], dtype=np.float32) / 255.0,
            ),
        )
        self.green_disc = _build_by_type(
            green_disc_builder,
            name="green_disc",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        # --- Red disc (kinematic) ---
        red_disc_builder = self.scene.create_actor_builder()
        red_disc_builder.add_cylinder_collision(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
        )
        red_disc_builder.add_cylinder_visual(
            radius=self.DISC_RADIUS,
            half_length=self.DISC_HALF_HEIGHT,
            material=sapien.render.RenderMaterial(
                base_color=np.array([200, 30, 30, 255], dtype=np.float32) / 255.0,
            ),
        )
        self.red_disc = _build_by_type(
            red_disc_builder,
            name="red_disc",
            body_type="kinematic",
            initial_pose=default_initial_pose,
        )

        # --- Blue cube (dynamic) ---
        cube_hs = np.array([self.CUBE_HALF_SIZE] * 3, dtype=np.float32)
        cube_builder = self.scene.create_actor_builder()
        cube_builder.add_box_collision(half_size=cube_hs)
        cube_builder.add_box_visual(
            half_size=cube_hs,
            material=sapien.render.RenderMaterial(
                base_color=np.array([30, 30, 220, 255], dtype=np.float32) / 255.0,
            ),
        )
        self.blue_cube = _build_by_type(
            cube_builder,
            name="blue_cube",
            body_type="dynamic",
            initial_pose=default_initial_pose,
        )

        # --- White signal lamp (off -> green) ---
        lamp_parts = shapes.build_color_switch_lamp(
            scene=self.scene,
            name="signal_lamp",
            body_type="kinematic",
            add_collision=False,
            initial_pose=default_initial_pose,
            base_radius=self.LAMP_BASE_RADIUS,
            base_half_height=self.LAMP_BASE_HALF_HEIGHT,
            stem_radius=self.LAMP_STEM_RADIUS,
            stem_half_height=self.LAMP_STEM_HALF_HEIGHT,
            bulb_radius=self.LAMP_BULB_RADIUS,
            bulb_off_color=np.array([220, 220, 220, 255], dtype=np.float32) / 255.0,
            bulb_on_color=np.array([0, 255, 0, 255], dtype=np.float32) / 255.0,
        )
        self.lamp_body = lamp_parts["body"]
        self.lamp_bulb_off = lamp_parts["bulb_off"]
        self.lamp_bulb_on = lamp_parts["bulb_on"]
        shapes._set_actor_visual_rgba(
            self.lamp_bulb_on,
            np.array([0, 255, 0, 255], dtype=np.float32) / 255.0,
            emission_scale=3.0,
            remove_textures=True,
        )

        # --- Per-env state tensors ---
        n = self.num_envs
        d = self.device
        self.signal_step = torch.zeros(n, dtype=torch.int64, device=d)
        self.window_start = torch.zeros(n, dtype=torch.int64, device=d)
        self.window_end = torch.zeros(n, dtype=torch.int64, device=d)
        self.green_disc_center = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.red_disc_center = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.lamp_on_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.lamp_off_pos = torch.zeros((n, 3), dtype=torch.float32, device=d)
        self.too_early = torch.zeros(n, dtype=torch.bool, device=d)
        self.success_flag = torch.zeros(n, dtype=torch.bool, device=d)
        self.failed = torch.zeros(n, dtype=torch.bool, device=d)
        self.disc_quat = torch.tensor(
            euler2quat(0, np.pi / 2, 0),
            dtype=torch.float32,
            device=d,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            env_idx = env_idx.to(self.device)

            self.task_cue = None
            self.reward_dict = None

            # Random center for the disc pair
            center_xy = torch.zeros((b, 2), device=self.device)
            center_xy[:, 0] = torch.rand(b, device=self.device) * 0.10 - 0.15
            center_xy[:, 1] = (torch.rand(b, device=self.device) - 0.5) * 0.10

            # Green disc
            green_xyz = torch.zeros((b, 3), device=self.device)
            green_xyz[:, 0] = center_xy[:, 0]
            green_xyz[:, 1] = center_xy[:, 1] - self.DISC_SEPARATION / 2
            green_xyz[:, 2] = self.DISC_HALF_HEIGHT

            # Red disc
            red_xyz = torch.zeros((b, 3), device=self.device)
            red_xyz[:, 0] = center_xy[:, 0]
            red_xyz[:, 1] = center_xy[:, 1] + self.DISC_SEPARATION / 2
            red_xyz[:, 2] = self.DISC_HALF_HEIGHT

            disc_q = self.disc_quat.unsqueeze(0).repeat(b, 1)
            self.green_disc.set_pose(Pose.create_from_pq(p=green_xyz, q=disc_q))
            self.red_disc.set_pose(Pose.create_from_pq(p=red_xyz, q=disc_q))

            # Blue cube on green disc
            cube_xyz = green_xyz.clone()
            cube_xyz[:, 2] = self.DISC_HALF_HEIGHT * 2 + self.CUBE_HALF_SIZE
            cube_q = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device).repeat(b, 1)
            self.blue_cube.set_pose(Pose.create_from_pq(p=cube_xyz, q=cube_q))

            # Lamp (behind the discs)
            lamp_xyz = torch.zeros((b, 3), device=self.device)
            lamp_xyz[:, 0] = center_xy[:, 0] + self.LAMP_FORWARD_OFFSET
            lamp_xyz[:, 1] = center_xy[:, 1]
            lamp_xyz[:, 2] = 0.0
            lamp_q = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device).repeat(b, 1)
            self.lamp_body.set_pose(Pose.create_from_pq(p=lamp_xyz, q=lamp_q))
            self.lamp_bulb_off.set_pose(Pose.create_from_pq(p=lamp_xyz, q=lamp_q))

            lamp_hidden_xyz = lamp_xyz.clone()
            lamp_hidden_xyz[:, 2] += self.HEIGHT_OFFSET
            self.lamp_bulb_on.set_pose(Pose.create_from_pq(p=lamp_hidden_xyz, q=lamp_q))

            # Store positions
            self.green_disc_center[env_idx] = green_xyz
            self.red_disc_center[env_idx] = red_xyz
            self.lamp_on_pos[env_idx] = lamp_xyz
            self.lamp_off_pos[env_idx] = lamp_hidden_xyz

            # Signal timing
            pre_steps = torch.randint(
                low=self.PRE_SIGNAL_STEPS[0],
                high=self.PRE_SIGNAL_STEPS[1] + 1,
                size=(b,),
                device=self.device,
                dtype=torch.int64,
            )
            self.signal_step[env_idx] = pre_steps
            delay = self.DELAY_STEPS
            tol = self.TOLERANCE_FRAC
            self.window_start[env_idx] = pre_steps + int(delay * (1 - tol))
            self.window_end[env_idx] = pre_steps + int(delay * (1 + tol))

            # Reset state
            self.too_early[env_idx] = False
            self.success_flag[env_idx] = False
            self.failed[env_idx] = False

            # Oracle info: the delay value (constant per variant, but stored as tensor)
            self.oracle_info = torch.full(
                (self.num_envs,),
                self.DELAY_STEPS,
                dtype=torch.float32,
                device=self.device,
            )

            if self.robot_uids in ("panda", "panda_wristcam"):
                qpos = np.array([0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04])
                qpos[:-2] += self._episode_rng.normal(0, self.robot_init_qpos_noise, len(qpos) - 2)
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)

            if hasattr(self, "_prev_action") and torch.is_tensor(self._prev_action):
                if self._prev_action.shape[0] >= int(env_idx.max().item()) + 1:
                    self._prev_action[env_idx] = 0

    def evaluate(self):
        elapsed = self.elapsed_steps.to(torch.int64)
        if elapsed.dim() > 1:
            elapsed = elapsed.squeeze(-1)

        # --- Lamp control ---
        signal_on = elapsed >= self.signal_step

        # Swap lamp bulbs: off-bulb visible when no signal, on-bulb visible after signal
        off_pose = self.lamp_bulb_off.pose.raw_pose.clone()
        off_pose[signal_on, :3] = self.lamp_off_pos[signal_on]
        off_pose[~signal_on, :3] = self.lamp_on_pos[~signal_on]
        self.lamp_bulb_off.pose = off_pose

        on_pose = self.lamp_bulb_on.pose.raw_pose.clone()
        on_pose[signal_on, :3] = self.lamp_on_pos[signal_on]
        on_pose[~signal_on, :3] = self.lamp_off_pos[~signal_on]
        self.lamp_bulb_on.pose = on_pose

        # --- Cube position check ---
        cube_pos = self.blue_cube.pose.p
        cube_xy = cube_pos[:, :2]
        red_disc_xy = self.red_disc_center[:, :2]

        cube_to_red_dist = torch.linalg.norm(cube_xy - red_disc_xy, dim=1)
        cube_z = cube_pos[:, 2]
        cube_on_red = (cube_to_red_dist < self.GOAL_THRESH) & (
            cube_z < self.DISC_HALF_HEIGHT * 2 + self.CUBE_HALF_SIZE * 3
        )

        # --- Timing checks ---
        in_window = (elapsed >= self.window_start) & (elapsed <= self.window_end)
        before_window = signal_on & (elapsed < self.window_start)
        past_window = elapsed > self.window_end

        # Too-early placement
        self.too_early = self.too_early | (cube_on_red & before_window)

        # Success: cube on red disc during the window without early placement
        self.success_flag = self.success_flag | (cube_on_red & in_window & ~self.too_early)

        # Failure: placed too early, or window passed without success
        self.failed = self.failed | self.too_early | (past_window & ~self.success_flag)

        self.obj_to_goal_pos = self.red_disc.pose.p - self.blue_cube.pose.p
        is_grasped = self.agent.is_grasping(self.blue_cube)

        return {
            "success": self.success_flag,
            "failed": self.failed,
            "too_early": self.too_early,
            "signal_on": signal_on,
            "in_window": in_window,
            "before_window": before_window,
            "past_window": past_window,
            "cube_on_red": cube_on_red,
            "cube_to_red_dist": cube_to_red_dist,
            "is_grasped": is_grasped,
            "obj_to_goal_pos": self.obj_to_goal_pos,
            "task_cue": self.task_cue,
            "language_instruction": self._lang_instruction,
            "oracle_info": self.oracle_info,
            "reward_dict": self.reward_dict,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                green_disc_pose=self.green_disc.pose.raw_pose,
                red_disc_pose=self.red_disc.pose.raw_pose,
                cube_pose=self.blue_cube.pose.raw_pose,
                signal_on=info["signal_on"],
                in_window=info["in_window"],
                is_grasped=info["is_grasped"],
                oracle_info=self.oracle_info,
            )
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if isinstance(info, dict):
            success = info.get("success", None)
            failed = info.get("failed", None)

            if torch.is_tensor(terminated):
                terminated = terminated.to(dtype=torch.bool)
                if torch.is_tensor(success):
                    terminated = terminated | success.to(dtype=torch.bool)
                if torch.is_tensor(failed):
                    terminated = terminated | failed.to(dtype=torch.bool)
            else:
                terminated = bool(terminated)
                if success is not None:
                    if torch.is_tensor(success):
                        terminated = terminated or bool(success.any().item())
                    else:
                        terminated = terminated or bool(success)
                if failed is not None:
                    if torch.is_tensor(failed):
                        terminated = terminated or bool(failed.any().item())
                    else:
                        terminated = terminated or bool(failed)

        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.blue_cube.pose.p

        tcp_to_cube = cube_pos - tcp_pos
        tcp_to_cube_dist = torch.linalg.norm(tcp_to_cube, dim=1)
        reaching_reward = 1 - torch.tanh(10.0 * tcp_to_cube_dist)

        obj_to_goal_dist = torch.linalg.norm(info["obj_to_goal_pos"], dim=1)
        place_reward = 1 - torch.tanh(5.0 * obj_to_goal_dist)

        is_grasped = info["is_grasped"].float()

        static_reward = 1 - torch.tanh(5.0 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], dim=1))

        # Smoothness penalty
        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=self.device)
        if not hasattr(self, "_prev_action") or self._prev_action is None or self._prev_action.shape != action.shape:
            self._prev_action = torch.zeros_like(action)
        delta_action = action - self._prev_action
        action_l2 = torch.linalg.norm(action, dim=1)
        delta_action_l2 = torch.linalg.norm(delta_action, dim=1)
        qvel_l2 = torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], dim=1)
        smooth_penalty = (
            self.ACTION_L2_COEF * torch.tanh(2.0 * action_l2)
            + self.ACTION_DELTA_L2_COEF * torch.tanh(5.0 * delta_action_l2)
            + self.QVEL_L2_COEF * torch.tanh(2.0 * qvel_l2)
        )

        signal_on = info["signal_on"].float()
        in_window = info["in_window"].float()
        before_window = info["before_window"].float()
        cube_on_red = info["cube_on_red"].float()

        # Reward structure:
        #   After signal: reaching + grasping incentives (agent can prepare)
        #   In window: placement reward
        #   Before window: penalty for premature placement on red disc
        #   Throughout: smoothness penalty
        reward = (
            2.0 * reaching_reward * signal_on
            + 3.0 * is_grasped * signal_on
            + 20.0 * place_reward * is_grasped * in_window
            + 5.0 * cube_on_red * in_window
            + 3.0 * static_reward * cube_on_red * in_window
            - 5.0 * cube_on_red * before_window
            - smooth_penalty
        )

        reward *= signal_on
        reward[info["success"]] = self.SUCCESS_BONUS
        reward[info["failed"]] = -self.FAILURE_PENALTY

        self.reward_dict = {
            "reaching_reward": reaching_reward,
            "place_reward": place_reward,
            "is_grasped": is_grasped,
            "tcp_to_cube_dist": tcp_to_cube_dist,
            "obj_to_goal_dist": obj_to_goal_dist,
            "cube_on_red": cube_on_red,
            "static_reward": static_reward,
            "smooth_penalty": smooth_penalty,
            "action_l2": action_l2,
            "delta_action_l2": delta_action_l2,
            "qvel_l2": qvel_l2,
        }
        self._prev_action = action.detach()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / self.SUCCESS_BONUS


# ---------------------------------------------------------------------------
# Standard tasks
# ---------------------------------------------------------------------------
@register_env("TimedTransferEasy-VLA-v0", max_episode_steps=200)
class TimedTransferEasyVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 10 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 100


@register_env("TimedTransferMedium-VLA-v0", max_episode_steps=250)
class TimedTransferMediumVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 50 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 150


@register_env("TimedTransferHard-VLA-v0", max_episode_steps=300)
class TimedTransferHardVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 100 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 200


# ---------------------------------------------------------------------------
# Long-horizon tasks
# ---------------------------------------------------------------------------
@register_env("TimedTransferEasy-Long-VLA-v0", max_episode_steps=600)
class TimedTransferEasyLongVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 200 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 300


@register_env("TimedTransferMedium-Long-VLA-v0", max_episode_steps=900)
class TimedTransferMediumLongVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 500 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 500


@register_env("TimedTransferHard-Long-VLA-v0", max_episode_steps=1200)
class TimedTransferHardLongVLAEnv(TimedTransferVLABaseEnv):
    """Delay = 1000 steps, tolerance +/-5 %."""

    DELAY_STEPS: int = 1000
