import os
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces


def _put_text_with_outline(
    image,
    text,
    org,
    font_face,
    font_scale,
    color,
    thickness,
    line_type=cv2.LINE_AA,
):
    """Draw readable overlay text over light and dark render backgrounds."""
    outline_thickness = max(2, int(thickness) + 2)
    cv2.putText(
        image,
        text,
        org,
        font_face,
        font_scale,
        (20, 20, 20),
        outline_thickness,
        line_type,
    )
    cv2.putText(
        image,
        text,
        org,
        font_face,
        font_scale,
        color,
        thickness,
        line_type,
    )


class StateOnlyTensorToDictWrapper(gym.ObservationWrapper):
    """Wrapper that converts tensor observation to a dictionary with 'state' key."""

    def __init__(self, env):
        super().__init__(env)

        orig_obs_space = env.observation_space

        self.observation_space = spaces.Dict({"state": orig_obs_space})

    def observation(self, obs):
        if not isinstance(obs, dict):
            obs = {"state": obs}
            b_ = obs["state"].shape[0]
        else:
            obs = obs.copy()
            b_ = obs["agent"]["qpos"].shape[0]
            # obs.update({'rgb': self.unwrapped.rgb.unsqueeze(-1)})

        task_cue_ = self.unwrapped.task_cue
        oracle_info_ = self.unwrapped.oracle_info

        if task_cue_ is not None:
            if len(task_cue_.shape) == 1:
                task_cue_ = task_cue_.unsqueeze(-1)
        else:
            task_cue_ = torch.ones(b_, 1) * 4242424242

        if oracle_info_ is not None:
            if len(oracle_info_.shape) == 1:
                oracle_info_ = oracle_info_.unsqueeze(-1)
        else:
            oracle_info_ = torch.ones(b_, 1) * 4242424242

        obs.update({"task_cue": task_cue_, "oracle_info": oracle_info_})
        return obs


class ConvertJointsToEEFXyzRpyGripperWrapper(gym.ObservationWrapper):
    """Convert flattened joint-state input into observation['proprio'].

    The VLA-facing proprio vector is xyz(3) + rpy(3) + gripper(1).

    Expected source layout for flattened joints is:
    [tcp_pose(7), qpos(n), qvel(n), ...], where tcp_pose is [x, y, z, qw, qx, qy, qz].
    """

    def __init__(
        self,
        env,
        qpos_dim: Optional[int] = None,
        gripper_finger_dims: int = 2,
    ):
        super().__init__(env)
        if gripper_finger_dims <= 0:
            raise ValueError(f"gripper_finger_dims must be > 0, got {gripper_finger_dims}")

        self.qpos_dim = qpos_dim if qpos_dim is None else int(qpos_dim)
        self.gripper_finger_dims = int(gripper_finger_dims)

        if self.qpos_dim is None:
            self.qpos_dim = self._infer_qpos_dim_from_env()

        if isinstance(self.observation_space, spaces.Dict) and "proprio" in self.observation_space.spaces:
            new_spaces = dict(self.observation_space.spaces)
            proprio_space = new_spaces["proprio"]
            if isinstance(proprio_space, spaces.Box):
                shape = tuple(proprio_space.shape)
                new_shape = (7,) if len(shape) == 0 else (*shape[:-1], 7)
                low = np.full(new_shape, -np.inf, dtype=np.float32)
                high = np.full(new_shape, np.inf, dtype=np.float32)
                new_spaces["proprio"] = spaces.Box(low=low, high=high, dtype=np.float32)
                self.observation_space = spaces.Dict(new_spaces)

    def _infer_qpos_dim_from_env(self) -> Optional[int]:
        try:
            qpos = self.unwrapped.agent.robot.get_qpos()
            if torch.is_tensor(qpos):
                if qpos.ndim == 1:
                    return int(qpos.shape[0])
                if qpos.ndim >= 2:
                    return int(qpos.shape[-1])
            arr = np.asarray(qpos)
            if arr.ndim == 1:
                return int(arr.shape[0])
            if arr.ndim >= 2:
                return int(arr.shape[-1])
        except Exception:
            return None
        return None

    @staticmethod
    def _quat_wxyz_to_rpy_torch(quat: torch.Tensor) -> torch.Tensor:
        quat = quat.to(torch.float32)
        quat = quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1e-8)
        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        return torch.stack([roll, pitch, yaw], dim=-1)

    @staticmethod
    def _quat_wxyz_to_rpy_np(quat: np.ndarray) -> np.ndarray:
        quat = quat.astype(np.float32, copy=False)
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        quat = quat / np.clip(norm, 1e-8, None)

        w = quat[..., 0]
        x = quat[..., 1]
        y = quat[..., 2]
        z = quat[..., 3]

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32, copy=False)

    def _infer_qpos_dim_from_joints_dim(self, joints_dim: int) -> int:
        if joints_dim <= 7:
            raise RuntimeError(f"Cannot parse joints dim={joints_dim}; expected > 7.")

        if self.qpos_dim is not None:
            if 7 + self.qpos_dim <= joints_dim:
                return int(self.qpos_dim)
            raise RuntimeError(f"qpos_dim={self.qpos_dim} is incompatible with joints_dim={joints_dim}.")

        if (joints_dim - 7) % 2 == 0:
            return int((joints_dim - 7) // 2)
        return int(max(1, joints_dim - 7))

    def observation(self, obs):
        if not isinstance(obs, dict) or "proprio" not in obs:
            return obs

        out = obs.copy()
        joints = out["proprio"]
        is_torch = torch.is_tensor(joints)

        if is_torch:
            arr = joints.to(torch.float32)
        else:
            arr = np.asarray(joints, dtype=np.float32)

        if arr.shape[-1] == 7:
            out["proprio"] = arr
            return out

        original_shape = tuple(arr.shape)
        original_ndim = len(original_shape)
        if original_ndim == 1:
            flat = arr.reshape(1, -1)
        else:
            flat = arr.reshape(-1, original_shape[-1])

        joints_dim = int(flat.shape[-1])
        if joints_dim < 7:
            raise RuntimeError(f"Expected joints dim >= 7, got {joints_dim}.")

        qpos_dim = self._infer_qpos_dim_from_joints_dim(joints_dim)
        qpos_start = 7
        qpos_end = min(7 + qpos_dim, joints_dim)
        if qpos_end <= qpos_start:
            raise RuntimeError(f"Invalid qpos slice for joints_dim={joints_dim}, qpos_dim={qpos_dim}.")

        xyz = flat[:, :3]
        quat = flat[:, 3:7]
        qpos = flat[:, qpos_start:qpos_end]

        if is_torch:
            rpy = self._quat_wxyz_to_rpy_torch(quat)
            if qpos.shape[-1] >= self.gripper_finger_dims:
                gripper = torch.sum(qpos[:, -self.gripper_finger_dims :], dim=-1, keepdim=True)
            else:
                gripper = qpos[:, -1:].clone()
            proprio = torch.cat([xyz, rpy, gripper.to(torch.float32)], dim=-1)
        else:
            rpy = self._quat_wxyz_to_rpy_np(quat)
            if qpos.shape[-1] >= self.gripper_finger_dims:
                gripper = np.sum(qpos[:, -self.gripper_finger_dims :], axis=-1, keepdims=True)
            else:
                gripper = qpos[:, -1:].copy()
            proprio = np.concatenate([xyz, rpy, gripper.astype(np.float32, copy=False)], axis=-1).astype(
                np.float32, copy=False
            )

        if original_ndim == 1:
            out["proprio"] = proprio[0]
        else:
            out["proprio"] = proprio.reshape(*original_shape[:-1], 7)
        return out


# class StateOnlyTensorToDictWrapper(gym.ObservationWrapper):
#     """Wrapper that converts tensor observation to a dictionary with 'state' key."""

#     def __init__(self, env):
#         super().__init__(env)

#         orig_obs_space = env.observation_space

#         self.observation_space = spaces.Dict({
#             'state': orig_obs_space
#         })

#     def observation(self, obs):
#         return {'state': obs, 'task_cue': self.unwrapped.task_cue.unsqueeze(-1)}

# class RotateAddAngleObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)

#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped


#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['oracle_info'] = self.angle_diff.unsqueeze(-1)
#         return obs

# class RotateAddAngleObservationWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)

#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped


#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['target_angle'] = self.target_angle
#         return obs


class RotateRenderAngleInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        self.current_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.current_obs = obs
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def render(self):
        # Get the base render from the environment
        frame = self.env.render()

        # Add text
        for i in range(len(frame)):
            # if isinstance(self.current_obs, dict):
            target_angle = str(np.round(self.info["task_cue"][i].item() * 180 / np.pi, 2))
            current_angle = str(np.round(self.info["relative_angle"][i].item() * 180 / np.pi, 2))
            _put_text_with_outline(
                frame[i],
                "Target : " + target_angle + " deg",
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA,
            )

            _put_text_with_outline(
                frame[i],
                "Current: " + current_angle + " deg",
                (10, 120),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA,
            )

            # _put_text_with_outline(
            #     frame[i],
            #     'Error: ' + error_angle + ' deg',
            #     (10, 120),  # position
            #     cv2.FONT_HERSHEY_SIMPLEX,  # font
            #     1.0,  # font scale
            #     (255, 255, 255),  # color (white)
            #     2,  # thickness
            #     cv2.LINE_AA
            # )

        return frame


class RenderStepInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        self.current_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_obs = obs
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        # Add text
        for i in range(len(frame)):
            img = np.ascontiguousarray(frame[i])
            # Env. step
            _put_text_with_outline(
                img,
                f"Step: {self.step_count[i]}",
                (10, 30),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA,
            )
            frame[i] = img

        return frame


class RenderRewardInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current reward on the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        self.reward = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.reward = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.reward = reward
        return obs, reward, terminated, truncated, info

    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        # Used by benchmark-video tooling to keep step/env overlays but suppress reward text.
        disable_reward_overlay = str(os.getenv("MIKASA_DISABLE_REWARD_OVERLAY", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if disable_reward_overlay:
            return frame

        for i in range(len(frame)):
            if self.reward is not None:
                render_reward = self.reward[i].detach().cpu().numpy()
            else:
                render_reward = 0.0
            img = np.ascontiguousarray(frame[i])
            _put_text_with_outline(
                img,
                f"Reward: {render_reward:.3f}",
                (10, 90),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA,
            )
            frame[i] = img

        return frame


class RenderPressProgressInfoWrapper(gym.Wrapper):
    """
    Renders button press progress:
      - raw presses (instant physical presses)
      - confirmed presses (completed press cycles)
      - target presses
    """

    def __init__(self, env):
        super().__init__(env)
        self.info = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        return obs, reward, terminated, truncated, info

    def _to_scalar(self, x, i):
        if torch.is_tensor(x):
            return int(x[i].detach().cpu().item())
        if isinstance(x, np.ndarray):
            return int(x[i].item())
        if isinstance(x, (list, tuple)):
            return int(x[i])
        return int(x)

    def _fallback_progress_from_env(self):
        base_env = self.env.unwrapped
        if not hasattr(base_env, "target_blinks"):
            return None, None, None
        target = base_env.target_blinks
        raw_current = None
        confirmed_current = None
        if hasattr(base_env, "raw_press_count"):
            raw_current = base_env.raw_press_count
        if hasattr(base_env, "press_count"):
            confirmed_current = base_env.press_count
        return raw_current, confirmed_current, target

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        raw_press_count = None
        confirmed_press_count = None
        target_blinks = None
        if self.info is not None and "target_blinks" in self.info:
            raw_press_count = self.info.get("raw_press_count", None)
            confirmed_press_count = self.info.get("press_count", None)
            target_blinks = self.info["target_blinks"]
        if target_blinks is None or (raw_press_count is None and confirmed_press_count is None):
            raw_fb, conf_fb, target_fb = self._fallback_progress_from_env()
            if raw_press_count is None:
                raw_press_count = raw_fb
            if confirmed_press_count is None:
                confirmed_press_count = conf_fb
            if target_blinks is None:
                target_blinks = target_fb
        if target_blinks is None:
            return frame
        for i in range(len(frame)):
            raw_done = self._to_scalar(raw_press_count, i) if raw_press_count is not None else -1
            conf_done = self._to_scalar(confirmed_press_count, i) if confirmed_press_count is not None else -1
            total = self._to_scalar(target_blinks, i)
            img = np.ascontiguousarray(frame[i])
            _put_text_with_outline(
                img,
                f"Press raw/conf: {raw_done}/{conf_done}/{total}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frame[i] = img

        return frame


class RenderWorkingBatteriesInfoWrapper(gym.Wrapper):
    """Renders progress of discovered working batteries: found / target."""

    def __init__(self, env):
        super().__init__(env)
        self.info = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        return obs, reward, terminated, truncated, info

    def _to_scalar(self, x, i):
        if torch.is_tensor(x):
            return int(x[i].detach().cpu().item())
        if isinstance(x, np.ndarray):
            return int(x[i].item())
        if isinstance(x, (list, tuple)):
            return int(x[i])
        return int(x)

    def _fallback_progress_from_env(self):
        base_env = self.env.unwrapped
        found = getattr(base_env, "found_working_count", None)
        target = getattr(base_env, "target_working_count", None)
        return found, target

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        found = None
        target = None
        if self.info is not None:
            found = self.info.get("found_working_count", None)
            target = self.info.get("target_working_count", None)
        if found is None or target is None:
            fb_found, fb_target = self._fallback_progress_from_env()
            if found is None:
                found = fb_found
            if target is None:
                target = fb_target
        if found is None or target is None:
            return frame

        for i in range(len(frame)):
            done = self._to_scalar(found, i)
            total = self._to_scalar(target, i)
            img = np.ascontiguousarray(frame[i])
            _put_text_with_outline(
                img,
                f"Working found: {done}/{total}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frame[i] = img
        return frame


class CameraShutdownWrapper(gym.Wrapper):
    r"""Wrapper that zeros out all camera observations

    if n_initial_steps = 4 then t \in [0, 4] (5 steps) action is zero
    if n_initial_steps = 9 then t \in [0, 9] (10 steps) action is zero
    if n_initial_steps = 19 then t \in [0, 19] (20 steps) action is zero

    """

    def __init__(self, env, n_initial_steps=19):
        super().__init__(env)

        render_camera_config = env.unwrapped._default_human_render_camera_configs
        self.width = render_camera_config.width
        self.height = render_camera_config.height

        self.n_initial_steps = n_initial_steps
        self.current_steps = None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_steps = info["elapsed_steps"].detach().cpu().numpy()

        # Zero out camera observations if they exist
        if (self.current_steps > self.n_initial_steps).any():
            if isinstance(obs, dict):
                for key in obs:
                    if "sensor_data" in key:
                        for key2 in obs["sensor_data"]:
                            if "hand_camera" in key2:
                                for key3 in obs[key][key2]:
                                    obs[key][key2][key3] *= 0
                            if "base_camera" in key2:
                                for key3 in obs[key][key2]:
                                    obs[key][key2][key3] *= 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_steps = info["elapsed_steps"].detach().cpu().numpy()

        return obs, info

    def render(self):
        img = self.env.render()
        if (self.current_steps > self.n_initial_steps).any():
            img[:, :, self.width :, :] *= 0

        return img


# class ShellGameAddBallInfoWrapper(gym.ObservationWrapper):
# ! not need now
#     """
#     A wrapper for the ShellGamePush and ShellGamePick environments that adds oracle information about the ball's position to the observation space.

#     This wrapper is intended for use during testing or oracle training only. It should not be used during memory evaluation
#     as it provides additional information that would not be available in a real-world scenario.

#     Attributes:
#         env (gym.Env): The environment to be wrapped.

#     Methods:
#         observation(obs): Modifies the observation to include the ball's position.
#     """
#     def __init__(self, env):
#         super().__init__(env)

#         init_obs = self.observation(self.base_env._init_raw_obs)
#         self.base_env.update_obs_space(init_obs)

#     @property
#     def base_env(self) -> BaseEnv:
#         return self.env.unwrapped


#     def observation(self, obs):
#         if isinstance(obs, dict):
#             obs = obs.copy()
#             obs['cup_with_ball_number'] = self.cup_with_ball_number
#         return obs


class InitialZeroActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_initial_steps=1):
        """
        A wrapper that forces zero actions for a specified number of initial steps in the environment.

        Args:
            env: environment
            n_initial_steps: number of steps with zero actions
        """
        super().__init__(env)
        self.n_initial_steps = n_initial_steps
        self.current_steps = None

    def action(self, action):
        """Modifies action before sending it to the environment"""
        if self.current_steps is None or (self.current_steps < self.n_initial_steps).any():
            # Zero out actions for environments still in initial phase
            mask = self.current_steps < self.n_initial_steps
            modified_action = action.clone()
            modified_action[mask] = 0
            return modified_action
        return action

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_steps = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the step counter"""
        obs, info = super().reset(**kwargs)
        self.current_steps = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info


class CurriculumPhaseNoopActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        """
        Forces zero actions during cue+empty phases for curriculum memory tasks.

        Expects the base env to expose:
          - cue_steps_per_env (torch.Tensor[int64], shape [num_envs])
          - empty_steps_per_env (torch.Tensor[int64], shape [num_envs])
        """
        super().__init__(env)
        self.current_steps = None

    def _to_torch(self, x):
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x)

    def _get_noop_mask(self):
        base_env = self.env.unwrapped
        if self.current_steps is None or not hasattr(base_env, "cue_steps_per_env"):
            return None

        current_steps = self._to_torch(self.current_steps).to(torch.int64)
        cue_steps = self._to_torch(base_env.cue_steps_per_env).to(torch.int64)
        freeze_until = cue_steps
        if hasattr(base_env, "empty_steps_per_env"):
            freeze_until = freeze_until + self._to_torch(base_env.empty_steps_per_env).to(torch.int64)
        return current_steps < freeze_until

    @staticmethod
    def _batch_action_for_mask(action, noop_mask):
        """Expand a broadcastable flat action before applying a per-env mask."""
        batch_size = int(noop_mask.numel())
        if torch.is_tensor(action):
            if action.ndim == 1:
                return action.unsqueeze(0).expand(batch_size, -1).clone()
            return action.clone()

        modified = np.array(action, copy=True)
        if modified.ndim == 1:
            return np.broadcast_to(modified, (batch_size, *modified.shape)).copy()
        return modified

    def action(self, action):
        noop_mask = self._get_noop_mask()
        if noop_mask is None:
            return action
        if not noop_mask.any().item():
            return action

        modified_action = self._batch_action_for_mask(action, noop_mask)
        if torch.is_tensor(modified_action):
            modified_action[noop_mask.to(device=modified_action.device)] = 0
        else:
            modified_action[noop_mask.detach().cpu().numpy()] = 0
        return modified_action

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_steps = info["elapsed_steps"]
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_steps = info["elapsed_steps"]
        return obs, info


class CurriculumPhaseNoopActionWrapperPdJointPos(CurriculumPhaseNoopActionWrapper):
    """Curriculum-phase noop wrapper for envs running in `pd_joint_pos` control mode.

    Plain `CurriculumPhaseNoopActionWrapper` sends action = 0, which in
    `pd_joint_pos` would command the robot to move toward qpos = [0, ..., 0]
    instead of holding the current pose. This subclass overrides the noop
    action to be the robot's current arm qpos plus a normalized gripper
    command — i.e., "stay where you are".
    """

    GRIPPER_LOW = -0.01
    GRIPPER_HIGH = 0.04

    def _build_hold_action(self, action_template):
        base_env = self.env.unwrapped
        robot = base_env.agent.robot

        qpos = robot.get_qpos()  # (n, 9) panda: 7 arm + 2 finger joints (mimic)
        qpos_arm = qpos[..., :-2].detach().cpu().numpy()  # (n, 7)
        qpos_gripper = qpos[..., -2].detach().cpu().numpy()  # (n,)

        mid = 0.5 * (self.GRIPPER_HIGH + self.GRIPPER_LOW)
        half = 0.5 * (self.GRIPPER_HIGH - self.GRIPPER_LOW)
        grip_norm = (qpos_gripper - mid) / half
        grip_norm = np.clip(grip_norm, -1.0, 1.0)

        hold = np.concatenate([qpos_arm, grip_norm[..., None]], axis=1).astype(np.float32)

        if np.asarray(action_template).ndim == 1:
            return hold[0]
        return hold

    def action(self, action):
        noop_mask = self._get_noop_mask()
        if noop_mask is None or not noop_mask.any().item():
            return action

        modified = self._batch_action_for_mask(action, noop_mask)
        if isinstance(modified, np.ndarray):
            hold = self._build_hold_action(modified)
            mask_np = noop_mask.detach().cpu().numpy()
            modified[mask_np] = hold[mask_np]
            return modified

        hold_np = self._build_hold_action(modified.detach().cpu().numpy())
        hold_t = torch.as_tensor(hold_np, dtype=modified.dtype, device=modified.device)
        mask_t = noop_mask.to(device=modified.device)
        modified[mask_t] = hold_t[mask_t]
        return modified


class ShellGameRenderCupInfoWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def _oracle_text(self, i: int) -> str:
        if self.info is None or "oracle_info" not in self.info:
            return "Target: N/A"

        value = self.info["oracle_info"][i]
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.asarray(arr)

        if arr.size == 0:
            return "Target: N/A"
        if arr.size == 1:
            idx = int(arr.reshape(-1)[0])
            if idx == 0:
                return "Target: Left"
            if idx == 1:
                return "Target: Center"
            if idx == 2:
                return "Target: Right"
            return f"Target: {idx}"

        vals = [int(x) for x in arr.reshape(-1).tolist()]
        return f"Oracle: {vals}"

    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        # Add text
        for i in range(len(frame)):
            cup = self._oracle_text(i)
            img = np.ascontiguousarray(frame[i])
            # Target cup
            _put_text_with_outline(
                img,
                cup,
                (10, 60),  # position
                cv2.FONT_HERSHEY_SIMPLEX,  # font
                1.0,  # font scale
                (255, 255, 255),  # color (white)
                2,  # thickness
                cv2.LINE_AA,
            )
            frame[i] = img

        return frame


class DebugRewardWrapper(gym.Wrapper):
    """
    A wrapper that renders the current step count and target cup on the screen.
    """

    def __init__(self, env):
        super().__init__(env)
        self.info = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        return obs, reward, terminated, truncated, info

    def render(self):
        # Get the base render from the environment
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        if os.environ.get("MIKASA_DISABLE_REWARD_OVERLAY", "0") == "1":
            return frame

        for i in range(len(frame)):
            if "reward_dict" in self.info and self.info["reward_dict"] is not None:
                for reward_num, (reward_key, reward_value) in enumerate(self.info["reward_dict"].items()):
                    img = np.ascontiguousarray(frame[i])
                    _put_text_with_outline(
                        img,
                        f"{reward_key}: {reward_value[i].detach().cpu().numpy():.3f}",
                        (10, 150 + (reward_num + 1) * 20),  # position
                        cv2.FONT_HERSHEY_SIMPLEX,  # font
                        0.5,  # font scale
                        (255, 255, 255),  # color (white)
                        1,  # thickness
                        cv2.LINE_AA,
                    )
                    frame[i] = img

        return frame


class RememberColorInfoWrapper(gym.Wrapper):
    """Render the target color as a color swatch for color-memory tasks."""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

    def _decode_color_rgb(self, color_id: int):
        color_dict = getattr(self.env.unwrapped, "color_dict", {})
        if color_id in color_dict:
            rgb = np.asarray(color_dict[color_id][:3], dtype=np.float32)
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            return int(rgb[0]), int(rgb[1]), int(rgb[2])
        return 255, 255, 255

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        if self.info is None or "oracle_info" not in self.info:
            return frame

        target_text = "Target:"
        (text_width, _), _ = cv2.getTextSize(
            target_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2,
        )
        square_size = 22
        square_x = 10 + text_width + 10
        square_y = 38

        for i in range(len(frame)):
            color_idx = int(self.info["oracle_info"][i].item())
            img = np.ascontiguousarray(frame[i])

            _put_text_with_outline(
                img,
                target_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                img,
                (square_x, square_y),
                (square_x + square_size, square_y + square_size),
                self._decode_color_rgb(color_idx),
                -1,
            )
            cv2.rectangle(
                img,
                (square_x, square_y),
                (square_x + square_size, square_y + square_size),
                (255, 255, 255),
                2,
            )
            frame[i] = img

        return frame


class RememberShapeInfoWrapper(gym.Wrapper):
    """Render target shape for remember-shape tasks."""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None
        self.SHAPES_names = {
            0: "cube",
            1: "sphere",
            2: "cylinder",
            3: "cross",
            4: "torus",
            5: "star",
            6: "pyramide",
            7: "t_shape",
            8: "crescent",
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        if self.info is None or "oracle_info" not in self.info:
            return frame

        for i in range(len(frame)):
            shape_idx = int(self.info["oracle_info"][i].item())
            shape_name = self.SHAPES_names.get(shape_idx, str(shape_idx))
            text = f"Target: {shape_name}"

            img = np.ascontiguousarray(frame[i])
            _put_text_with_outline(
                img,
                text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            frame[i] = img

        return frame


class RememberShapeAndColorInfoWrapper(gym.Wrapper):
    """Render target shape+color text for remember-shape-and-color tasks."""

    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

        self._env = env
        self.shape_dict = self._env.BASE_SHAPES

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def decode_shape(self, shape_id: int) -> str:
        return self.shape_dict.get(shape_id, "Unknown")

    def _decode_color_rgb(self, color_id: int):
        # COLOR_PALETTE stores RGBA in [0, 1]. Frames here are RGB, so keep RGB channel order.
        if hasattr(self._env, "COLOR_PALETTE") and color_id in self._env.COLOR_PALETTE:
            rgba = np.asarray(self._env.COLOR_PALETTE[color_id], dtype=np.float32)
            rgb = np.clip(rgba[:3] * 255.0, 0.0, 255.0).astype(np.uint8)
            return int(rgb[0]), int(rgb[1]), int(rgb[2])

        fallback = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255),
        }
        return fallback.get(color_id, (255, 255, 255))

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        if self.info is None or "oracle_info" not in self.info:
            return frame

        for i in range(len(frame)):
            shape_id = int(self.info["oracle_info"][i][0].item())
            color_id = int(self.info["oracle_info"][i][1].item())

            shape_name = self.decode_shape(shape_id)
            color_rgb = self._decode_color_rgb(color_id)

            img = np.ascontiguousarray(frame[i])
            target_text = "Target:"
            _put_text_with_outline(
                img,
                target_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            (text_width, _), _ = cv2.getTextSize(
                target_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                2,
            )
            square_size = 22
            square_x = 10 + text_width + 10
            square_y = 38
            cv2.rectangle(
                img,
                (square_x, square_y),
                (square_x + square_size, square_y + square_size),
                color_rgb,
                -1,
            )
            cv2.rectangle(
                img,
                (square_x, square_y),
                (square_x + square_size, square_y + square_size),
                (255, 255, 255),
                2,
            )

            _put_text_with_outline(
                img,
                shape_name,
                (square_x + square_size + 12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            frame[i] = img

        return frame


class RenderTraceShapeDebugWrapper(gym.Wrapper):
    """Debug overlay for TraceShape and TraceShapeSeq tasks."""

    SHAPE_NAMES = {0: "Circle", 1: "Square", 2: "Triangle"}

    def __init__(self, env, minimap_size=160, minimap_top=130):
        super().__init__(env)
        self.minimap_size = minimap_size
        self.minimap_margin = 10
        self.minimap_top = minimap_top
        self.info = None
        self._trails = {}

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        n = len(info["elapsed_steps"]) if "elapsed_steps" in info else 1
        self._trails = {i: [] for i in range(n)}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        base_env = self.env.unwrapped
        if hasattr(base_env, "green_cube"):
            green_xy = base_env.green_cube.pose.p[:, :2].detach().cpu().numpy()
            action_mask = info.get("action_mask", None)
            if action_mask is not None:
                if torch.is_tensor(action_mask):
                    action_mask = action_mask.detach().cpu().numpy()
                for i in range(len(green_xy)):
                    if action_mask[i]:
                        self._trails.setdefault(i, []).append(green_xy[i].copy())
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _bool_from_info(self, key, idx):
        if self.info is None or key not in self.info:
            return None
        value = self.info[key]
        if torch.is_tensor(value):
            return bool(value[idx].item())
        return bool(np.asarray(value)[idx])

    def _get_active_shape_idx(self, base_env, env_idx: int, seq_len: int) -> int:
        if self.info is not None and "active_shape_idx" in self.info:
            active_val = self.info["active_shape_idx"]
            if torch.is_tensor(active_val):
                active_idx = int(active_val[env_idx].item())
            else:
                active_idx = int(np.asarray(active_val)[env_idx])
        elif hasattr(base_env, "active_shape_idx"):
            active_idx = int(base_env.active_shape_idx[env_idx].item())
        else:
            active_idx = 0
        max_idx = max(seq_len - 1, 0)
        return int(np.clip(active_idx, 0, max_idx))

    def _extract_trace_view(self, base_env, env_idx: int):
        wp_raw = self._to_numpy(base_env.waypoints[env_idx])
        if wp_raw.ndim == 2:
            waypoints = wp_raw
            checkpoints = self._to_numpy(base_env.checkpoints[env_idx])
            visited = self._to_numpy(base_env.checkpoint_visited[env_idx]).astype(bool)
            shape_id = int(base_env.shape_type[env_idx].item()) if hasattr(base_env, "shape_type") else -1
            seq_text = None
            return waypoints, checkpoints, visited, shape_id, seq_text

        seq_len = int(base_env.sequence_len[env_idx].item()) if hasattr(base_env, "sequence_len") else wp_raw.shape[0]
        seq_len = max(seq_len, 1)
        active_idx = self._get_active_shape_idx(base_env, env_idx, seq_len)

        waypoints = self._to_numpy(base_env.waypoints[env_idx, active_idx])
        checkpoints = self._to_numpy(base_env.checkpoints[env_idx, active_idx])
        visited = self._to_numpy(base_env.checkpoint_visited[env_idx, active_idx]).astype(bool)

        shape_id = -1
        if hasattr(base_env, "shape_sequence"):
            shape_id = int(base_env.shape_sequence[env_idx, active_idx].item())

        done_count = 0
        if hasattr(base_env, "shape_closed"):
            shape_closed = self._to_numpy(base_env.shape_closed[env_idx, :seq_len]).astype(bool)
            done_count = int(shape_closed.sum())
        seq_text = f"Seq {active_idx + 1}/{seq_len} Done {done_count}/{seq_len}"
        return waypoints, checkpoints, visited, shape_id, seq_text

    def _to_px(self, xy, center, scale, x0, y0, size):
        px = int((xy[0] - center[0]) * scale + size / 2) + x0
        py = int((xy[1] - center[1]) * (-scale) + size / 2) + y0
        return (px, py)

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)
        single_frame = frame.ndim == 3
        if single_frame:
            frame = frame[None, ...]

        base_env = self.env.unwrapped
        if not hasattr(base_env, "waypoints") or not hasattr(base_env, "checkpoints"):
            return frame[0] if single_frame else frame

        size = self.minimap_size
        margin = self.minimap_margin

        for i in range(len(frame)):
            img = np.ascontiguousarray(frame[i])
            h, w = img.shape[:2]

            waypoints, checkpoints, visited, shape_id, seq_text = self._extract_trace_view(base_env, i)
            if waypoints.shape[0] == 0 or checkpoints.shape[0] == 0:
                frame[i] = img
                continue

            center = waypoints.mean(axis=0)
            extent = max(np.max(np.abs(waypoints - center)), 0.01)

            # Keep the minimap on the main view instead of the right camera strip.
            x0 = margin
            text_height = 72 if seq_text is not None else 54
            max_top = max(margin, h - size - text_height - margin)
            y0 = min(max(margin, self.minimap_top), max_top)

            overlay = img.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (80, 80, 80), 1)

            scale = (size - 20) / (2 * extent * 1.3)

            def to_px(xy):
                return self._to_px(xy, center, scale, x0, y0, size)

            pts = [to_px(waypoints[j]) for j in range(len(waypoints))]
            pts.append(pts[0])
            for j in range(len(pts) - 1):
                cv2.line(img, pts[j], pts[j + 1], (255, 255, 0), 1, cv2.LINE_AA)

            trail = self._trails.get(i, [])
            if len(trail) > 1:
                trail_pts = [to_px(p) for p in trail]
                for j in range(len(trail_pts) - 1):
                    cv2.line(img, trail_pts[j], trail_pts[j + 1], (50, 220, 50), 1, cv2.LINE_AA)

            for j in range(len(checkpoints)):
                px = to_px(checkpoints[j])
                color = (0, 200, 0) if visited[j] else (0, 0, 200)
                cv2.circle(img, px, 4, color, -1, cv2.LINE_AA)
                cv2.circle(img, px, 4, (255, 255, 255), 1, cv2.LINE_AA)

            green_xy = base_env.green_cube.pose.p[i, :2].detach().cpu().numpy()
            gpx = to_px(green_xy)
            cv2.circle(img, gpx, 5, (50, 255, 50), -1, cv2.LINE_AA)
            cv2.circle(img, gpx, 5, (255, 255, 255), 2, cv2.LINE_AA)

            shape_name = self.SHAPE_NAMES.get(shape_id, f"Shape {shape_id}")
            n_visited = int(visited.sum())
            n_total = len(visited)
            _put_text_with_outline(
                img,
                f"{shape_name} [{n_visited}/{n_total}]",
                (x0, y0 + size + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            closed = self._bool_from_info("is_active_contour_closed", i)
            if closed is None:
                closed = self._bool_from_info("is_contour_closed", i)
            if closed is None:
                start_cp = checkpoints[0]
                start_dist = float(np.linalg.norm(green_xy - start_cp))
                cp_thresh = float(getattr(base_env, "CHECKPOINT_THRESH", 0.035))
                closed = bool(n_visited == n_total and start_dist < cp_thresh)
            _put_text_with_outline(
                img,
                f"Closed: {'YES' if closed else 'NO'}",
                (x0, y0 + size + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0) if closed else (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

            phase = "PRE-DEMO"
            if self.info is not None:
                if "action_mask" in self.info:
                    am = self.info["action_mask"]
                    dm = self.info.get("demo_mask", None)
                    if torch.is_tensor(am):
                        am = am[i].item()
                    if am:
                        phase = "ACTION"
                    elif dm is not None:
                        if torch.is_tensor(dm):
                            dm = dm[i].item()
                        if dm:
                            phase = "DEMO"

            phase_colors = {
                "PRE-DEMO": (200, 200, 200),
                "DEMO": (0, 0, 255),
                "ACTION": (0, 200, 0),
            }
            _put_text_with_outline(
                img,
                phase,
                (x0, y0 + size + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                phase_colors.get(phase, (255, 255, 255)),
                1,
                cv2.LINE_AA,
            )
            if seq_text is not None:
                _put_text_with_outline(
                    img,
                    seq_text,
                    (x0, y0 + size + 72),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

            frame[i] = img

        return frame[0] if single_frame else frame


class MemoryCapacityInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.step_count = 0
        self.current_obs = None
        self.info = None

        self._env = env
        self.color_dict = self._env.color_dict
        self.colors_names = {
            0: "Red",
            1: "Green",
            2: "Blue",
            3: "Yellow",
            4: "Magenta",
            5: "Cyan",
            6: "Maroon",
            7: "Olive",
            8: "Teal",
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def _decode_color_rgb(self, color_id: int):
        if color_id in self.color_dict:
            rgb = np.asarray(self.color_dict[color_id][:3], dtype=np.float32)
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            return int(rgb[0]), int(rgb[1]), int(rgb[2])
        return 255, 255, 255

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        if self.info is None or "oracle_info" not in self.info:
            return frame
        if not hasattr(self._env, "touched_cubes"):
            return frame

        for i in range(len(frame)):
            seq_of_cubes = self.info["oracle_info"][i]
            if torch.is_tensor(seq_of_cubes):
                seq_of_cubes = seq_of_cubes.detach().cpu().numpy()
            else:
                seq_of_cubes = np.asarray(seq_of_cubes)

            touched_cubes = self._env.touched_cubes[i]
            if torch.is_tensor(touched_cubes):
                touched_cubes = touched_cubes.detach().cpu().numpy()
            else:
                touched_cubes = np.asarray(touched_cubes)

            img = np.ascontiguousarray(frame[i])

            _put_text_with_outline(
                img,
                "Target: ",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            (text_width, _), _ = cv2.getTextSize(
                "Target: ",
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                2,
            )

            current_x = 10 + text_width
            square_size = 15
            for color_id in seq_of_cubes.reshape(-1).tolist():
                color_id = int(color_id)
                x1 = int(current_x)
                y1 = 45
                x2 = x1 + square_size
                y2 = y1 + square_size

                fill_color = self._decode_color_rgb(color_id)
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    fill_color,
                    -1,
                )

                is_touched = bool(touched_cubes[color_id]) if color_id < len(touched_cubes) else False
                outline_color = (255, 255, 255) if is_touched else (0, 0, 0)
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    outline_color,
                    2,
                )

                current_x += square_size + 10

            frame[i] = img

        return frame


class RenderTimedTransferInfoWrapper(gym.Wrapper):
    """Renders a countdown and timing info for TimedTransfer tasks.

    Displayed on each frame:
      - Steps remaining until the target placement moment
      - Current phase (WAIT / COUNTING / WINDOW / LATE)
      - Window bounds and cube-on-red status
    """

    def __init__(self, env):
        super().__init__(env)
        self.info = None
        self.step_count = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.info = info
        self.step_count = info["elapsed_steps"].detach().cpu().numpy()
        return obs, reward, terminated, truncated, info

    def _to_scalar(self, x, i):
        if torch.is_tensor(x):
            return int(x[i].detach().cpu().item())
        if isinstance(x, np.ndarray):
            return int(x[i].item())
        return int(x)

    def render(self):
        frame = self.env.render()
        if torch.is_tensor(frame):
            frame = frame.detach().cpu().numpy()
        frame = np.ascontiguousarray(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8, copy=False)

        base_env = self.env.unwrapped
        if not hasattr(base_env, "signal_step") or not hasattr(base_env, "window_start"):
            return frame

        for i in range(len(frame)):
            img = np.ascontiguousarray(frame[i])
            h, w = img.shape[:2]

            elapsed = self._to_scalar(self.step_count, i) if self.step_count is not None else 0
            signal = self._to_scalar(base_env.signal_step, i)
            delay = int(base_env.DELAY_STEPS)
            target_step = signal + delay
            w_start = self._to_scalar(base_env.window_start, i)
            w_end = self._to_scalar(base_env.window_end, i)

            # Countdown
            remaining = max(0, target_step - elapsed)

            # Phase
            if elapsed < signal:
                phase = "WAIT"
                phase_color = (200, 200, 200)
            elif elapsed < w_start:
                phase = "COUNTING"
                phase_color = (0, 200, 255)
            elif elapsed <= w_end:
                phase = "WINDOW"
                phase_color = (0, 255, 0)
            else:
                phase = "LATE"
                phase_color = (0, 0, 255)

            # Cube on red
            cube_on_red = False
            if self.info is not None and "cube_on_red" in self.info:
                val = self.info["cube_on_red"]
                if torch.is_tensor(val):
                    cube_on_red = bool(val[i].item())
                else:
                    cube_on_red = bool(np.asarray(val).reshape(-1)[i])

            # Draw on the right side of the frame to avoid overlap with other wrappers
            rx = w - 260

            # Draw countdown (large)
            _put_text_with_outline(
                img,
                f"T-{remaining}",
                (rx, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Draw phase
            _put_text_with_outline(
                img,
                phase,
                (rx, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                phase_color,
                2,
                cv2.LINE_AA,
            )

            # Draw window info
            _put_text_with_outline(
                img,
                f"[{w_start},{w_end}] d={delay}",
                (rx, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Draw cube status
            cube_text = "ON RED" if cube_on_red else "not on red"
            cube_color = (0, 255, 0) if cube_on_red else (150, 150, 150)
            _put_text_with_outline(
                img,
                cube_text,
                (rx, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                cube_color,
                1,
                cv2.LINE_AA,
            )

            frame[i] = img

        return frame
