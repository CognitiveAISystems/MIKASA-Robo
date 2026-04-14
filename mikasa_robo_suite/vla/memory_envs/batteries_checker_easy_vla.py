"""Easy Batteries Checker variants for the VLA benchmark."""

import torch
from mani_skill.utils.registration import register_env

from .batteries_checker_hard_vla import BatteriesCheckerVLABaseEnv


class BatteriesCheckerEasyVLABaseEnv(BatteriesCheckerVLABaseEnv):
    """Simplified version of Batteries Checker.

    In this task, the robot checks batteries one by one using the socket and
    the lamp. A tested battery does not need to be placed back manually: once
    the check is complete, the environment returns it to its original tray slot
    automatically. This keeps the memory component but removes part of the
    manipulation burden.

    Episode flow:
    - The robot picks one battery and inserts it into the socket.
    - The lamp reveals whether that battery is working.
    - The environment snaps the battery back to its home slot.
    - The robot presses the button to confirm that this battery was checked.

    Success (`success=True`):
    - Same criterion as the hard variant: all working batteries must be found
      through completed check-confirm cycles.

    How to customize:
    - `ACTIVE_BATTERY_COUNT` changes how many batteries are present and therefore
      how much search the agent must do.
    - `WORKING_BATTERY_COUNT` changes how many positive findings exist in the tray.
    - The hard-variant socket, tray, and button thresholds still control how
      strict insertion and confirmation are.
    """

    LANGUAGE_INSTRUCTION = "Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm."

    def evaluate(self):
        info = super().evaluate()

        auto_return_mask = (
            info["action_mask"] & (self.action_stage == self.STAGE_RETURN) & (self.active_battery_idx >= 0)
        )
        if not auto_return_mask.any():
            return info

        env_ids = torch.where(auto_return_mask)[0]
        active_idx = self.active_battery_idx[env_ids]
        home_pos = self.tray_slot_positions[env_ids, active_idx]
        home_quat = self.battery_quat.unsqueeze(0).repeat(env_ids.shape[0], 1)

        for bat_id in torch.unique(active_idx).tolist():
            bat_id = int(bat_id)
            per_battery_mask = active_idx == bat_id
            per_battery_env_ids = env_ids[per_battery_mask]
            if per_battery_env_ids.numel() == 0:
                continue

            raw_pose = self.batteries[bat_id].pose.raw_pose.clone()
            raw_pose[per_battery_env_ids, :3] = home_pos[per_battery_mask]
            raw_pose[per_battery_env_ids, 3:7] = home_quat[per_battery_mask]
            self.batteries[bat_id].pose = raw_pose

            zero_vel = torch.zeros((self.num_envs, 3), device=self.device)
            self.batteries[bat_id].set_linear_velocity(zero_vel)
            self.batteries[bat_id].set_angular_velocity(zero_vel)

        self.new_return_event[env_ids] = True
        self.action_stage[env_ids] = self.STAGE_CONFIRM

        info["new_return_event"] = self.new_return_event
        info["action_stage"] = self.action_stage
        info["socket_has_battery"][env_ids] = False
        info["socket_has_working"][env_ids] = False
        info["socket_battery_idx"][env_ids] = -1
        return info


@register_env("BatteriesCheckerEasy-3-VLA-v0", max_episode_steps=540)
class BatteriesCheckerEasy3VLAEnv(BatteriesCheckerEasyVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 3
    WORKING_BATTERY_COUNT = 1


@register_env("BatteriesCheckerEasy-6-VLA-v0", max_episode_steps=1080)
class BatteriesCheckerEasy6VLAEnv(BatteriesCheckerEasyVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 6
    WORKING_BATTERY_COUNT = 3


@register_env("BatteriesCheckerEasy-9-VLA-v0", max_episode_steps=1620)
class BatteriesCheckerEasy9VLAEnv(BatteriesCheckerEasyVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 9
    WORKING_BATTERY_COUNT = 5


@register_env("BatteriesCheckerEasy-12-VLA-v0", max_episode_steps=2160)
class BatteriesCheckerEasy12VLAEnv(BatteriesCheckerEasyVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 12
    WORKING_BATTERY_COUNT = 7


@register_env("BatteriesCheckerEasy-15-VLA-v0", max_episode_steps=2400)
class BatteriesCheckerEasy15VLAEnv(BatteriesCheckerEasyVLABaseEnv):
    ACTIVE_BATTERY_COUNT = 15
    WORKING_BATTERY_COUNT = 9
