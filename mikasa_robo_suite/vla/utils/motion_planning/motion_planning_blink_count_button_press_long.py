import argparse
import os
import sys
import types
import warnings
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


def _configure_runtime_warnings_and_env():
    # Silence known third-party warnings for cleaner CLI logs.
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="Failed to find Vulkan ICD file.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

    # Help SAPIEN discover Vulkan ICD on common Linux paths.
    if not os.environ.get("VK_ICD_FILENAMES"):
        for icd in (
            "/etc/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
        ):
            if os.path.exists(icd):
                os.environ["VK_ICD_FILENAMES"] = icd
                break


_configure_runtime_warnings_and_env()

import gymnasium as gym
import numpy as np
import sapien
import torch
from IPython.display import Video
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.utils.wrappers.record import RecordEpisode
from transforms3d.euler import quat2euler
from transforms3d.quaternions import qinverse, qmult

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.memory_envs import *
from mikasa_robo_suite.vla.utils.wrappers import *

"""
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press_long.py --env-id BlinkCountButtonPressEasy-Long-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press_long.py --env-id BlinkCountButtonPressMedium-Long-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_blink_count_button_press_long.py --env-id BlinkCountButtonPressHard-Long-VLA-v0
"""

WAYPOINT_STRIDE = 2

VALID_LONG_ENV_IDS = {
    "BlinkCountButtonPressEasy-Long-VLA-v0",
    "BlinkCountButtonPressMedium-Long-VLA-v0",
    "BlinkCountButtonPressHard-Long-VLA-v0",
}


# CurriculumPhaseNoopActionWrapperPdJointPos is imported via
# `from mikasa_robo_suite.vla.utils.wrappers import *` above.


def _to_bool_scalar(x):
    if x is None:
        return False
    if torch.is_tensor(x):
        return bool(x.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    return bool(arr[0]) if arr.size > 0 else False


def _elapsed_from_info(info):
    if info is None or "elapsed_steps" not in info:
        return None
    x = info["elapsed_steps"]
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(x).reshape(-1)[0])


def _validate_flatten_obs(obs):
    if not isinstance(obs, dict):
        raise RuntimeError(f"Expected dict observation from FlattenRGBDObservationWrapper, got {type(obs).__name__}.")
    if "rgb" not in obs or "proprio" not in obs:
        raise RuntimeError(
            "Missing required keys in observation. "
            "Use StateOnlyTensorToDictWrapper + "
            "FlattenRGBDObservationWrapper(rgb=True, joints=True, state=False)."
        )


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def _build_ee_delta_action(prev_raw_pose: np.ndarray, next_raw_pose: np.ndarray, gripper_cmd: float) -> np.ndarray:
    prev_raw_pose = np.asarray(prev_raw_pose, dtype=np.float32)
    next_raw_pose = np.asarray(next_raw_pose, dtype=np.float32)

    dpos = next_raw_pose[:3] - prev_raw_pose[:3]
    q_prev = _normalize_quat_wxyz(prev_raw_pose[3:7])
    q_next = _normalize_quat_wxyz(next_raw_pose[3:7])
    q_delta = _normalize_quat_wxyz(qmult(q_next, qinverse(q_prev)))
    droll, dpitch, dyaw = quat2euler(q_delta)

    ee_delta = np.array(
        [dpos[0], dpos[1], dpos[2], droll, dpitch, dyaw, float(gripper_cmd)],
        dtype=np.float32,
    )
    return ee_delta


class EEDeltaActionLoggerWrapper(gym.Wrapper):
    """Logs 7D ee-delta actions (dx,dy,dz,droll,dpitch,dyaw,gripper) for each env.step."""

    def __init__(self, env):
        super().__init__(env)
        self.ee_delta_actions = []

    def reset(self, **kwargs):
        self.ee_delta_actions = []
        return self.env.reset(**kwargs)

    def step(self, action):
        prev_raw = self.unwrapped.agent.tcp.pose.raw_pose[0].detach().cpu().numpy().copy()

        obs, reward, terminated, truncated, info = super().step(action)

        next_raw = self.unwrapped.agent.tcp.pose.raw_pose[0].detach().cpu().numpy().copy()
        if torch.is_tensor(action):
            action_np = action.detach().cpu().numpy().reshape(-1)
        else:
            action_np = np.asarray(action).reshape(-1)
        gripper_cmd = float(action_np[-1]) if action_np.size > 0 else 0.0

        ee_action = _build_ee_delta_action(prev_raw, next_raw, gripper_cmd)
        self.ee_delta_actions.append(ee_action)
        return obs, reward, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default="BlinkCountButtonPressMedium-Long-VLA-v0",
        help="Registered BlinkCountButtonPress*-Long-VLA-v0 env id.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", type=int, default=0)
    parser.add_argument("--save-video", type=int, default=1)
    parser.add_argument(
        "--overlay-info",
        type=int,
        default=1,
        help="Whether to draw step/reward/reward_dict overlays on rendered video (0/1).",
    )
    parser.add_argument("--trajectory-dir", type=str, default=None)
    parser.add_argument("--trajectory-name", type=str, default="trajectory")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if env_id not in VALID_LONG_ENV_IDS:
        raise ValueError(f"Expected one of {sorted(VALID_LONG_ENV_IDS)}, got {env_id!r}.")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

    wrappers_list = [
        (CurriculumPhaseNoopActionWrapperPdJointPos, {}),
        (RenderPressProgressInfoWrapper, {}),
    ]
    if bool(args.overlay_info):
        wrappers_list.extend(
            [
                (RenderStepInfoWrapper, {}),
                (RenderRewardInfoWrapper, {}),
                (DebugRewardWrapper, {}),
            ]
        )

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
        max_episode_steps=max_env_steps,
    )

    for wrapper_cls, wrapper_kwargs in wrappers_list:
        env = wrapper_cls(env, **wrapper_kwargs)

    env = StateOnlyTensorToDictWrapper(env)
    env = FlattenRGBDObservationWrapper(
        env,
        rgb=True,
        depth=False,
        state=False,
        oracle=False,
        joints=True,
    )

    out_dir = args.trajectory_dir or f"./videos/{env_id}/oracle_motionplanning"
    env = RecordEpisode(
        env,
        output_dir=out_dir,
        save_trajectory=bool(args.save_trajectory),
        trajectory_name=args.trajectory_name,
        save_video=bool(args.save_video),
        info_on_video=False,
        max_steps_per_video=max_env_steps,
        video_fps=30,
        source_type="motionplanning",
        source_desc="oracle via PandaArmMotionPlanningSolver",
    )

    env = EEDeltaActionLoggerWrapper(env)

    env_u = env.unwrapped
    obs, info = env.reset(seed=args.seed)
    _validate_flatten_obs(obs)

    base_pose = env_u.agent.robot.pose
    assert base_pose is not None, "robot.pose is None"

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=base_pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
        joint_vel_limits=2.0,
        joint_acc_limits=2.0,
    )

    episode_done = {"value": False}

    planner.gripper_state = -1  # keep closed throughout
    planner.open_gripper = lambda *args, **kwargs: None

    def fast_follow_path(self, result, refine_steps: int = 0):  # noqa: ARG001
        if episode_done["value"]:
            return None, None, True, False, None

        n_step = result["position"].shape[0]
        if n_step <= 0:
            return None, None, None, None, None

        idxs = list(range(0, n_step, WAYPOINT_STRIDE))
        if idxs[-1] != n_step - 1:
            idxs.append(n_step - 1)

        obs = reward = terminated = truncated = info = None
        k = 0

        while k < len(idxs):
            cur_elapsed = int(np.asarray(self.env.unwrapped.elapsed_steps.detach().cpu()).reshape(-1)[0])
            if cur_elapsed >= max_env_steps - 1:
                episode_done["value"] = True
                break

            i = idxs[k]
            qpos = result["position"][i]

            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][i]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.elapsed_steps += 1

            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                break

            elapsed = _elapsed_from_info(info)
            if elapsed is not None and elapsed >= max_env_steps:
                episode_done["value"] = True
                break

            action_mask = None
            if info is not None:
                action_mask = info.get("action_mask", None)
            if action_mask is not None and (not _to_bool_scalar(action_mask)):
                continue

            k += 1

        return obs, reward, terminated, truncated, info

    planner.follow_path = types.MethodType(fast_follow_path, planner)

    def build_hold_action_pd_joint_pos() -> np.ndarray:
        qpos = env_u.agent.robot.get_qpos()[0].detach().cpu().numpy()
        qpos_arm = qpos[:-2]
        qpos_gripper = float(qpos[-2])

        mid = 0.5 * (
            CurriculumPhaseNoopActionWrapperPdJointPos.GRIPPER_HIGH
            + CurriculumPhaseNoopActionWrapperPdJointPos.GRIPPER_LOW
        )
        half = 0.5 * (
            CurriculumPhaseNoopActionWrapperPdJointPos.GRIPPER_HIGH
            - CurriculumPhaseNoopActionWrapperPdJointPos.GRIPPER_LOW
        )
        grip_norm = np.clip((qpos_gripper - mid) / half, -1.0, 1.0)
        return np.concatenate([qpos_arm, np.array([grip_norm], dtype=np.float32)], axis=0).astype(np.float32)

    def is_action_phase_open(step_info) -> bool:
        return isinstance(step_info, dict) and _to_bool_scalar(step_info.get("action_mask", False))

    # Long variants have cue phase; ensure we start pressing only after action phase starts.
    if not is_action_phase_open(info):
        for _ in range(max_env_steps):
            hold_action = build_hold_action_pd_joint_pos()
            obs, _, warm_terminated, warm_truncated, info = env.step(hold_action)
            if _to_bool_scalar(warm_terminated) or _to_bool_scalar(warm_truncated):
                episode_done["value"] = True
                break
            if is_action_phase_open(info):
                break

    tcp_raw = env_u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
    tcp_q = tcp_raw[3:]

    target_blinks = int(env_u.target_blinks[0].item())
    button_xy = env_u.button_xy[0].detach().cpu().numpy()
    button_top_z = float(env_u.button_top_z[0].item())
    confirm_button_xy = env_u.confirm_button_xy[0].detach().cpu().numpy()
    confirm_button_top_z = float(env_u.confirm_button_top_z[0].item())

    T = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
    EXTRA_PRESS_DEPTH = 0.1
    required_lift = float(env_u.REQUIRED_LIFT_HEIGHT)

    # Depth threshold where raw press event first triggers.
    press_threshold_z = button_top_z + float(env_u.BUTTON_PRESS_Z_MARGIN) - T
    z_press = press_threshold_z - EXTRA_PRESS_DEPTH

    # Confirmed press requires lift from press-start height by REQUIRED_LIFT_HEIGHT.
    # Set clear height relative to threshold (not relative to z_press) to avoid getting stuck at raw=1.
    z_unpress = press_threshold_z + required_lift + 0.03
    z_approach = z_unpress + 0.05
    z_confirm_press = confirm_button_top_z + float(env_u.BUTTON_PRESS_Z_MARGIN) - T - EXTRA_PRESS_DEPTH
    z_confirm_unpress = z_confirm_press + 0.08
    z_confirm_approach = z_confirm_unpress + 0.05
    z_confirm_lift = z_confirm_unpress

    def move_pose(p):
        if episode_done["value"]:
            return None, None, True, False, None
        return planner.move_to_pose_with_screw(p, dry_run=False, refine_steps=0)

    terminated = False
    truncated = False

    # 1) Approach above button
    target1 = sapien.Pose(p=[button_xy[0], button_xy[1], z_approach], q=tcp_q)
    _, _, terminated, truncated, _ = move_pose(target1)

    # 2) N presses: press -> clear
    for _ in range(target_blinks):
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            break

        target_press = sapien.Pose(p=[button_xy[0], button_xy[1], z_press], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_press)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            break

        target_clear = sapien.Pose(p=[button_xy[0], button_xy[1], z_unpress], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_clear)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            break

        # No extra lift waypoint here: z_unpress already satisfies lift confirmation.

    # 3) Submit answer with black button.
    if not (_to_bool_scalar(terminated) or _to_bool_scalar(truncated)):
        target_submit_approach = sapien.Pose(
            p=[confirm_button_xy[0], confirm_button_xy[1], z_confirm_approach],
            q=tcp_q,
        )
        _, _, terminated, truncated, _ = move_pose(target_submit_approach)

    if not (_to_bool_scalar(terminated) or _to_bool_scalar(truncated)):
        target_submit_press = sapien.Pose(
            p=[confirm_button_xy[0], confirm_button_xy[1], z_confirm_press],
            q=tcp_q,
        )
        _, _, terminated, truncated, _ = move_pose(target_submit_press)

    if not (_to_bool_scalar(terminated) or _to_bool_scalar(truncated)):
        target_submit_clear = sapien.Pose(
            p=[confirm_button_xy[0], confirm_button_xy[1], z_confirm_unpress],
            q=tcp_q,
        )
        _, _, terminated, truncated, _ = move_pose(target_submit_clear)

    if not (_to_bool_scalar(terminated) or _to_bool_scalar(truncated)):
        target_submit_lift = sapien.Pose(
            p=[confirm_button_xy[0], confirm_button_xy[1], z_confirm_lift],
            q=tcp_q,
        )
        _, _, terminated, truncated, _ = move_pose(target_submit_lift)

    info2 = env_u.evaluate()
    elapsed_steps = int(np.asarray(env_u.elapsed_steps.detach().cpu()).reshape(-1)[0])

    print("env_id:", env_id)
    print("max_episode_steps:", max_env_steps)
    print("elapsed_steps:", elapsed_steps)
    print("target_blinks:", target_blinks)
    print("raw_press_count:", int(info2["raw_press_count"][0].item()))
    print("press_count:", int(info2["press_count"][0].item()))
    print("submit_attempted:", bool(info2["submit_attempted"][0].item()))
    print("submit_success:", bool(info2["submit_success"][0].item()))
    print("success:", bool(info2["success"][0].item()))
    print("terminated:", _to_bool_scalar(terminated))
    print("truncated:", _to_bool_scalar(truncated))

    if bool(args.save_trajectory):
        ee_actions = np.asarray(env.ee_delta_actions, dtype=np.float32)
        ee_path = os.path.join(out_dir, f"{args.trajectory_name}_ee_delta_actions.npy")
        np.save(ee_path, ee_actions)
        print("Saved ee-delta actions:", ee_path, ee_actions.shape)

    env.close()

    mp4s = sorted(glob(f"{out_dir}/*.mp4"))
    print("Saved videos:", mp4s)
    if bool(args.save_video):
        assert len(mp4s) > 0, f"No mp4 found in {out_dir}"
        Video(mp4s[-1], embed=True, width=640)


if __name__ == "__main__":
    main()
