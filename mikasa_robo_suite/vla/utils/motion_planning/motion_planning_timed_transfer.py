"""Oracle motion planning for TimedTransfer VLA tasks.

Strategy:
1. Pick up the blue cube and move above the red disc as fast as possible.
2. Hold the cube above the red disc until the placement window approaches.
3. Lower the cube onto the red disc and release at the right moment.

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferEasy-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferMedium-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferHard-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferEasy-Long-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferMedium-Long-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_timed_transfer.py --env-id TimedTransferHard-Long-VLA-v0
"""

import argparse
import os
import sys
import types
import warnings
from glob import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))


def _configure_runtime_warnings_and_env():
    warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="Failed to find Vulkan ICD file.*", category=UserWarning)
    warnings.filterwarnings("ignore", message="The pynvml package is deprecated.*", category=FutureWarning)

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

WAYPOINT_STRIDE = 2

VALID_ENV_PREFIXES = ("TimedTransfer",)


def _to_bool_scalar(x):
    if x is None:
        return False
    if torch.is_tensor(x):
        return bool(x.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    return bool(arr[0]) if arr.size > 0 else False


def _to_int_scalar(x):
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    return int(arr[0]) if arr.size > 0 else 0


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


def _normalize_move_result(result):
    if isinstance(result, tuple):
        if len(result) == 5:
            return result
        if len(result) == 4:
            obs, reward, terminated, truncated = result
            return obs, reward, terminated, truncated, None
    return None, None, True, False, {"planner_status": result}


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


def build_hold_action_pd_joint_pos(base_env, gripper_state_override: float | None = None):
    """Build a hold-still action for pd_joint_pos control mode."""
    robot = base_env.agent.robot
    qpos = robot.get_qpos()
    qpos_arm = qpos[..., :-2].detach().cpu().numpy()

    if gripper_state_override is None:
        qpos_gripper = qpos[..., -2].detach().cpu().numpy()
        gripper_low = -0.01
        gripper_high = 0.04
        mid = 0.5 * (gripper_high + gripper_low)
        half = 0.5 * (gripper_high - gripper_low)
        grip_norm = np.clip((qpos_gripper - mid) / half, -1.0, 1.0)
    else:
        grip_norm = np.full(
            (qpos_arm.shape[0],),
            float(np.clip(gripper_state_override, -1.0, 1.0)),
            dtype=np.float32,
        )

    action = np.concatenate([qpos_arm, grip_norm[..., None]], axis=1).astype(np.float32)
    return action[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default="TimedTransferHard-VLA-v0",
        help="Registered TimedTransfer-* env id.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", type=int, default=0)
    parser.add_argument("--save-video", type=int, default=1)
    parser.add_argument(
        "--overlay-info",
        type=int,
        default=1,
        help="Whether to draw step/reward/timing overlays on rendered video (0/1).",
    )
    parser.add_argument("--trajectory-dir", type=str, default=None)
    parser.add_argument("--trajectory-name", type=str, default="trajectory")
    parser.add_argument(
        "--gripper-close-steps",
        type=int,
        default=5,
        help="Hold steps after closing gripper to secure the grasp.",
    )
    parser.add_argument(
        "--gripper-open-steps",
        type=int,
        default=5,
        help="Hold steps after opening gripper to release the cube.",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=5,
        help="Hold steps after release for the cube to settle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if not any(env_id.startswith(p) for p in VALID_ENV_PREFIXES):
        raise ValueError(f"Expected a TimedTransfer-* env, got {env_id!r}.")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

    wrappers_list = [
        (RenderTimedTransferInfoWrapper, {}),
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
    obs, _ = env.reset(seed=args.seed)
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

        for i in idxs:
            cur_elapsed = _to_int_scalar(env_u.elapsed_steps)
            if cur_elapsed >= max_env_steps - 1:
                episode_done["value"] = True
                break

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

        return obs, reward, terminated, truncated, info

    planner.follow_path = types.MethodType(fast_follow_path, planner)

    # ---------------------------------------------------------------
    # Read scene layout from the environment
    # ---------------------------------------------------------------
    tcp_raw = env_u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
    tcp_q = tcp_raw[3:]  # top-down grasp orientation

    cube_pos = env_u.blue_cube.pose.p[0].detach().cpu().numpy()
    env_u.green_disc_center[0].detach().cpu().numpy()
    red_disc_pos = env_u.red_disc_center[0].detach().cpu().numpy()

    disc_hh = float(env_u.DISC_HALF_HEIGHT)
    cube_hs = float(env_u.CUBE_HALF_SIZE)

    cube_x, cube_y, cube_z = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])
    red_x, red_y, red_z = float(red_disc_pos[0]), float(red_disc_pos[1]), float(red_disc_pos[2])

    # Z heights for the manipulation sequence
    z_approach = cube_z + 0.12  # above cube for approach
    z_grasp = cube_z  # at cube center for grasping
    z_lift = 0.20  # well above table
    z_place = red_z + disc_hh + cube_hs + 0.003  # cube resting on red disc

    # Timing
    signal_step = _to_int_scalar(env_u.signal_step[0])
    window_start = _to_int_scalar(env_u.window_start[0])
    window_end = _to_int_scalar(env_u.window_end[0])
    delay = int(env_u.DELAY_STEPS)

    print("=== TimedTransfer Oracle MP ===")
    print(f"env_id: {env_id}")
    print(f"delay: {delay} steps")
    print(f"signal_step: {signal_step}")
    print(f"window: [{window_start}, {window_end}] (size={window_end - window_start + 1})")
    print(f"cube: [{cube_x:.3f}, {cube_y:.3f}, {cube_z:.3f}]")
    print(f"red_disc: [{red_x:.3f}, {red_y:.3f}, {red_z:.3f}]")
    print()

    # ---------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------
    def move_pose(p):
        """Move TCP to a target pose using screw planner."""
        if episode_done["value"]:
            return None, None, True, False, None
        result = planner.move_to_pose_with_screw(p, dry_run=False, refine_steps=0)
        return _normalize_move_result(result)

    def hold_steps(k: int = 1):
        """Hold current position for k steps with current gripper state."""
        obs = reward = terminated = truncated = info = None
        for _ in range(k):
            if episode_done["value"]:
                break
            hold = build_hold_action_pd_joint_pos(
                env_u,
                gripper_state_override=float(planner.gripper_state),
            )
            obs, reward, terminated, truncated, info = env.step(hold)
            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                break
            elapsed = _elapsed_from_info(info)
            if elapsed is not None and elapsed >= max_env_steps:
                episode_done["value"] = True
                break
        return obs, reward, terminated, truncated, info

    def get_elapsed():
        return _to_int_scalar(env_u.elapsed_steps)

    # ---------------------------------------------------------------
    # Phase 1: Pick up the cube from the green disc
    # ---------------------------------------------------------------
    planner.gripper_state = 1  # open gripper

    # Approach above cube
    target_approach = sapien.Pose(p=[cube_x, cube_y, z_approach], q=tcp_q)
    _, _, terminated, truncated, _ = move_pose(target_approach)

    # Lower to grasp position
    if not episode_done["value"]:
        target_grasp = sapien.Pose(p=[cube_x, cube_y, z_grasp], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_grasp)

    # Close gripper to grasp the cube
    if not episode_done["value"]:
        planner.gripper_state = -1  # close
        hold_steps(args.gripper_close_steps)

    # Lift the cube
    if not episode_done["value"]:
        target_lift = sapien.Pose(p=[cube_x, cube_y, z_lift], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_lift)

    # ---------------------------------------------------------------
    # Phase 2: Move above the red disc and hold until window
    # ---------------------------------------------------------------
    if not episode_done["value"]:
        target_above_red = sapien.Pose(p=[red_x, red_y, z_lift], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_above_red)

    # Wait above the red disc until shortly before the placement window opens.
    # A small lead makes the replayed pd_ee_delta_pose trajectory land inside
    # the same timing window more reliably for short-horizon variants.
    if not episode_done["value"]:
        elapsed = get_elapsed()
        placement_lead_steps = 3
        remaining = max(0, window_start - elapsed - placement_lead_steps)
        if remaining > 0:
            print(
                f"Holding above red disc for {remaining} steps "
                f"(elapsed={elapsed}, window_start={window_start}, lead={placement_lead_steps})"
            )
            hold_steps(remaining)

    # ---------------------------------------------------------------
    # Phase 3: Place the cube on the red disc (during the window)
    # ---------------------------------------------------------------
    if not episode_done["value"]:
        target_place = sapien.Pose(p=[red_x, red_y, z_place], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_place)

    # Open gripper to release
    if not episode_done["value"]:
        planner.gripper_state = 1  # open
        hold_steps(args.gripper_open_steps)

    # Settle: let the cube rest on the disc
    if not episode_done["value"]:
        hold_steps(args.settle_steps)

    # Retreat upward
    if not episode_done["value"]:
        target_retreat = sapien.Pose(p=[red_x, red_y, z_lift], q=tcp_q)
        _, _, terminated, truncated, _ = move_pose(target_retreat)

    # ---------------------------------------------------------------
    # Report results
    # ---------------------------------------------------------------
    info_final = env_u.evaluate()
    elapsed_steps = get_elapsed()

    success = _to_bool_scalar(info_final["success"])
    failed = _to_bool_scalar(info_final["failed"])
    too_early = _to_bool_scalar(info_final["too_early"])
    cube_on_red = _to_bool_scalar(info_final["cube_on_red"])
    cube_to_red_dist = float(info_final["cube_to_red_dist"][0].item())

    print()
    print("=== Results ===")
    print(f"env_id: {env_id}")
    print(f"delay: {delay}")
    print(f"max_episode_steps: {max_env_steps}")
    print(f"elapsed_steps: {elapsed_steps}")
    print(f"window: [{window_start}, {window_end}]")
    print(f"cube_to_red_dist: {cube_to_red_dist:.5f}")
    print(f"cube_on_red: {cube_on_red}")
    print(f"too_early: {too_early}")
    print(f"success: {success}")
    print(f"failed: {failed}")
    print(f"terminated: {_to_bool_scalar(terminated)}")
    print(f"truncated: {_to_bool_scalar(truncated)}")

    if bool(args.save_trajectory):
        ee_actions = np.asarray(env.ee_delta_actions, dtype=np.float32)
        ee_path = os.path.join(out_dir, f"{args.trajectory_name}_ee_delta_actions.npy")
        np.save(ee_path, ee_actions)
        print(f"Saved ee-delta actions: {ee_path} {ee_actions.shape}")

    env.close()

    mp4s = sorted(glob(f"{out_dir}/*.mp4"))
    print(f"Saved videos: {mp4s}")
    if bool(args.save_video):
        assert len(mp4s) > 0, f"No mp4 found in {out_dir}"
        Video(mp4s[-1], embed=True, width=640)


if __name__ == "__main__":
    main()
