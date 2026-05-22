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
from tqdm import tqdm
from transforms3d.euler import quat2euler
from transforms3d.quaternions import axangle2quat, qinverse, qmult

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.memory_envs import *
from mikasa_robo_suite.vla.utils.wrappers import *

"""
Hard batteries checker oracle via motion planning.

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_hard.py --env-id BatteriesCheckerHard-3-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_hard.py --env-id BatteriesCheckerHard-15-VLA-v0
"""

DEFAULT_ENV_ID = "BatteriesCheckerHard-3-VLA-v0"
WAYPOINT_STRIDE = 4
CARRY_WAYPOINT_STRIDE = 2
GRASP_REINFORCE_CLOSE_STEPS = 2
GRASP_REINFORCE_SETTLE_STEPS = 1
PICK_PREOPEN_STATE = 0.10
SOCKET_PICK_PREOPEN_STATE = 0.22
RETURN_RELEASE_OPEN_STATE = 0.28
BUTTON_INITIAL_EXTRA_PRESS_DEPTH = 0.040
MAX_SOCKET_INSERT_RETRIES = 16
MAX_SOCKET_PICK_RETRIES = 12
MAX_SLOT_RETURN_RETRIES = 12
MAX_CONFIRM_PRESS_ATTEMPTS = 12
TRANSPORT_Z_MARGIN = 0.16
TRANSPORT_Z_MAX = 0.27


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
    robot = base_env.agent.robot
    qpos = robot.get_qpos()  # (n, 9) panda: 7 arm + 2 fingers
    qpos_arm = qpos[..., :-2].detach().cpu().numpy()
    if gripper_state_override is None:
        qpos_gripper = qpos[..., -2].detach().cpu().numpy()
        gripper_low = -0.01
        gripper_high = 0.04
        mid = 0.5 * (gripper_high + gripper_low)
        half = 0.5 * (gripper_high - gripper_low)
        grip_norm = np.clip((qpos_gripper - mid) / half, -1.0, 1.0)
    else:
        grip_norm = np.full((qpos_arm.shape[0],), float(np.clip(gripper_state_override, -1.0, 1.0)), dtype=np.float32)

    hold = np.concatenate([qpos_arm, grip_norm[..., None]], axis=1).astype(np.float32)
    return hold[0]


class IgnoreSuccessTerminationWrapper(gym.Wrapper):
    """Keep episode alive after success so we can finish the scripted cycle."""

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if isinstance(info, dict) and "success" in info:
            success = info["success"]
            if torch.is_tensor(terminated):
                terminated = terminated.to(dtype=torch.bool)
                if torch.is_tensor(success):
                    terminated = terminated & (~success.to(dtype=torch.bool))
                else:
                    terminated = terminated & (~torch.as_tensor(success, device=terminated.device).to(dtype=torch.bool))
            else:
                s = bool(success.any().item()) if torch.is_tensor(success) else bool(success)
                terminated = bool(terminated) and (not s)
        return obs, reward, terminated, truncated, info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Registered BatteriesCheckerHard-* env id.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", type=int, default=0)
    parser.add_argument("--save-video", type=int, default=1)
    parser.add_argument(
        "--overlay-info",
        type=int,
        default=1,
        help="Whether to draw step/reward overlays on rendered video (0/1).",
    )
    parser.add_argument("--trajectory-dir", type=str, default=None)
    parser.add_argument("--trajectory-name", type=str, default="trajectory")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id
    if not env_id.startswith("BatteriesCheckerHard-"):
        raise ValueError(
            f"This script supports only hard envs. Got env_id={env_id!r}. "
            "Use BatteriesCheckerHard-* (e.g. BatteriesCheckerHard-6-VLA-v0)."
        )

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

    wrappers_list = [
        (RenderWorkingBatteriesInfoWrapper, {}),
    ]
    if bool(args.overlay_info):
        wrappers_list.extend(
            [
                (RenderStepInfoWrapper, {}),
                (RenderRewardInfoWrapper, {}),
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

    env = IgnoreSuccessTerminationWrapper(env)
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

    # Required for RecordEpisode trajectory buffers.
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

    def fast_follow_path(self, result, refine_steps: int = 0):
        del refine_steps
        if episode_done["value"]:
            return None, None, True, False, None

        n_step = result["position"].shape[0]
        if n_step <= 0:
            return None, None, None, None, None

        stride = (
            CARRY_WAYPOINT_STRIDE
            if any(bool(env_u.agent.is_grasping(bat)[0].item()) for bat in env_u.batteries)
            else WAYPOINT_STRIDE
        )
        idxs = list(range(0, n_step, stride))
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

    tcp_raw = env_u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
    tcp_q_nominal = tcp_raw[3:]
    tcp_q_state = {"q": tcp_q_nominal.copy()}

    tray_slots = env_u.tray_slot_positions[0].detach().cpu().numpy()
    socket_pos = env_u.socket_slot_pos[0].detach().cpu().numpy()
    button_xy = env_u.button_xy[0].detach().cpu().numpy()
    button_top_z = float(env_u.button_top_z[0].item())

    battery_half_h = float(env_u.BATTERY_HALF_HEIGHT)
    tray_top_z = float(np.max(tray_slots[:, 2]))

    z_transport = min(
        max(tray_top_z, float(socket_pos[2]), button_top_z) + TRANSPORT_Z_MARGIN,
        TRANSPORT_Z_MAX,
    )
    z_slot_approach = tray_top_z + 0.15
    z_socket_approach = float(socket_pos[2]) + 0.12

    z_pick_slot = tray_top_z + battery_half_h * 0.20
    z_pick_slot_refine = z_pick_slot - 0.010

    z_insert_socket = float(socket_pos[2]) + 0.004
    # Grab the battery inside the socket — descend close to socket top level.
    float(socket_pos[2]) + battery_half_h * 0.1
    z_place_slot = tray_top_z + 0.014
    z_place_slot_refine = z_place_slot - 0.001

    press_depth = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
    z_button_press = button_top_z + float(env_u.BUTTON_PRESS_Z_MARGIN) - press_depth - BUTTON_INITIAL_EXTRA_PRESS_DEPTH
    z_button_release = button_top_z + 0.08
    z_button_approach = z_button_release + 0.08

    yaw_candidates = [0.0, np.pi / 2, -np.pi / 2, np.pi]
    q_candidates_nominal = []
    for yaw in yaw_candidates:
        dq = axangle2quat([0, 0, 1], yaw)
        q_candidates_nominal.append(qmult(dq, tcp_q_nominal))

    def move_pose_xyz(x, y, z):
        if episode_done["value"]:
            return None, None, True, False, None

        q_try_list = [tcp_q_state["q"]]
        for q_nom in q_candidates_nominal:
            if not np.allclose(q_nom, q_try_list[0]):
                q_try_list.append(q_nom)

        z_try_list = [float(z)]
        if z_try_list[0] > 0.30:
            z_try_list.extend([z_try_list[0] - 0.03, z_try_list[0] - 0.06])

        last_err = None
        for z_try in z_try_list:
            for q_try in q_try_list:
                pose = sapien.Pose(p=[float(x), float(y), float(z_try)], q=q_try)
                out = planner.move_to_pose_with_screw(pose, dry_run=False, refine_steps=0)
                if out == -1:
                    out = planner.move_to_pose_with_RRTConnect(pose, dry_run=False, refine_steps=0)
                if out != -1:
                    tcp_q_state["q"] = np.asarray(q_try, dtype=np.float64)
                    return out
                last_err = f"Planner failed for z={z_try:.3f}, q={np.asarray(q_try).round(3).tolist()}"

        print(f"[warn] planner failed to reach [{x:.3f}, {y:.3f}, {z:.3f}]: {last_err}")
        return None, None, None, None, None

    def move_via_clearance(x, y, z_target, z_clear):
        cur = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
        cx, cy, cz = float(cur[0]), float(cur[1]), float(cur[2])
        if cz < z_clear - 1e-4:
            move_pose_xyz(cx, cy, z_clear)
        move_pose_xyz(float(x), float(y), z_clear)
        if abs(float(z_target) - z_clear) > 1e-4:
            move_pose_xyz(float(x), float(y), float(z_target))

    def hold_steps(k=1):
        for _ in range(k):
            if episode_done["value"]:
                return
            hold = build_hold_action_pd_joint_pos(
                env_u,
                gripper_state_override=float(planner.gripper_state),
            )
            _, _, terminated, truncated, _ = env.step(hold)
            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                return

    def set_gripper_state_safe(gripper_state: float, t: int = 2):
        planner.gripper_state = gripper_state
        if episode_done["value"]:
            return
        for _ in range(t):
            if episode_done["value"]:
                return
            qpos = env_u.agent.robot.get_qpos()[0, :-2].detach().cpu().numpy()
            if planner.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, planner.gripper_state])
            else:
                action = np.hstack([qpos, qpos * 0, planner.gripper_state])
            _, _, terminated, truncated, _ = env.step(action)
            planner.elapsed_steps += 1
            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                return

    def open_gripper_safe(t: int = 2):
        set_gripper_state_safe(1, t=t)

    def close_gripper_safe(t: int = 2):
        set_gripper_state_safe(-1, t=t)

    def preopen_gripper_for_pick_safe(t: int = 1):
        set_gripper_state_safe(PICK_PREOPEN_STATE, t=t)

    def release_gripper_for_place_safe(t: int = 2):
        set_gripper_state_safe(RETURN_RELEASE_OPEN_STATE, t=t)

    def is_grasping_battery(slot_idx: int) -> bool:
        return bool(env_u.agent.is_grasping(env_u.batteries[slot_idx])[0].item())

    def reinforce_grasp(
        slot_idx: int, close_steps: int = GRASP_REINFORCE_CLOSE_STEPS, settle_steps: int = GRASP_REINFORCE_SETTLE_STEPS
    ) -> bool:
        close_gripper_safe(t=close_steps)
        hold_steps(settle_steps)
        return is_grasping_battery(slot_idx)

    def move_via_clearance_carry(slot_idx: int, x: float, y: float, z_target: float, z_clear: float) -> bool:
        if not is_grasping_battery(slot_idx):
            if not reinforce_grasp(slot_idx, close_steps=3, settle_steps=2):
                return False

        cur = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
        cx, cy, cz = float(cur[0]), float(cur[1]), float(cur[2])

        if cz < z_clear - 1e-4:
            move_pose_xyz(cx, cy, z_clear)
            if not is_grasping_battery(slot_idx):
                return False

        move_pose_xyz(float(x), float(y), z_clear)
        if not is_grasping_battery(slot_idx):
            return False

        if abs(float(z_target) - z_clear) > 1e-4:
            move_pose_xyz(float(x), float(y), float(z_target))
            if not is_grasping_battery(slot_idx):
                return False

        return True

    def get_stage() -> int:
        return int(env_u.action_stage[0].item())

    def get_active_idx() -> int:
        return int(env_u.active_battery_idx[0].item())

    def get_counts():
        info_now = env_u.evaluate()
        checked = _to_int_scalar(info_now["checked_count"])
        found = _to_int_scalar(info_now["found_working_count"])
        target = _to_int_scalar(info_now["target_working_count"])
        success = _to_bool_scalar(info_now["success"])
        return checked, found, target, success

    def wait_stage(target_stage: int, max_steps: int = 3) -> bool:
        for _ in range(max_steps):
            if get_stage() == target_stage:
                return True
            hold_steps(1)
            if episode_done["value"]:
                return False
        return get_stage() == target_stage

    def try_pick_from_slot(slot_idx: int, sx: float, sy: float) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        offsets = [(0.0, 0.0), (0.0015, 0.0), (-0.0015, 0.0), (0.0, 0.0015), (0.0, -0.0015)]

        for dx, dy in offsets:
            tx = sx + dx
            ty = sy + dy
            move_via_clearance(tx, ty, z_slot_approach, z_transport)
            preopen_gripper_for_pick_safe(t=1)
            move_pose_xyz(tx, ty, z_pick_slot)
            move_pose_xyz(tx, ty, z_pick_slot_refine)
            close_gripper_safe(t=3)
            move_pose_xyz(tx, ty, z_slot_approach + 0.05)
            move_pose_xyz(tx, ty, z_transport)

            grasped = bool(env_u.agent.is_grasping(battery_actor)[0].item())
            battery_z = float(battery_actor.pose.p[0, 2].item())
            lifted = battery_z > (tray_top_z + 0.020)
            if grasped or lifted:
                return True

            open_gripper_safe(t=2)

        return False

    def try_pick_from_pose(slot_idx: int, px: float, py: float, pz: float) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        offsets = [
            (0.0, 0.0),
            (0.0025, 0.0),
            (-0.0025, 0.0),
            (0.0, 0.0025),
            (0.0, -0.0025),
        ]
        z_approach = max(z_slot_approach, z_socket_approach - 0.02)
        z_pick_base = float(pz + battery_half_h * 0.15)
        z_pick_base = min(z_pick_base, z_approach - 0.01)
        z_pick_base = max(z_pick_base, tray_top_z + 0.004)

        for dx, dy in offsets:
            tx = px + dx
            ty = py + dy
            move_via_clearance(tx, ty, z_approach, z_transport)
            preopen_gripper_for_pick_safe(t=2)
            hold_steps(1)
            move_pose_xyz(tx, ty, z_pick_base)
            move_pose_xyz(tx, ty, z_pick_base - 0.006)
            close_gripper_safe(t=6)
            hold_steps(2)
            move_pose_xyz(tx, ty, z_approach + 0.04)
            move_pose_xyz(tx, ty, z_transport)

            grasped = bool(env_u.agent.is_grasping(battery_actor)[0].item())
            battery_z = float(battery_actor.pose.p[0, 2].item())
            lifted = battery_z > (tray_top_z + 0.020)
            if grasped or lifted:
                return True

            open_gripper_safe(t=2)

        return False

    def try_insert_into_socket(slot_idx: int) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        target_stage = int(env_u.STAGE_RETURN)
        xy_offsets = [
            (0.0, 0.0),
            (0.002, 0.0),
            (-0.002, 0.0),
            (0.0, 0.002),
            (0.0, -0.002),
        ]
        z_offsets = [0.0, -0.003, -0.006]

        attempt = 0
        while attempt < MAX_SOCKET_INSERT_RETRIES:
            for dx, dy in xy_offsets:
                tx = float(socket_pos[0] + dx)
                ty = float(socket_pos[1] + dy)
                for dz in z_offsets:
                    attempt += 1
                    if attempt > MAX_SOCKET_INSERT_RETRIES or episode_done["value"]:
                        return False

                    move_via_clearance(tx, ty, z_socket_approach, z_transport)
                    move_pose_xyz(tx, ty, z_insert_socket + dz)
                    open_gripper_safe(t=2)
                    hold_steps(2)
                    move_pose_xyz(tx, ty, z_socket_approach)

                    if wait_stage(target_stage, max_steps=4):
                        return True

                    move_pose_xyz(tx, ty, z_insert_socket + dz + 0.003)
                    close_gripper_safe(t=2)
                    move_pose_xyz(tx, ty, z_socket_approach)
                    if not bool(env_u.agent.is_grasping(battery_actor)[0].item()):
                        preopen_gripper_for_pick_safe(t=1)

        return False

    def try_extract_from_socket(slot_idx: int) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        z_offsets = [-0.006, -0.012, -0.018, -0.024]
        xy_offsets = [
            (0.0, 0.0),
            (0.0015, 0.0),
            (-0.0015, 0.0),
            (0.0, 0.0015),
            (0.0, -0.0015),
        ]

        def has_stable_grasp(k: int = 2) -> bool:
            for _ in range(k):
                if not is_grasping_battery(slot_idx):
                    return False
                hold_steps(1)
                if episode_done["value"]:
                    return False
            return is_grasping_battery(slot_idx)

        for attempt in range(MAX_SOCKET_PICK_RETRIES):
            if episode_done["value"]:
                return False

            battery_pos = battery_actor.pose.p[0].detach().cpu().numpy()
            base_pick_z = float(socket_pos[2]) - 0.002
            base_pick_z = min(base_pick_z, z_socket_approach - 0.015)
            base_pick_z = max(base_pick_z, float(socket_pos[2]) - 0.024)

            # Bias first attempts to exact socket center for a first-shot extraction.
            if attempt < 2:
                ox, oy = 0.0, 0.0
                tx = float(socket_pos[0])
                ty = float(socket_pos[1])
            else:
                ox, oy = xy_offsets[attempt % len(xy_offsets)]
                tx = float(battery_pos[0] + ox)
                ty = float(battery_pos[1] + oy)
            dz = z_offsets[attempt % len(z_offsets)]

            # 1) Position XY precisely above battery at approach height, then settle.
            move_pose_xyz(tx, ty, z_socket_approach)
            set_gripper_state_safe(SOCKET_PICK_PREOPEN_STATE, t=5)
            hold_steps(4)

            # 2) Descend and allow one micro-regrasp before escalating to next retry.
            local_z_offsets = [0.0, -0.006]
            grasped_this_attempt = False
            for local_dz in local_z_offsets:
                move_pose_xyz(tx, ty, base_pick_z + dz + local_dz)
                hold_steps(3)

                # 3) Close gripper firmly.
                close_gripper_safe(t=12)
                hold_steps(3)

                if not is_grasping_battery(slot_idx):
                    if not reinforce_grasp(slot_idx, close_steps=4, settle_steps=2):
                        set_gripper_state_safe(SOCKET_PICK_PREOPEN_STATE, t=2)
                        hold_steps(1)
                        continue

                if not has_stable_grasp(k=2):
                    set_gripper_state_safe(SOCKET_PICK_PREOPEN_STATE, t=2)
                    hold_steps(1)
                    continue

                # 4) Lift in two stages to reduce slip.
                z_mid = min(z_socket_approach + 0.04, z_transport)
                move_pose_xyz(tx, ty, z_mid)
                if not has_stable_grasp(k=1):
                    set_gripper_state_safe(SOCKET_PICK_PREOPEN_STATE, t=2)
                    hold_steps(1)
                    continue

                move_pose_xyz(tx, ty, z_transport)
                if has_stable_grasp(k=1):
                    grasped_this_attempt = True
                    break

                set_gripper_state_safe(SOCKET_PICK_PREOPEN_STATE, t=2)
                hold_steps(1)

            if grasped_this_attempt:
                return True

            open_gripper_safe(t=2)

        return False

    def try_return_to_home(slot_idx: int, home_x: float, home_y: float) -> bool:
        target_stage = int(env_u.STAGE_CONFIRM)
        offsets = [
            (0.0, 0.0),
            (0.001, 0.0),
            (-0.001, 0.0),
            (0.0, 0.001),
            (0.0, -0.001),
        ]
        z_offsets = [0.0, -0.0015]

        attempt = 0
        while attempt < MAX_SLOT_RETURN_RETRIES:
            for dx, dy in offsets:
                tx = home_x + dx
                ty = home_y + dy
                for dz in z_offsets:
                    attempt += 1
                    if attempt > MAX_SLOT_RETURN_RETRIES or episode_done["value"]:
                        return False

                    if not move_via_clearance_carry(slot_idx, tx, ty, z_slot_approach, z_transport):
                        continue
                    if not reinforce_grasp(slot_idx, close_steps=1, settle_steps=1):
                        continue

                    move_pose_xyz(tx, ty, z_place_slot + dz)
                    if not is_grasping_battery(slot_idx):
                        continue
                    move_pose_xyz(tx, ty, z_place_slot_refine + dz)
                    if not is_grasping_battery(slot_idx):
                        continue
                    release_gripper_for_place_safe(t=2)
                    hold_steps(2)
                    move_pose_xyz(tx, ty, z_slot_approach + 0.06)
                    move_pose_xyz(tx, ty, z_transport)

                    if wait_stage(target_stage, max_steps=3):
                        return True

                    # Re-grasp the same battery from where it landed and retry.
                    cur_pos = env_u.batteries[slot_idx].pose.p[0].detach().cpu().numpy()
                    picked_back = try_pick_from_slot(
                        slot_idx,
                        float(cur_pos[0]),
                        float(cur_pos[1]),
                    )
                    if not picked_back:
                        return False

        return False

    def get_button_press_metrics():
        tcp_pos = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
        tcp_xy = tcp_pos[:2]
        tcp_z = float(tcp_pos[2])
        xy_dist = float(np.linalg.norm(tcp_xy - button_xy))
        raw_depth = float(button_top_z + float(env_u.BUTTON_PRESS_Z_MARGIN) - tcp_z)
        depth = float(np.clip(raw_depth, 0.0, float(env_u.BUTTON_CAP_TRAVEL)))
        threshold = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
        return xy_dist, depth, threshold

    def press_confirm_button() -> bool:
        before_checked, before_found, _, _ = get_counts()
        close_gripper_safe(t=1)

        xy_offsets = [
            (0.0, 0.0),
            (0.010, 0.0),
            (-0.010, 0.0),
            (0.0, 0.010),
            (0.0, -0.010),
            (0.007, 0.007),
            (-0.007, 0.007),
            (0.007, -0.007),
            (-0.007, -0.007),
        ]
        z_offsets = [0.0, -0.004, -0.008]

        for attempt in range(MAX_CONFIRM_PRESS_ATTEMPTS):
            if episode_done["value"]:
                return False
            dx, dy = xy_offsets[attempt % len(xy_offsets)]
            dz = z_offsets[(attempt // len(xy_offsets)) % len(z_offsets)]
            tx = float(button_xy[0] + dx)
            ty = float(button_xy[1] + dy)
            press_z = float(z_button_press + dz)

            move_via_clearance(tx, ty, z_button_approach, z_transport)
            hold_steps(3 if attempt == 0 else 2)

            move_pose_xyz(tx, ty, press_z)
            hold_steps(3 if attempt == 0 else 2)

            if get_stage() == int(env_u.STAGE_INSERT):
                move_pose_xyz(tx, ty, z_button_release)
                move_pose_xyz(tx, ty, z_transport)
                return True

            after_checked, after_found, _, _ = get_counts()
            if after_checked > before_checked or after_found > before_found:
                move_pose_xyz(tx, ty, z_button_release)
                move_pose_xyz(tx, ty, z_transport)
                return True

            xy_dist, depth, threshold = get_button_press_metrics()

            move_pose_xyz(tx, ty, z_button_release)
            hold_steps(2)
            move_pose_xyz(tx, ty, z_transport)

            if get_stage() == int(env_u.STAGE_INSERT):
                return True

            if attempt == MAX_CONFIRM_PRESS_ATTEMPTS - 1:
                print(
                    f"[confirm] failed: stage={get_stage()} checked={after_checked} "
                    f"found={after_found} xy_dist={xy_dist:.4f} "
                    f"depth={depth:.4f} thr={threshold:.4f}"
                )

        return False

    def recover_to_insert_stage() -> bool:
        stage = get_stage()
        if stage == int(env_u.STAGE_INSERT):
            return True

        if stage == int(env_u.STAGE_CONFIRM):
            if press_confirm_button():
                return True
            hold_steps(2)
            return press_confirm_button()

        if stage == int(env_u.STAGE_RETURN):
            active_idx = get_active_idx()
            if active_idx < 0:
                return False

            extracted = try_extract_from_socket(active_idx)
            if not extracted:
                cur_pos = env_u.batteries[active_idx].pose.p[0].detach().cpu().numpy()
                extracted = try_pick_from_slot(active_idx, float(cur_pos[0]), float(cur_pos[1]))
            if not extracted:
                cur_pos = env_u.batteries[active_idx].pose.p[0].detach().cpu().numpy()
                extracted = try_pick_from_pose(
                    active_idx,
                    float(cur_pos[0]),
                    float(cur_pos[1]),
                    float(cur_pos[2]),
                )
            if not extracted:
                return False

            home = tray_slots[active_idx]
            returned = try_return_to_home(active_idx, float(home[0]), float(home[1]))
            if not returned:
                return False
            return press_confirm_button()

        return False

    open_gripper_safe(t=2)

    num_batteries = int(env_u.active_battery_count[0].item())

    for idx in tqdm(range(num_batteries)):
        if episode_done["value"]:
            break

        if bool(env_u.checked_mask[0, idx].item()):
            continue

        if get_stage() != int(env_u.STAGE_INSERT):
            recovered = recover_to_insert_stage()
            if not recovered:
                print(f"[{idx:02d}] recovery failed, stage={get_stage()} active_idx={get_active_idx()}")
                continue

        bat_pos = env_u.batteries[idx].pose.p[0].detach().cpu().numpy()
        picked = try_pick_from_slot(idx, float(bat_pos[0]), float(bat_pos[1]))
        if not picked:
            print(f"[{idx:02d}] pick failed")
            continue

        inserted = try_insert_into_socket(idx)
        move_pose_xyz(float(socket_pos[0]), float(socket_pos[1]), z_transport)
        if not inserted:
            print(f"[{idx:02d}] insert failed")
            continue

        active_idx = get_active_idx()
        if active_idx < 0:
            active_idx = idx

        extracted = try_extract_from_socket(active_idx)
        if not extracted:
            print(f"[{idx:02d}] extract from socket failed for active={active_idx}")
            continue

        home = tray_slots[active_idx]
        returned = try_return_to_home(active_idx, float(home[0]), float(home[1]))
        if not returned:
            print(f"[{idx:02d}] return to slot failed for active={active_idx}")
            continue

        confirmed = press_confirm_button()
        checked, found, target, success = get_counts()
        print(
            f"[{idx:02d}] active={active_idx} confirmed={confirmed} "
            f"checked={checked} found={found}/{target} stage={get_stage()} success={success}"
        )

    info_final = env_u.evaluate()
    print("checked_count:", int(info_final["checked_count"][0].item()))
    print("found_working_count:", int(info_final["found_working_count"][0].item()))
    print("target_working_count:", int(info_final["target_working_count"][0].item()))
    print("success:", bool(info_final["success"][0].item()))
    print("elapsed_steps:", int(env_u.elapsed_steps[0].item()))

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
