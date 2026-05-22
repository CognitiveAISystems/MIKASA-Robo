import argparse
import json
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
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_easy.py --env-id BatteriesCheckerEasy-15-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_easy.py --env-id BatteriesCheckerEasy-6-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_batteries_checker_easy.py --env-id BatteriesCheckerEasy-6-VLA-v0 --seed 123
"""

DEFAULT_ENV_ID = "BatteriesCheckerEasy-15-VLA-v0"
WAYPOINT_STRIDE = 4
MAX_RESET_ATTEMPTS = 32
SOCKET_X_REACHABLE_MAX = 0.22
PICK_PREOPEN_STATE = 0.10  # narrower than full-open to avoid touching neighbor batteries
TRANSPORT_Z_MARGIN = 0.16
TRANSPORT_Z_MAX = 0.24


class IgnoreSuccessTerminationWrapper(gym.Wrapper):
    """Keep episode alive after success so we can finish scripted cycle."""

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
    return int(arr[0]) if arr.size > 0 else False


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


def build_hold_action_pd_joint_pos(base_env):
    robot = base_env.agent.robot
    qpos = robot.get_qpos()  # shape: (n, 9)
    qpos_arm = qpos[..., :-2].detach().cpu().numpy()
    qpos_gripper = qpos[..., -2].detach().cpu().numpy()

    # Panda gripper normalization for pd_joint_pos
    gripper_low = -0.01
    gripper_high = 0.04
    mid = 0.5 * (gripper_high + gripper_low)
    half = 0.5 * (gripper_high - gripper_low)
    grip_norm = np.clip((qpos_gripper - mid) / half, -1.0, 1.0)

    action = np.concatenate([qpos_arm, grip_norm[..., None]], axis=1).astype(np.float32)
    return action[0]


def wait_until_action_phase(env, env_u, max_wait_steps=80):
    # Cue phase is disabled in BatteriesChecker now.
    del env, env_u, max_wait_steps
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Registered BatteriesCheckerEasy-* env id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for reset sampling (retry i uses seed+i).",
    )
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
    if not env_id.startswith("BatteriesCheckerEasy-"):
        raise ValueError(
            f"This oracle is Easy-only (reverted). Got env_id={env_id!r}. "
            "Use BatteriesCheckerEasy-* (e.g. BatteriesCheckerEasy-6-VLA-v0)."
        )
    env_spec = gym.spec(env_id)
    max_env_steps = env_spec.max_episode_steps

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

    env_u = env.unwrapped
    reset_success = False
    chosen_seed = None
    for reset_try in range(MAX_RESET_ATTEMPTS):
        trial_seed = args.seed + reset_try
        _, info = env.reset(seed=trial_seed)
        del info
        wait_until_action_phase(env, env_u)
        socket_x = float(env_u.socket_slot_pos[0, 0].item())
        if socket_x <= SOCKET_X_REACHABLE_MAX:
            reset_success = True
            chosen_seed = trial_seed
            break
        print(f"[reset {reset_try:02d}] socket_x={socket_x:.3f} is too far, retrying...")
    if not reset_success:
        raise RuntimeError(
            f"Could not sample reachable scene after {MAX_RESET_ATTEMPTS} resets "
            f"(required socket_x <= {SOCKET_X_REACHABLE_MAX})."
        )

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
    if chosen_seed is None:
        chosen_seed = args.seed
    obs, _ = env.reset(seed=chosen_seed)
    _validate_flatten_obs(obs)
    env_u = env.unwrapped

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

            elapsed = int(np.asarray(self.env.unwrapped.elapsed_steps.detach().cpu()).reshape(-1)[0])
            if elapsed >= max_env_steps - 1:
                episode_done["value"] = True
                break
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
    # Lower grasp height to avoid "passing above" small battery cylinders.
    z_grasp_slot = tray_top_z + battery_half_h * 0.20
    z_grasp_slot_refine = z_grasp_slot - 0.010
    z_insert_socket = float(socket_pos[2]) + 0.004

    t_press = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
    z_press = button_top_z + float(env_u.BUTTON_PRESS_Z_MARGIN) - t_press - 0.06
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

        raise RuntimeError(f"Planner failed to move to pose: {[x, y, z]}. {last_err}")

    def move_via_clearance(x, y, z_target, z_clear):
        """Collision-robust motion: up -> XY transfer -> down."""
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
            hold = build_hold_action_pd_joint_pos(env_u)
            _, _, terminated, truncated, _ = env.step(hold)
            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                return

    def set_gripper_state_safe(gripper_state: float, t: int = 2):
        """Drive gripper while respecting episode termination."""
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

    def get_stage():
        return int(env_u.action_stage[0].item())

    def get_counts():
        info_now = env_u.evaluate()
        checked = _to_int_scalar(info_now["checked_count"])
        found = _to_int_scalar(info_now["found_working_count"])
        return checked, found

    def wait_stage_confirm(max_steps: int = 2) -> bool:
        target_stage = int(env_u.STAGE_CONFIRM)
        for _ in range(max_steps):
            if get_stage() == target_stage:
                return True
            hold_steps(1)
            if episode_done["value"]:
                return False
        return get_stage() == target_stage

    def try_pick_from_slot(slot_idx: int, sx: float, sy: float) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        # No lateral search in crowded tray: descend only at target center.
        offsets = [(0.0, 0.0)]
        for dx, dy in offsets:
            tx = sx + dx
            ty = sy + dy
            move_via_clearance(tx, ty, z_slot_approach, z_transport)
            # Keep fingers relatively narrow while approaching dense battery rows.
            preopen_gripper_for_pick_safe(t=1)
            move_pose_xyz(tx, ty, z_grasp_slot)
            move_pose_xyz(tx, ty, z_grasp_slot_refine)
            close_gripper_safe(t=3)
            move_pose_xyz(tx, ty, z_slot_approach + 0.05)
            move_pose_xyz(tx, ty, z_transport)

            grasped = bool(env_u.agent.is_grasping(battery_actor)[0].item())
            battery_z = float(battery_actor.pose.p[0, 2].item())
            lifted = battery_z > (tray_top_z + 0.025)
            if grasped or lifted:
                return True

            open_gripper_safe(t=2)
        return False

    def try_insert_battery_into_checker(slot_idx: int) -> bool:
        battery_actor = env_u.batteries[slot_idx]
        xy_offsets = [(0.0, 0.0), (0.002, 0.0), (-0.002, 0.0), (0.0, 0.002), (0.0, -0.002)]
        z_offsets = [0.0, -0.003, -0.006]

        for dx, dy in xy_offsets:
            tx = float(socket_pos[0] + dx)
            ty = float(socket_pos[1] + dy)
            for dz in z_offsets:
                move_via_clearance(tx, ty, z_socket_approach, z_transport)
                move_pose_xyz(tx, ty, z_insert_socket + dz)
                open_gripper_safe(t=2)
                hold_steps(3)
                move_pose_xyz(tx, ty, z_socket_approach)

                if wait_stage_confirm(max_steps=4):
                    return True

                move_pose_xyz(tx, ty, z_insert_socket + dz + 0.002)
                close_gripper_safe(t=2)
                move_pose_xyz(tx, ty, z_socket_approach)
                if not bool(env_u.agent.is_grasping(battery_actor)[0].item()):
                    continue
        return False

    # Start with opened gripper.
    open_gripper_safe(t=2)

    num_batteries = int(env_u.active_battery_count[0].item())

    for idx in tqdm(range(num_batteries)):
        if episode_done["value"]:
            break

        slot = tray_slots[idx]
        _sx_slot, _sy_slot, _sz = float(slot[0]), float(slot[1]), float(slot[2])
        # Use actual battery pose for pick center; physics may shift it slightly.
        bat_pos = env_u.batteries[idx].pose.p[0].detach().cpu().numpy()
        sx_pick, sy_pick = float(bat_pos[0]), float(bat_pos[1])

        # 1) Pick battery from tray slot
        picked = try_pick_from_slot(idx, sx_pick, sy_pick)
        if not picked:
            print(f"[{idx:02d}] pick failed, skip slot")
            continue
        move_via_clearance(sx_pick, sy_pick, z_transport, z_transport)

        # 2) Insert into checker with short correction retries
        inserted = try_insert_battery_into_checker(idx)
        move_pose_xyz(socket_pos[0], socket_pos[1], z_transport)
        if not inserted:
            print(f"[{idx:02d}] insert failed, skip confirm")
            continue

        # 4) Confirm by button press
        close_gripper_safe(t=1)
        move_via_clearance(button_xy[0], button_xy[1], z_button_approach, z_transport)
        move_pose_xyz(button_xy[0], button_xy[1], z_press)
        hold_steps(3)
        move_pose_xyz(button_xy[0], button_xy[1], z_button_release)
        move_pose_xyz(button_xy[0], button_xy[1], z_transport)

        checked, found = get_counts()
        print(f"[{idx:02d}] checked={checked} found_working={found} stage={get_stage()}")

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

        meta_path = os.path.join(out_dir, f"{args.trajectory_name}_meta.json")
        meta = {
            "env_id": env_id,
            "requested_seed": int(args.seed),
            "actual_seed": int(chosen_seed if chosen_seed is not None else args.seed),
            "max_reset_attempts": int(MAX_RESET_ATTEMPTS),
            "socket_x_reachable_max": float(SOCKET_X_REACHABLE_MAX),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("Saved trajectory metadata:", meta_path, meta)

    env.close()

    mp4s = sorted(glob(f"{out_dir}/*.mp4"))
    print("Saved videos:", mp4s)
    if bool(args.save_video):
        assert len(mp4s) > 0, f"No mp4 found in {out_dir}"
        Video(mp4s[-1], embed=True, width=640)


if __name__ == "__main__":
    main()
