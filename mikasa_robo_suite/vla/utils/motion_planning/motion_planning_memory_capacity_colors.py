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
from transforms3d.quaternions import axangle2quat, qinverse, qmult

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.memory_envs import *
from mikasa_robo_suite.vla.utils.wrappers import *

"""
Unified oracle motion planning for memory-capacity color tasks.

Supported families:
- BunchOfColors{3,5,7}[-Long]-VLA-v0
- SeqOfColors{3,5,7}[-Long]-VLA-v0
- ChainOfColors{3,5,7}[-Long]-VLA-v0

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_memory_capacity_colors.py --env-id BunchOfColors5-Long-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_memory_capacity_colors.py --env-id SeqOfColors7-VLA-v0 --seed 123
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_memory_capacity_colors.py --env-id ChainOfColors3-Long-VLA-v0 --seed 7
"""

DEFAULT_ENV_ID = "ChainOfColors5-Long-VLA-v0"
VALID_ENV_IDS = {
    "BunchOfColors3-Long-VLA-v0",
    "BunchOfColors5-Long-VLA-v0",
    "BunchOfColors7-Long-VLA-v0",
    "SeqOfColors3-Long-VLA-v0",
    "SeqOfColors5-Long-VLA-v0",
    "SeqOfColors7-Long-VLA-v0",
    "ChainOfColors3-Long-VLA-v0",
    "ChainOfColors5-Long-VLA-v0",
    "ChainOfColors7-Long-VLA-v0",
    "BunchOfColors3-VLA-v0",
    "BunchOfColors5-VLA-v0",
    "BunchOfColors7-VLA-v0",
    "SeqOfColors3-VLA-v0",
    "SeqOfColors5-VLA-v0",
    "SeqOfColors7-VLA-v0",
    "ChainOfColors3-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors7-VLA-v0",
}


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


def _resolve_family(env_id: str):
    if env_id.startswith("BunchOfColors"):
        return "bunch"
    if env_id.startswith("SeqOfColors"):
        return "seq"
    if env_id.startswith("ChainOfColors"):
        return "chain"
    raise ValueError(f"Unsupported env family for env_id={env_id!r}")


def _target_idx_from_next_mask(env_u):
    if not hasattr(env_u, "next_target_mask"):
        return None
    x = env_u.next_target_mask[0]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).astype(bool)
    idxs = np.flatnonzero(x)
    if len(idxs) == 0:
        return None
    return int(idxs[0])


def _button_press_pending(env_u) -> bool:
    if not hasattr(env_u, "button_pressable") or not hasattr(env_u, "button_pressed"):
        return False

    pressable = env_u.button_pressable[0]
    pressed = env_u.button_pressed[0]
    return _to_bool_scalar(pressable) and (not _to_bool_scalar(pressed))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        default=DEFAULT_ENV_ID,
        help="Registered Bunch/Seq/Chain VLA env id.",
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

    parser.add_argument("--waypoint-stride", type=int, default=2)
    parser.add_argument(
        "--max-waypoints-per-plan",
        type=int,
        default=0,
        help="Cap executed waypoints per planned path (<=0 disables cap).",
    )
    parser.add_argument("--approach-z-offset", type=float, default=0.08)
    parser.add_argument(
        "--touch-z-extra",
        type=float,
        default=0.005,
        help="Extra TCP height above cube top-touch center.",
    )
    parser.add_argument(
        "--travel-clearance-z",
        type=float,
        default=0.18,
        help="Global safe Z for XY transfers.",
    )
    parser.add_argument("--touch-settle-steps", type=int, default=1)
    parser.add_argument("--post-touch-retreat", type=int, default=0)
    parser.add_argument("--manip-phase-buffer-steps", type=int, default=1)
    parser.add_argument(
        "--max-target-attempts",
        type=int,
        default=48,
        help="Upper bound on target-touch iterations.",
    )
    parser.add_argument(
        "--short-episode-steps",
        type=int,
        default=200,
        help="If >0, override max_episode_steps for non-Long envs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if env_id not in VALID_ENV_IDS:
        allowed = ", ".join(sorted(VALID_ENV_IDS))
        raise ValueError(f"Unsupported env_id={env_id!r}. Allowed: {allowed}")
    if args.waypoint_stride <= 0:
        raise ValueError(f"--waypoint-stride must be > 0, got {args.waypoint_stride}")
    if args.max_waypoints_per_plan == 1:
        raise ValueError("--max-waypoints-per-plan must be <=0 or >=2")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

    if "-Long-" not in env_id and int(args.short_episode_steps) > 0:
        max_env_steps = max(max_env_steps, int(args.short_episode_steps))

    wrappers_list = [(MemoryCapacityInfoWrapper, {})]
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
        joint_vel_limits=2.6,
        joint_acc_limits=2.6,
    )
    planner.gripper_state = 1.0

    episode_done = {"value": False}

    def fast_follow_path(self, result, refine_steps: int = 0):  # noqa: ARG001
        if episode_done["value"]:
            return None, None, True, False, None

        n_step = result["position"].shape[0]
        if n_step <= 0:
            return None, None, None, None, None

        if int(args.max_waypoints_per_plan) > 0:
            n_waypoints = min(max(int(args.max_waypoints_per_plan), 2), n_step)
            idxs = np.linspace(0, n_step - 1, num=n_waypoints, dtype=np.int64).tolist()
        else:
            idxs = list(range(0, n_step, int(args.waypoint_stride)))
            if idxs[-1] != n_step - 1:
                idxs.append(n_step - 1)

        obs_ = reward_ = terminated_ = truncated_ = info_ = None
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

            obs_, reward_, terminated_, truncated_, info_ = self.env.step(action)
            self.elapsed_steps += 1

            if _to_bool_scalar(terminated_) or _to_bool_scalar(truncated_):
                episode_done["value"] = True
                break

            elapsed = _elapsed_from_info(info_)
            if elapsed is not None and elapsed >= max_env_steps:
                episode_done["value"] = True
                break

        return obs_, reward_, terminated_, truncated_, info_

    planner.follow_path = types.MethodType(fast_follow_path, planner)

    tcp_raw = env_u.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
    tcp_q_nominal = tcp_raw[3:]
    tcp_q_state = {"q": tcp_q_nominal.copy()}

    yaw_candidates = [0.0, np.pi / 2, -np.pi / 2, np.pi]
    q_candidates_nominal = []
    for yaw in yaw_candidates:
        dq = axangle2quat([0, 0, 1], yaw)
        q_candidates_nominal.append(qmult(dq, tcp_q_nominal))

    def move_pose_xyz(x: float, y: float, z: float):
        if episode_done["value"]:
            return None, None, True, False, None

        q_try_list = [tcp_q_state["q"]]
        for q_nom in q_candidates_nominal:
            if not np.allclose(q_nom, q_try_list[0]):
                q_try_list.append(q_nom)

        last_err = None
        for q_try in q_try_list:
            pose = sapien.Pose(p=[float(x), float(y), float(z)], q=q_try)
            out = planner.move_to_pose_with_screw(pose, dry_run=False, refine_steps=0)
            if out == -1:
                out = planner.move_to_pose_with_RRTConnect(
                    pose,
                    dry_run=False,
                    refine_steps=0,
                )
            if out != -1:
                tcp_q_state["q"] = np.asarray(q_try, dtype=np.float64)
                return out
            last_err = f"planner failed for q={np.asarray(q_try).round(3).tolist()}"

        print(f"[warn] failed move to [{x:.3f}, {y:.3f}, {z:.3f}] ({last_err})")
        return None, None, None, None, None

    def move_via_clearance(x: float, y: float, z_target: float, z_clear: float):
        cur = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
        cx, cy, cz = float(cur[0]), float(cur[1]), float(cur[2])
        out = (None, None, None, None, None)

        if cz < z_clear - 1e-4:
            out = move_pose_xyz(cx, cy, z_clear)
            if _to_bool_scalar(out[2]) or _to_bool_scalar(out[3]):
                return out

        out = move_pose_xyz(float(x), float(y), z_clear)
        if _to_bool_scalar(out[2]) or _to_bool_scalar(out[3]):
            return out

        if abs(float(z_target) - z_clear) > 1e-4:
            out = move_pose_xyz(float(x), float(y), float(z_target))

        return out

    def hold_steps(k: int = 1):
        obs_ = reward_ = terminated_ = truncated_ = info_ = None
        for _ in range(k):
            if episode_done["value"]:
                break
            hold = build_hold_action_pd_joint_pos(
                env_u,
                gripper_state_override=float(planner.gripper_state),
            )
            obs_, reward_, terminated_, truncated_, info_ = env.step(hold)
            if _to_bool_scalar(terminated_) or _to_bool_scalar(truncated_):
                episode_done["value"] = True
                break
            elapsed = _elapsed_from_info(info_)
            if elapsed is not None and elapsed >= max_env_steps:
                episode_done["value"] = True
                break
        return obs_, reward_, terminated_, truncated_, info_

    cue_steps = _to_int_scalar(env_u.cue_steps_per_env[0])
    empty_steps = _to_int_scalar(env_u.empty_steps_per_env[0])
    manip_start_step = cue_steps + empty_steps

    while (_to_int_scalar(env_u.elapsed_steps) < manip_start_step + int(args.manip_phase_buffer_steps)) and (
        not episode_done["value"]
    ):
        hold_steps(1)

    family = _resolve_family(env_id)
    attempts = 0
    terminated = truncated = False

    while attempts < int(args.max_target_attempts):
        if episode_done["value"]:
            break

        info_now = env_u.evaluate()
        if _to_bool_scalar(info_now["success"]):
            break

        target_idx = _target_idx_from_next_mask(env_u)
        if target_idx is None:
            if _button_press_pending(env_u):
                prev_gripper_state = float(planner.gripper_state)
                planner.gripper_state = -1.0
                try:
                    bx = float(env_u.button_xy[0, 0].detach().cpu().item())
                    by = float(env_u.button_xy[0, 1].detach().cpu().item())
                    btop = float(env_u.button_top_z[0].detach().cpu().item())

                    press_threshold = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
                    extra_press_depth = 0.1
                    z_press = btop + float(env_u.BUTTON_PRESS_Z_MARGIN) - press_threshold - extra_press_depth
                    z_unpress = z_press + 0.05
                    z_approach = z_unpress + 0.10
                    z_lift = btop + float(getattr(env_u, "REQUIRED_LIFT_HEIGHT", 0.1)) + 0.02

                    _, _, terminated, truncated, _ = move_pose_xyz(bx, by, z_approach)
                    if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                        episode_done["value"] = True
                        break

                    _, _, terminated, truncated, _ = move_pose_xyz(bx, by, z_press)
                    if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                        episode_done["value"] = True
                        break

                    _, _, terminated, truncated, _ = move_pose_xyz(bx, by, z_unpress)
                    if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                        episode_done["value"] = True
                        break

                    _, _, terminated, truncated, _ = move_pose_xyz(bx, by, z_lift)
                    if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                        episode_done["value"] = True
                        break

                    attempts += 1
                    continue
                finally:
                    planner.gripper_state = prev_gripper_state

            hold_steps(1)
            attempts += 1
            continue

        if target_idx not in env_u.cubes:
            raise RuntimeError(f"Target idx {target_idx} not found in cubes map.")

        cube_pos = env_u.cubes[target_idx].pose.p[0].detach().cpu().numpy()
        tx, ty, tz = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])

        z_touch = tz + float(env_u.CUBE_HALFSIZE) + float(args.touch_z_extra)
        z_approach = z_touch + float(args.approach_z_offset)
        z_clear = max(float(args.travel_clearance_z), z_approach)

        _, _, terminated, truncated, _ = move_via_clearance(tx, ty, z_approach, z_clear)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        _, _, terminated, truncated, _ = move_pose_xyz(tx, ty, z_touch)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        if terminated is None and truncated is None:
            attempts += 1
            hold_steps(1)
            continue

        _, _, terminated, truncated, _ = hold_steps(int(args.touch_settle_steps))
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        _, _, terminated, truncated, _ = move_pose_xyz(tx, ty, z_approach)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        if terminated is None and truncated is None:
            attempts += 1
            hold_steps(1)
            continue

        if int(args.post_touch_retreat) > 0:
            _, _, terminated, truncated, _ = move_pose_xyz(tx, ty, z_approach)
            if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
                episode_done["value"] = True
                break

        attempts += 1

    info_final = env_u.evaluate()
    success_final = _to_bool_scalar(info_final["success"])
    target_idx = _target_idx_from_next_mask(env_u)

    print("env_id:", env_id)
    print("task_family:", family)
    print("max_episode_steps:", max_env_steps)
    print("cue_steps:", cue_steps)
    print("empty_steps:", empty_steps)
    print("manip_start_step:", manip_start_step)
    print("attempts:", attempts)
    print("next_target_idx:", -1 if target_idx is None else int(target_idx))
    print("touched_total:", int(env_u.touched_cubes[0].to(torch.int32).sum().item()))
    if hasattr(env_u, "button_pressed"):
        print("button_pressed:", bool(_to_bool_scalar(env_u.button_pressed[0])))
    print("elapsed_steps:", _to_int_scalar(env_u.elapsed_steps[0]))
    print("success:", bool(success_final))
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
