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
Oracle motion planner for TraceShapeSeq procedural-memory tasks.

Supported:
- TraceShapeSeqEasy-VLA-v0
- TraceShapeSeqMedium-VLA-v0
- TraceShapeSeqHard-VLA-v0

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_trace_shape_seq.py --env-id TraceShapeSeqEasy-VLA-v0 --seed 123 --save-video 0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_trace_shape_seq.py --env-id TraceShapeSeqMedium-VLA-v0 --seed 7 --save-video 0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_trace_shape_seq.py --env-id TraceShapeSeqHard-VLA-v0 --seed 42 --save-video 1
"""

DEFAULT_ENV_ID = "TraceShapeSeqHard-VLA-v0"
VALID_ENV_IDS = {
    "TraceShapeSeqEasy-VLA-v0",
    "TraceShapeSeqMedium-VLA-v0",
    "TraceShapeSeqHard-VLA-v0",
}

SHAPE_NAME = {
    0: "Circle",
    1: "Square",
    2: "Triangle",
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


def _to_float_scalar(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().reshape(-1)[0].item())
    arr = np.asarray(x).reshape(-1)
    return float(arr[0]) if arr.size > 0 else 0.0


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


def _parse_float_list(csv_text: str):
    vals = []
    for tok in csv_text.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if len(vals) == 0:
        raise ValueError("Empty float list.")
    return vals


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-trajectory", type=int, default=0)
    parser.add_argument("--save-video", type=int, default=1)
    parser.add_argument("--overlay-info", type=int, default=1)
    parser.add_argument("--trajectory-dir", type=str, default=None)
    parser.add_argument("--trajectory-name", type=str, default="trajectory")

    parser.add_argument("--waypoint-stride", type=int, default=2)
    parser.add_argument("--max-waypoints-per-plan", type=int, default=0)
    parser.add_argument("--manip-phase-buffer-steps", type=int, default=2)

    parser.add_argument("--travel-clearance-z", type=float, default=0.20)
    parser.add_argument("--grasp-approach-z-offset", type=float, default=0.12)
    parser.add_argument("--grasp-z-offsets", type=str, default="0.012,0.008,0.004,0.0,-0.003")
    parser.add_argument("--grasp-settle-steps", type=int, default=2)
    parser.add_argument("--max-grasp-attempts", type=int, default=5)
    parser.add_argument("--gripper-preopen", type=float, default=0.22)
    parser.add_argument("--lift-after-grasp", type=float, default=0.0)

    parser.add_argument("--trace-z", type=float, default=0.11)
    parser.add_argument("--trace-cube-z-offset", type=float, default=0.002)
    parser.add_argument("--trace-tcp-min-z", type=float, default=0.06)
    parser.add_argument("--trace-tcp-max-z", type=float, default=0.11)
    parser.add_argument("--trace-settle-steps", type=int, default=1)
    parser.add_argument("--max-checkpoint-passes", type=int, default=3)
    parser.add_argument("--trace-waypoint-stride", type=int, default=4)
    parser.add_argument("--max-sequence-passes", type=int, default=32)

    parser.add_argument("--submit-extra-press-depth", type=float, default=0.008)
    parser.add_argument("--submit-unpress-offset", type=float, default=0.05)
    parser.add_argument("--submit-approach-offset", type=float, default=0.10)
    parser.add_argument("--submit-lift-offset", type=float, default=0.10)
    parser.add_argument("--max-submit-attempts", type=int, default=3)
    parser.add_argument("--regrasp-on-drop", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if env_id not in VALID_ENV_IDS:
        raise ValueError(f"Unsupported env_id={env_id!r}. Allowed: {sorted(VALID_ENV_IDS)}")
    if args.waypoint_stride <= 0:
        raise ValueError(f"--waypoint-stride must be > 0, got {args.waypoint_stride}")
    if args.max_waypoints_per_plan == 1:
        raise ValueError("--max-waypoints-per-plan must be <=0 or >=2")
    if args.trace_waypoint_stride <= 0:
        raise ValueError(f"--trace-waypoint-stride must be > 0, got {args.trace_waypoint_stride}")
    if args.max_sequence_passes <= 0:
        raise ValueError(f"--max-sequence-passes must be > 0, got {args.max_sequence_passes}")
    if args.max_submit_attempts <= 0:
        raise ValueError(f"--max-submit-attempts must be > 0, got {args.max_submit_attempts}")

    grasp_z_offsets = _parse_float_list(args.grasp_z_offsets)
    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

    wrappers_list = [
        (CurriculumPhaseNoopActionWrapperPdJointPos, {}),
        (RenderTraceShapeDebugWrapper, {}),
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
        joint_vel_limits=2.6,
        joint_acc_limits=2.6,
    )
    planner.gripper_state = float(np.clip(args.gripper_preopen, -1.0, 1.0))

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

    def _is_done_tuple(out):
        return _to_bool_scalar(out[2]) or _to_bool_scalar(out[3])

    def _is_plan_failed(out):
        return out[0] is None and out[1] is None and out[2] is None and out[3] is None and out[4] is None

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
            if _is_done_tuple(out):
                return out
            if _is_plan_failed(out):
                return out

        out = move_pose_xyz(float(x), float(y), z_clear)
        if _is_done_tuple(out):
            return out
        if _is_plan_failed(out):
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

    def set_gripper_state_safe(gripper_state: float, t: int = 2):
        planner.gripper_state = float(np.clip(gripper_state, -1.0, 1.0))
        if episode_done["value"]:
            return
        for _ in range(max(1, int(t))):
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

    def preopen_gripper_safe(t: int = 1):
        set_gripper_state_safe(float(args.gripper_preopen), t=t)

    def open_gripper_safe(t: int = 2):
        set_gripper_state_safe(1.0, t=t)

    def close_gripper_safe(t: int = 2):
        set_gripper_state_safe(-1.0, t=t)

    def green_pos_xyz() -> np.ndarray:
        return env_u.green_cube.pose.p[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def tcp_pos_xyz() -> np.ndarray:
        return env_u.agent.tcp.pose.p[0].detach().cpu().numpy().astype(np.float32, copy=False)

    def is_grasping_green() -> bool:
        try:
            return bool(env_u.agent.is_grasping(env_u.green_cube)[0].item())
        except Exception:
            return False

    def compute_trace_tcp_z() -> float:
        """Choose TCP Z so grasped cube center stays close to table surface."""
        if is_grasping_green():
            tcp_z = float(tcp_pos_xyz()[2])
            cube_z = float(green_pos_xyz()[2])
            tcp_minus_cube_z = tcp_z - cube_z
            target_cube_z = float(env_u.CUBE_HALFSIZE) + float(args.trace_cube_z_offset)
            z = target_cube_z + tcp_minus_cube_z
        else:
            z = float(args.trace_z)

        z = min(float(args.trace_tcp_max_z), max(float(args.trace_tcp_min_z), z))
        return z

    cue_steps = _to_int_scalar(env_u.cue_steps_per_env[0]) if hasattr(env_u, "cue_steps_per_env") else 0
    while (_to_int_scalar(env_u.elapsed_steps) < cue_steps + int(args.manip_phase_buffer_steps)) and (
        not episode_done["value"]
    ):
        hold_steps(1)

    def try_grasp_green_cube(max_attempts: int) -> bool:
        if episode_done["value"]:
            return False

        xy_jitter = [
            (0.0, 0.0),
            (0.006, 0.0),
            (-0.006, 0.0),
            (0.0, 0.006),
            (0.0, -0.006),
            (0.004, 0.004),
            (0.004, -0.004),
            (-0.004, 0.004),
            (-0.004, -0.004),
        ]

        attempts = 0
        while attempts < max_attempts and not episode_done["value"]:
            cube_pos = green_pos_xyz()
            base_x, base_y, base_z = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])
            ox, oy = xy_jitter[attempts % len(xy_jitter)]
            tx = base_x + ox
            ty = base_y + oy

            for z_off in grasp_z_offsets:
                z_grasp = base_z + float(z_off)
                z_approach = z_grasp + float(args.grasp_approach_z_offset)
                z_clear = max(float(args.travel_clearance_z), z_approach)

                out = move_via_clearance(tx, ty, z_approach, z_clear)
                if _is_done_tuple(out):
                    return False
                if _is_plan_failed(out):
                    continue

                preopen_gripper_safe(t=1)
                out = move_pose_xyz(tx, ty, z_grasp)
                if _is_done_tuple(out):
                    return False
                if _is_plan_failed(out):
                    continue

                out = move_pose_xyz(tx, ty, z_grasp - 0.004)
                if _is_done_tuple(out):
                    return False
                if _is_plan_failed(out):
                    continue

                close_gripper_safe(t=5)
                hold_steps(int(args.grasp_settle_steps))

                if float(args.lift_after_grasp) > 1e-4:
                    z_lift = z_grasp + float(args.lift_after_grasp)
                    out = move_pose_xyz(tx, ty, z_lift)
                    if _is_done_tuple(out):
                        return False
                    if _is_plan_failed(out):
                        continue
                    hold_steps(1)

                grasped = is_grasping_green()
                lifted = float(green_pos_xyz()[2]) > (base_z + 0.01)
                if grasped or lifted:
                    return True

                open_gripper_safe(t=2)
                hold_steps(1)

            attempts += 1

        return False

    grasp_success = try_grasp_green_cube(int(args.max_grasp_attempts))
    if not grasp_success:
        print("[warn] failed to establish stable grasp; continue with best-effort tracing")

    def trace_to_xy(x: float, y: float, z: float):
        out = move_pose_xyz(float(x), float(y), float(z))
        if _is_done_tuple(out):
            return out
        if _is_plan_failed(out):
            z_clear = max(float(args.travel_clearance_z), float(z) + 0.03)
            out = move_via_clearance(float(x), float(y), float(z), z_clear)
        return out

    trace_z = compute_trace_tcp_z()
    cur_tcp = tcp_pos_xyz()
    out = move_pose_xyz(
        float(cur_tcp[0]),
        float(cur_tcp[1]),
        trace_z,
    )
    if _is_plan_failed(out):
        out = move_via_clearance(
            float(cur_tcp[0]),
            float(cur_tcp[1]),
            trace_z,
            max(float(args.travel_clearance_z), trace_z + 0.03),
        )
    if _is_done_tuple(out):
        episode_done["value"] = True

    terminated = truncated = False

    def sequence_len() -> int:
        if hasattr(env_u, "sequence_len"):
            return max(1, _to_int_scalar(env_u.sequence_len[0]))
        return 1

    def active_shape_idx_from_info(info) -> int:
        if info is not None and "active_shape_idx" in info:
            idx = _to_int_scalar(info["active_shape_idx"])
        elif hasattr(env_u, "active_shape_idx"):
            idx = _to_int_scalar(env_u.active_shape_idx[0])
        else:
            idx = 0
        return int(np.clip(idx, 0, sequence_len() - 1))

    def active_shape_name(info) -> str:
        shape_id = -1
        if info is not None and "active_shape_id" in info:
            shape_id = _to_int_scalar(info["active_shape_id"])
        elif hasattr(env_u, "shape_sequence"):
            idx = active_shape_idx_from_info(info)
            shape_id = _to_int_scalar(env_u.shape_sequence[0, idx])
        return SHAPE_NAME.get(int(shape_id), f"Shape {shape_id}")

    submit_release_state = {"released_cube": False}

    def press_submit_button(max_attempts: int):
        if episode_done["value"]:
            return False, True, False, False
        if not hasattr(env_u, "button_xy") or not hasattr(env_u, "button_top_z"):
            return False, False, False, False

        bx = float(env_u.button_xy[0, 0].detach().cpu().item())
        by = float(env_u.button_xy[0, 1].detach().cpu().item())
        btop = float(env_u.button_top_z[0].detach().cpu().item())

        # Release the traced cube before pressing submit to avoid accidental contact artifacts.
        if not submit_release_state["released_cube"]:
            cur = tcp_pos_xyz()
            z_release = max(float(args.travel_clearance_z), float(cur[2]) + 0.03)
            out_release = move_pose_xyz(float(cur[0]), float(cur[1]), z_release)
            if _is_done_tuple(out_release):
                return (
                    False,
                    True,
                    _to_bool_scalar(out_release[2]),
                    _to_bool_scalar(out_release[3]),
                )
            open_gripper_safe(t=3)
            if episode_done["value"]:
                return False, True, False, False
            hold_steps(int(args.trace_settle_steps))
            if episode_done["value"]:
                return False, True, False, False
            # Press submit with closed fingers as requested.
            close_gripper_safe(t=2)
            if episode_done["value"]:
                return False, True, False, False
            submit_release_state["released_cube"] = True

        planner.gripper_state = -1.0
        press_threshold = float(env_u.BUTTON_CAP_TRAVEL * env_u.BUTTON_PRESS_EVENT_RATIO)
        z_required_for_press = btop + float(env_u.BUTTON_PRESS_Z_MARGIN) - press_threshold
        # Keep press target below the event threshold, but avoid unrealistically deep targets.
        z_press = z_required_for_press - float(args.submit_extra_press_depth)
        button_base_top_z = float(env_u.BUTTON_BASE_HALF_SIZE[2]) * 2.0
        z_press = max(button_base_top_z + 0.004, z_press)
        z_unpress = z_press + float(args.submit_unpress_offset)
        z_approach = z_unpress + float(args.submit_approach_offset)
        z_lift = btop + float(args.submit_lift_offset)
        z_clear = max(float(args.travel_clearance_z), z_approach)

        for _ in range(max_attempts):
            info_now = env_u.evaluate()
            if _to_bool_scalar(info_now.get("submitted", False)) or _to_bool_scalar(info_now.get("success", False)):
                return True, False, False, False

            out_local = move_via_clearance(bx, by, z_approach, z_clear)
            if _is_done_tuple(out_local):
                return (
                    False,
                    True,
                    _to_bool_scalar(out_local[2]),
                    _to_bool_scalar(out_local[3]),
                )
            if _is_plan_failed(out_local):
                continue

            out_local = move_pose_xyz(bx, by, z_press)
            if _is_done_tuple(out_local):
                return (
                    False,
                    True,
                    _to_bool_scalar(out_local[2]),
                    _to_bool_scalar(out_local[3]),
                )
            if _is_plan_failed(out_local):
                continue

            hold_steps(int(args.trace_settle_steps))
            if episode_done["value"]:
                return False, True, False, False

            out_local = move_pose_xyz(bx, by, z_unpress)
            if _is_done_tuple(out_local):
                return (
                    False,
                    True,
                    _to_bool_scalar(out_local[2]),
                    _to_bool_scalar(out_local[3]),
                )
            if _is_plan_failed(out_local):
                continue

            out_local = move_pose_xyz(bx, by, z_lift)
            if _is_done_tuple(out_local):
                return (
                    False,
                    True,
                    _to_bool_scalar(out_local[2]),
                    _to_bool_scalar(out_local[3]),
                )

            info_now = env_u.evaluate()
            if _to_bool_scalar(info_now.get("submitted", False)) or _to_bool_scalar(info_now.get("success", False)):
                return True, False, False, False
        return False, False, False, False

    seq_passes = 0
    while seq_passes < int(args.max_sequence_passes):
        if episode_done["value"]:
            break

        info_now = env_u.evaluate()
        if _to_bool_scalar(info_now.get("success", False)) or _to_bool_scalar(info_now.get("submitted", False)):
            break

        if _to_bool_scalar(info_now.get("all_shapes_closed", False)):
            submitted, done_step, term_step, trunc_step = press_submit_button(int(args.max_submit_attempts))
            if done_step:
                episode_done["value"] = True
                terminated = bool(term_step)
                truncated = bool(trunc_step)
                break
            if submitted:
                break
            hold_steps(1)
            seq_passes += 1
            continue

        active_idx = active_shape_idx_from_info(info_now)
        checkpoints = env_u.checkpoints[0, active_idx].detach().cpu().numpy()
        waypoints = env_u.waypoints[0, active_idx].detach().cpu().numpy()
        visited = env_u.checkpoint_visited[0, active_idx].detach().cpu().numpy().astype(bool)

        for _ in range(int(args.max_checkpoint_passes)):
            if episode_done["value"]:
                break
            pending = [idx for idx in range(len(visited)) if not visited[idx]]
            if len(pending) == 0:
                break

            progress_made = False
            for cp_idx in pending:
                if episode_done["value"]:
                    break
                cp = checkpoints[int(cp_idx)]
                out = trace_to_xy(float(cp[0]), float(cp[1]), trace_z)
                if _is_done_tuple(out):
                    episode_done["value"] = True
                    terminated = _to_bool_scalar(out[2])
                    truncated = _to_bool_scalar(out[3])
                    break
                if _is_plan_failed(out):
                    continue

                progress_made = True
                hold_steps(int(args.trace_settle_steps))
                info_now = env_u.evaluate()
                if _to_bool_scalar(info_now.get("success", False)) or _to_bool_scalar(info_now.get("submitted", False)):
                    break
                if _to_bool_scalar(info_now.get("all_shapes_closed", False)):
                    break
                if active_shape_idx_from_info(info_now) != active_idx:
                    break

                if bool(args.regrasp_on_drop) and (not is_grasping_green()):
                    regrasp_ok = try_grasp_green_cube(max_attempts=2)
                    if regrasp_ok:
                        trace_z = compute_trace_tcp_z()

                visited = env_u.checkpoint_visited[0, active_idx].detach().cpu().numpy().astype(bool)

            if episode_done["value"]:
                break
            info_now = env_u.evaluate()
            if _to_bool_scalar(info_now.get("success", False)) or _to_bool_scalar(info_now.get("submitted", False)):
                break
            if _to_bool_scalar(info_now.get("all_shapes_closed", False)):
                break
            if active_shape_idx_from_info(info_now) != active_idx:
                break
            if _to_bool_scalar(info_now.get("is_active_contour_closed", False)):
                break
            if not progress_made:
                break

        if episode_done["value"]:
            break

        info_now = env_u.evaluate()
        if _to_bool_scalar(info_now.get("success", False)) or _to_bool_scalar(info_now.get("submitted", False)):
            break
        if _to_bool_scalar(info_now.get("all_shapes_closed", False)):
            seq_passes += 1
            continue
        if active_shape_idx_from_info(info_now) != active_idx:
            seq_passes += 1
            continue
        if _to_bool_scalar(info_now.get("is_active_contour_closed", False)):
            seq_passes += 1
            continue

        n_wp = len(waypoints)
        idxs = list(range(0, n_wp, int(args.trace_waypoint_stride)))
        if len(idxs) == 0 or idxs[-1] != n_wp - 1:
            idxs.append(n_wp - 1)
        if idxs[-1] != 0:
            idxs.append(0)

        for wp_i in idxs:
            if episode_done["value"]:
                break
            wp = waypoints[int(wp_i)]
            out = trace_to_xy(float(wp[0]), float(wp[1]), trace_z)
            if _is_done_tuple(out):
                episode_done["value"] = True
                terminated = _to_bool_scalar(out[2])
                truncated = _to_bool_scalar(out[3])
                break
            if _is_plan_failed(out):
                continue

            hold_steps(int(args.trace_settle_steps))
            info_now = env_u.evaluate()
            if _to_bool_scalar(info_now.get("success", False)) or _to_bool_scalar(info_now.get("submitted", False)):
                break
            if _to_bool_scalar(info_now.get("all_shapes_closed", False)):
                break
            if active_shape_idx_from_info(info_now) != active_idx:
                break

            if bool(args.regrasp_on_drop) and (not is_grasping_green()):
                regrasp_ok = try_grasp_green_cube(max_attempts=1)
                if regrasp_ok:
                    trace_z = compute_trace_tcp_z()

        seq_passes += 1

    info_now = env_u.evaluate()
    if (
        (not episode_done["value"])
        and (not _to_bool_scalar(info_now.get("submitted", False)))
        and (not _to_bool_scalar(info_now.get("success", False)))
    ):
        submitted, done_step, term_step, trunc_step = press_submit_button(int(args.max_submit_attempts))
        if done_step:
            episode_done["value"] = True
            terminated = bool(term_step)
            truncated = bool(trunc_step)
        elif submitted:
            info_now = env_u.evaluate()

    info_final = env_u.evaluate()
    success_final = _to_bool_scalar(info_final.get("success", False))
    submitted_final = _to_bool_scalar(info_final.get("submitted", False))
    all_shapes_closed_final = _to_bool_scalar(info_final.get("all_shapes_closed", False))
    sequence_progress = _to_float_scalar(info_final.get("sequence_progress", 0.0))
    active_idx_final = active_shape_idx_from_info(info_final)
    active_name_final = active_shape_name(info_final)

    seq_len = sequence_len()
    if hasattr(env_u, "shape_sequence"):
        shape_sequence_final = env_u.shape_sequence[0, :seq_len].detach().cpu().numpy().astype(np.int64).tolist()
    else:
        shape_sequence_final = []

    if hasattr(env_u, "shape_closed"):
        closed_mask = env_u.shape_closed[0, :seq_len].detach().cpu().numpy().astype(bool)
        closed_count = int(closed_mask.sum())
    else:
        closed_count = int(round(sequence_progress * seq_len))

    active_visited = env_u.checkpoint_visited[0, active_idx_final].detach().cpu().numpy().astype(bool)
    active_visited_count = int(active_visited.sum())
    active_total_count = int(active_visited.shape[0])

    print("env_id:", env_id)
    print("shape_sequence_ids:", shape_sequence_final)
    print("max_episode_steps:", int(max_env_steps))
    print("cue_steps:", int(cue_steps))
    print("elapsed_steps:", _to_int_scalar(env_u.elapsed_steps[0]))
    print("grasp_success:", bool(grasp_success))
    print("is_grasping_green:", bool(is_grasping_green()))
    print("sequence_len:", int(seq_len))
    print("active_shape_idx:", int(active_idx_final))
    print("active_shape_name:", active_name_final)
    print("sequence_closed:", f"{closed_count}/{seq_len}")
    print("active_checkpoints_visited:", f"{active_visited_count}/{active_total_count}")
    print("all_shapes_closed:", bool(all_shapes_closed_final))
    print("submitted:", bool(submitted_final))
    if hasattr(env_u, "button_pressed"):
        print("button_pressed:", bool(_to_bool_scalar(env_u.button_pressed[0])))
    print("sequence_progress:", float(sequence_progress))
    print("success:", bool(success_final))
    print("terminated:", bool(terminated))
    print("truncated:", bool(truncated))

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
