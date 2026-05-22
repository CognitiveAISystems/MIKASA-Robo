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
from transforms3d.quaternions import axangle2quat, qmult

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.memory_envs import *
from mikasa_robo_suite.vla.utils.wrappers import *

"""
Oracle motion planner for shell-game shuffle touch tasks.

Supported:
- ShellGameShuffleTouch-VLA-v0
- ShellGameShuffleTouch-Long-VLA-v0
- ShellGameShuffleColorLampTouch-VLA-v0
- ShellGameShuffleColorLampTouch-Long-VLA-v0

Example:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_shuffle.py --env-id ShellGameShuffleTouch-Long-VLA-v0 --seed 123 --save-video 0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_shuffle.py --env-id ShellGameShuffleColorLampTouch-Long-VLA-v0 --seed 7 --save-video 0
"""

DEFAULT_ENV_ID = "ShellGameShuffleTouch-Long-VLA-v0"
VALID_ENV_IDS = {
    "ShellGameShuffleTouch-VLA-v0",
    "ShellGameShuffleTouch-Long-VLA-v0",
    "ShellGameShuffleColorLampTouch-VLA-v0",
    "ShellGameShuffleColorLampTouch-Long-VLA-v0",
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


def _resolve_family(env_id: str) -> str:
    if env_id.startswith("ShellGameShuffleColorLampTouch"):
        return "shuffle_color_lamp_touch"
    if env_id.startswith("ShellGameShuffleTouch"):
        return "shuffle_touch"
    raise ValueError(f"Unsupported env_id family: {env_id}")


def _target_slot_from_oracle(env_u, info_now):
    cand = None
    if isinstance(info_now, dict) and "oracle_info" in info_now:
        cand = info_now["oracle_info"]
    if cand is None and hasattr(env_u, "oracle_info"):
        cand = env_u.oracle_info
    if cand is None:
        return None

    if torch.is_tensor(cand):
        arr = cand.detach().cpu().reshape(-1).numpy()
    else:
        arr = np.asarray(cand).reshape(-1)

    if arr.size == 0:
        return None
    slot = int(arr[0])
    if slot not in (0, 1, 2):
        return None
    return slot


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

    parser.add_argument("--approach-z-offset", type=float, default=0.10)
    parser.add_argument("--target-z-offset", type=float, default=0.02)
    parser.add_argument("--travel-clearance-z", type=float, default=0.22)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--manip-phase-buffer-steps", type=int, default=2)
    parser.add_argument("--max-target-attempts", type=int, default=1)
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
        raise ValueError(f"Unsupported env_id={env_id!r}. Allowed: {sorted(VALID_ENV_IDS)}")
    if args.waypoint_stride <= 0:
        raise ValueError(f"--waypoint-stride must be > 0, got {args.waypoint_stride}")
    if args.max_waypoints_per_plan == 1:
        raise ValueError("--max-waypoints-per-plan must be <=0 or >=2")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)
    if "-Long-" not in env_id and int(args.short_episode_steps) > 0:
        max_env_steps = max(max_env_steps, int(args.short_episode_steps))

    wrappers_list = []
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

    cue_steps = _to_int_scalar(env_u.cue_steps_per_env[0]) if hasattr(env_u, "cue_steps_per_env") else 0
    shuffle_steps = _to_int_scalar(env_u.shuffle_steps_per_env[0]) if hasattr(env_u, "shuffle_steps_per_env") else 0
    manip_start_step = cue_steps + shuffle_steps

    while _to_int_scalar(env_u.elapsed_steps) < manip_start_step + int(args.manip_phase_buffer_steps) and (
        not episode_done["value"]
    ):
        hold_steps(1)

    family = _resolve_family(env_id)
    attempts = 0
    terminated = truncated = False
    target_slot = None

    while attempts < int(args.max_target_attempts):
        if episode_done["value"]:
            break

        info_now = env_u.evaluate()
        if _to_bool_scalar(info_now["success"]):
            break

        elapsed_now = _to_int_scalar(env_u.elapsed_steps)
        if elapsed_now < manip_start_step:
            hold_steps(1)
            attempts += 1
            continue

        target_slot = _target_slot_from_oracle(env_u, info_now)
        if target_slot is None:
            hold_steps(1)
            attempts += 1
            continue

        if not hasattr(env_u, "slot_positions"):
            raise RuntimeError("Environment does not expose slot_positions required by shell-game planner.")

        target_pos = env_u.slot_positions[0, int(target_slot)].detach().cpu().numpy()
        tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])

        z_goal = tz + float(args.target_z_offset)
        z_approach = z_goal + float(args.approach_z_offset)
        z_clear = max(float(args.travel_clearance_z), z_approach)

        _, _, terminated, truncated, _ = move_via_clearance(tx, ty, z_approach, z_clear)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        _, _, terminated, truncated, _ = move_pose_xyz(tx, ty, z_goal)
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        _, _, terminated, truncated, _ = hold_steps(int(args.settle_steps))
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break
        _, _, terminated, truncated, _ = hold_steps(int(args.settle_steps))
        if _to_bool_scalar(terminated) or _to_bool_scalar(truncated):
            episode_done["value"] = True
            break

        info_now = env_u.evaluate()
        if _to_bool_scalar(info_now.get("success", False)):
            attempts += 1
            break

        attempts += 1

    info_final = env_u.evaluate()
    success_final = _to_bool_scalar(info_final["success"])
    if target_slot is None:
        target_slot = _target_slot_from_oracle(env_u, info_final)

    print("env_id:", env_id)
    print("task_family:", family)
    print("max_episode_steps:", max_env_steps)
    print("cue_steps:", cue_steps)
    print("shuffle_steps:", shuffle_steps)
    print("manip_start_step:", manip_start_step)
    print("attempts:", attempts)
    print("target_slot:", -1 if target_slot is None else int(target_slot))
    print("elapsed_steps:", _to_int_scalar(env_u.elapsed_steps[0]))
    print("success:", bool(success_final))
    if isinstance(info_final, dict) and "is_obj_placed" in info_final:
        print("is_obj_placed:", _to_bool_scalar(info_final["is_obj_placed"]))
    if isinstance(info_final, dict) and "is_robot_static" in info_final:
        print("is_robot_static:", _to_bool_scalar(info_final["is_robot_static"]))
    if isinstance(info_final, dict) and "is_mug_displacement_ok" in info_final:
        print("is_mug_displacement_ok:", _to_bool_scalar(info_final["is_mug_displacement_ok"]))
    if isinstance(info_final, dict) and "mug_max_displacement" in info_final:
        mm = info_final["mug_max_displacement"]
        if torch.is_tensor(mm):
            mm = float(mm.detach().cpu().reshape(-1)[0].item())
        else:
            mm = float(np.asarray(mm).reshape(-1)[0])
        print("mug_max_displacement:", mm)
    print("terminated:", _to_bool_scalar(terminated))
    print("truncated:", _to_bool_scalar(truncated))

    env.close()

    mp4s = sorted(glob(f"{out_dir}/*.mp4"))
    print("Saved videos:", mp4s)
    if bool(args.save_video):
        assert len(mp4s) > 0, f"No mp4 found in {out_dir}"
        Video(mp4s[-1], embed=True, width=640)


if __name__ == "__main__":
    main()
