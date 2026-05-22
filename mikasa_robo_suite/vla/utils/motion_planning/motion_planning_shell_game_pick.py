"""Oracle motion planning for ShellGamePick VLA tasks.

Strategy:
1. Wait for the cue phase to end (mugs appear on the table).
2. Determine which mug hides the ball (from oracle_info).
3. Open gripper, approach the handle of the target mug from above.
4. Lower to handle height, close gripper to grasp.
5. Lift the mug to the goal position above the original ball location.
6. Hold until success (is_obj_placed & is_robot_static).

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_pick.py --env-id ShellGamePick-VLA-v0 --seed 42 --save-video 1
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_shell_game_pick.py --env-id ShellGamePick-VLA-v0 --seed 123 --save-video 0
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
from transforms3d.quaternions import axangle2quat, mat2quat, qmult

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.memory_envs import *
from mikasa_robo_suite.vla.utils.wrappers import *

DEFAULT_ENV_ID = "ShellGamePick-VLA-v0"
VALID_ENV_IDS = {"ShellGamePick-VLA-v0"}

# Handle direction in world frame for the fixed mug quaternion q=[0,1,2.5,0] normalized.
# Computed as: 180-deg rotation around [0.371, 0.929, 0] applied to local +X.
HANDLE_DIR = np.array([-0.7241, 0.6897, 0.0])
HANDLE_DIR /= np.linalg.norm(HANDLE_DIR)


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
    return float(np.asarray(x).reshape(-1)[0])


def _elapsed_from_info(info):
    if info is None or "elapsed_steps" not in info:
        return None
    x = info["elapsed_steps"]
    if torch.is_tensor(x):
        return int(x.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(x).reshape(-1)[0])


def _validate_flatten_obs(obs):
    if not isinstance(obs, dict):
        raise RuntimeError(f"Expected dict observation, got {type(obs).__name__}.")
    if "rgb" not in obs or "proprio" not in obs:
        raise RuntimeError("Missing 'rgb'/'proprio' keys in observation.")


def build_hold_action_pd_joint_pos(base_env, gripper_state_override: float | None = None):
    robot = base_env.agent.robot
    qpos = robot.get_qpos()
    qpos_arm = qpos[..., :-2].detach().cpu().numpy()

    if gripper_state_override is None:
        qpos_gripper = qpos[..., -2].detach().cpu().numpy()
        gripper_low, gripper_high = -0.01, 0.04
        mid = 0.5 * (gripper_high + gripper_low)
        half = 0.5 * (gripper_high - gripper_low)
        grip_norm = np.clip((qpos_gripper - mid) / half, -1.0, 1.0)
    else:
        grip_norm = np.full((qpos_arm.shape[0],), float(np.clip(gripper_state_override, -1.0, 1.0)), dtype=np.float32)

    action = np.concatenate([qpos_arm, grip_norm[..., None]], axis=1).astype(np.float32)
    return action[0]


def _target_mug_position(env_u, slot: int) -> np.ndarray:
    """Return the (x, y, z) center of the target mug in world frame."""
    mugs = [env_u.mug_left, env_u.mug_center, env_u.mug_right]
    pos = mugs[slot].pose.p[0].detach().cpu().numpy()
    return pos.astype(np.float64)


def _goal_position(env_u) -> np.ndarray:
    return env_u.goal_site.pose.p[0].detach().cpu().numpy().astype(np.float64)


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

    # Handle grasp geometry
    parser.add_argument(
        "--handle-distance", type=float, default=0.12, help="Distance from mug center to handle grasp point (meters)."
    )
    parser.add_argument(
        "--pre-approach-distance",
        type=float,
        default=0.10,
        help="How far out from handle to start the side approach (meters).",
    )
    parser.add_argument(
        "--grasp-z-offset", type=float, default=0.0, help="Z offset relative to mug center for handle grasp height."
    )
    parser.add_argument("--travel-clearance-z", type=float, default=0.25)
    parser.add_argument("--gripper-close-steps", type=int, default=15)
    parser.add_argument("--settle-steps", type=int, default=25)
    parser.add_argument("--manip-phase-buffer-steps", type=int, default=2)
    parser.add_argument("--short-episode-steps", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if env_id not in VALID_ENV_IDS:
        raise ValueError(f"Unsupported env_id={env_id!r}. Allowed: {sorted(VALID_ENV_IDS)}")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)
    if int(args.short_episode_steps) > 0:
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
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, oracle=False, joints=True)

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

    episode_done = {"value": False}

    # ---------------------------------------------------------------
    # Override follow_path for waypoint-strided stepping
    # ---------------------------------------------------------------
    def fast_follow_path(self, result, refine_steps: int = 0):  # noqa: ARG001
        if episode_done["value"]:
            return None, None, True, False, None
        n_step = result["position"].shape[0]
        if n_step <= 0:
            return None, None, None, None, None

        idxs = list(range(0, n_step, int(args.waypoint_stride)))
        if idxs[-1] != n_step - 1:
            idxs.append(n_step - 1)

        obs_ = reward_ = terminated_ = truncated_ = info_ = None
        for i in idxs:
            if _to_int_scalar(env_u.elapsed_steps) >= max_env_steps - 1:
                episode_done["value"] = True
                break
            qpos = result["position"][i]
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

    # ---------------------------------------------------------------
    # TCP orientation: horizontal (parallel to table), approaching
    # strictly along the +X axis so the gripper doesn't push
    # neighboring mugs aside.
    #   Z-axis = [1, 0, 0] (approach along +X toward mug)
    #   X-axis = [0, 0, 1] (up — fingers close horizontally)
    #   Y-axis = Z × X (right-hand rule)
    # ---------------------------------------------------------------
    approach_dir = np.array([1.0, 0.0, 0.0])  # approach along +X

    z_ax = approach_dir
    x_ax = np.array([0.0, 0.0, 1.0])  # up — fingers close horizontally (parallel to table)
    y_ax = np.cross(z_ax, x_ax)
    y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= np.linalg.norm(x_ax)

    R_horizontal = np.column_stack([x_ax, y_ax, z_ax])
    tcp_q_horizontal = mat2quat(R_horizontal)  # [w, x, y, z]

    # Roll variants around the approach axis (different finger alignments)
    q_candidates = [tcp_q_horizontal]
    for roll in [np.pi / 2, -np.pi / 2, np.pi]:
        dq_roll = axangle2quat(approach_dir, roll)
        q_candidates.append(qmult(dq_roll, tcp_q_horizontal))

    tcp_q_state = {"q": tcp_q_horizontal.copy()}

    # ---------------------------------------------------------------
    # Movement helpers
    # ---------------------------------------------------------------
    def move_pose(pose: sapien.Pose):
        if episode_done["value"]:
            return None, None, True, False, None
        q_try_list = [tcp_q_state["q"]]
        for q_nom in q_candidates:
            if not np.allclose(q_nom, q_try_list[0]):
                q_try_list.append(q_nom)

        for q_try in q_try_list:
            target = sapien.Pose(p=pose.p, q=q_try)
            out = planner.move_to_pose_with_screw(target, dry_run=False, refine_steps=0)
            if out == -1:
                out = planner.move_to_pose_with_RRTConnect(target, dry_run=False, refine_steps=0)
            if out != -1:
                tcp_q_state["q"] = np.asarray(q_try, dtype=np.float64)
                return out
        print(f"[warn] failed move to [{pose.p[0]:.3f}, {pose.p[1]:.3f}, {pose.p[2]:.3f}]")
        return None, None, None, None, None

    def move_xyz(x, y, z):
        return move_pose(sapien.Pose(p=[float(x), float(y), float(z)], q=tcp_q_state["q"]))

    def hold_steps(k: int = 1):
        obs_ = reward_ = terminated_ = truncated_ = info_ = None
        for _ in range(k):
            if episode_done["value"]:
                break
            hold = build_hold_action_pd_joint_pos(env_u, gripper_state_override=float(planner.gripper_state))
            obs_, reward_, terminated_, truncated_, info_ = env.step(hold)
            if _to_bool_scalar(terminated_) or _to_bool_scalar(truncated_):
                episode_done["value"] = True
                break
            elapsed = _elapsed_from_info(info_)
            if elapsed is not None and elapsed >= max_env_steps:
                episode_done["value"] = True
                break
        return obs_, reward_, terminated_, truncated_, info_

    def is_done():
        return episode_done["value"]

    # ---------------------------------------------------------------
    # Phase 0: Wait for cue phase to end
    # ---------------------------------------------------------------
    cue_steps = _to_int_scalar(env_u.cue_steps_per_env[0]) if hasattr(env_u, "cue_steps_per_env") else 0
    manip_start = cue_steps + int(args.manip_phase_buffer_steps)

    while _to_int_scalar(env_u.elapsed_steps) < manip_start and not is_done():
        hold_steps(1)

    # ---------------------------------------------------------------
    # Determine target mug from oracle
    # ---------------------------------------------------------------
    info_now = env_u.evaluate()
    oracle = info_now.get("oracle_info", None)
    if oracle is None:
        oracle = env_u.oracle_info
    if torch.is_tensor(oracle):
        target_slot = int(oracle.detach().cpu().reshape(-1)[0].item())
    else:
        target_slot = int(np.asarray(oracle).reshape(-1)[0])

    mug_pos = _target_mug_position(env_u, target_slot)
    goal_pos = _goal_position(env_u)
    mug_x, mug_y, mug_z = float(mug_pos[0]), float(mug_pos[1]), float(mug_pos[2])

    # Handle grasp point: offset from mug center along handle direction
    handle_offset = HANDLE_DIR * float(args.handle_distance)
    handle_x = mug_x + handle_offset[0]
    handle_y = mug_y + handle_offset[1]
    handle_z = mug_z + float(args.grasp_z_offset)

    goal_x, goal_y, goal_z = float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2])

    # Pre-approach point: offset from handle strictly along -X (away from mug).
    # The gripper approaches along +X so it doesn't push neighboring mugs.
    pre_dist = float(args.pre_approach_distance)
    pre_x = handle_x - pre_dist  # further back in -X
    pre_y = handle_y  # same Y — straight-line approach along X
    pre_z = handle_z  # same height as handle

    z_clear = float(args.travel_clearance_z)

    print("=== ShellGamePick Oracle MP ===")
    print(f"env_id: {env_id}")
    print(f"target_slot: {target_slot} ({'left' if target_slot == 0 else 'center' if target_slot == 1 else 'right'})")
    print(f"mug_center: [{mug_x:.3f}, {mug_y:.3f}, {mug_z:.3f}]")
    print(f"handle_grasp: [{handle_x:.3f}, {handle_y:.3f}, {handle_z:.3f}]")
    print(f"pre_approach: [{pre_x:.3f}, {pre_y:.3f}, {pre_z:.3f}]")
    print(f"goal: [{goal_x:.3f}, {goal_y:.3f}, {goal_z:.3f}]")
    print()

    # ---------------------------------------------------------------
    # Phase 1: Open gripper, fly to pre-approach with horizontal orientation.
    #   Instead of rotating in place (planner picks wrong arc), we combine
    #   translation and rotation: lift → move to pre-approach at clearance
    #   with target orientation → lower.
    # ---------------------------------------------------------------
    planner.gripper_state = 1  # open

    # 1a. Lift to clearance height (keep whatever orientation)
    if not is_done():
        cur = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
        move_xyz(float(cur[0]), float(cur[1]), z_clear)

    # 1b. Move to pre-approach XY at clearance, switching to horizontal
    #     orientation during the translation (combined motion).
    if not is_done():
        move_pose(sapien.Pose(p=[pre_x, pre_y, z_clear], q=tcp_q_horizontal))

    # 1c. Lower to handle height
    if not is_done():
        move_xyz(pre_x, pre_y, pre_z)

    # ---------------------------------------------------------------
    # Phase 2: Approach handle horizontally (side approach) and grasp
    # ---------------------------------------------------------------
    if not is_done():
        move_xyz(handle_x, handle_y, handle_z)

    if not is_done():
        planner.gripper_state = -1  # close
        hold_steps(args.gripper_close_steps)

    # ---------------------------------------------------------------
    # Phase 3: Lift the mug straight up.
    #   Read actual TCP orientation after grasping (may differ slightly
    #   from tcp_q_state) and use that for the lift target to avoid
    #   any unwanted rotation during the lift.
    # ---------------------------------------------------------------
    if not is_done():
        cur_tcp_pose = env_u.agent.tcp.pose
        cur_p = cur_tcp_pose.p[0].detach().cpu().numpy()
        cur_q = cur_tcp_pose.q[0].detach().cpu().numpy()
        lift_target = sapien.Pose(
            p=[float(cur_p[0]), float(cur_p[1]), z_clear],
            q=cur_q,  # keep exact current orientation — no rotation
        )
        out = planner.move_to_pose_with_screw(lift_target, dry_run=False, refine_steps=0)
        if out == -1:
            print("[warn] lift screw failed, trying RRT")
            planner.move_to_pose_with_RRTConnect(lift_target, dry_run=False, refine_steps=0)

    # ---------------------------------------------------------------
    # Phase 4: Move to goal position
    # ---------------------------------------------------------------
    if not is_done():
        move_xyz(goal_x, goal_y, z_clear)

    if not is_done():
        move_xyz(goal_x, goal_y, goal_z)

    # ---------------------------------------------------------------
    # Phase 5: Hold for success
    # ---------------------------------------------------------------
    if not is_done():
        hold_steps(args.settle_steps)

    # ---------------------------------------------------------------
    # Report results
    # ---------------------------------------------------------------
    info_final = env_u.evaluate()
    success_final = _to_bool_scalar(info_final["success"])
    elapsed_steps = _to_int_scalar(env_u.elapsed_steps[0])

    print(f"max_episode_steps: {max_env_steps}")
    print(f"cue_steps: {cue_steps}")
    print(f"elapsed_steps: {elapsed_steps}")
    print(f"success: {success_final}")
    print(f"is_obj_placed: {_to_bool_scalar(info_final.get('is_obj_placed', False))}")
    print(f"is_robot_static: {_to_bool_scalar(info_final.get('is_robot_static', False))}")
    print(f"is_grasped: {_to_bool_scalar(info_final.get('is_grasped', False))}")

    env.close()

    mp4s = sorted(glob(f"{out_dir}/*.mp4"))
    print(f"Saved videos: {mp4s}")
    if bool(args.save_video) and mp4s:
        Video(mp4s[-1], embed=True, width=640)


if __name__ == "__main__":
    main()
