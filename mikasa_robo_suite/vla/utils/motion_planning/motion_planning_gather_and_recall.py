"""Oracle motion planning for GatherAndRecall VLA tasks.

Strategy:
1. Pick up each cube one by one and place it on the target disc.
2. The lamp flashes a color (red/green/blue) during the cube-moving phase.
   The oracle reads the true flash color from the environment state.
3. After all cubes are placed, press the button matching the flash color.

Examples:
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_gather_and_recall.py --env-id GatherAndRecall3-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_gather_and_recall.py --env-id GatherAndRecall5-VLA-v0
python mikasa_robo_suite/vla/utils/motion_planning/motion_planning_gather_and_recall.py --env-id GatherAndRecall7-VLA-v0
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

VALID_ENV_PREFIXES = ("GatherAndRecall",)


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
        default="GatherAndRecall5-VLA-v0",
        help="Registered GatherAndRecall-* env id.",
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
        default=3,
        help="Hold steps after release for the cube to settle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id

    if not any(env_id.startswith(p) for p in VALID_ENV_PREFIXES):
        raise ValueError(f"Expected a GatherAndRecall-* env, got {env_id!r}.")

    env_spec = gym.spec(env_id)
    max_env_steps = int(env_spec.max_episode_steps)

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

    n_cubes = int(env_u.N_CUBES)
    disc_pos = env_u.disc.pose.p[0].detach().cpu().numpy()
    disc_x, disc_y = float(disc_pos[0]), float(disc_pos[1])
    disc_hh = float(env_u.DISC_HALF_HEIGHT)
    cube_hs = float(env_u.CUBE_HALF_SIZE)

    # Buttons positions (3 buttons: red=0, green=1, blue=2)
    buttons_xy = []
    for btn_idx in range(3):
        bxy = env_u.buttons_xy[btn_idx, 0].detach().cpu().numpy()
        buttons_xy.append((float(bxy[0]), float(bxy[1])))

    button_top_z = float(env_u.button_top_z[0].detach().cpu().item())
    flash_color = _to_int_scalar(env_u.flash_color[0])
    flash_trigger_count = _to_int_scalar(env_u.flash_trigger_count[0])
    color_names = ["RED", "GREEN", "BLUE"]

    # Keep transfers high enough for wider scene layouts.
    z_lift = max(0.24, button_top_z + 0.12)
    z_place = disc_hh * 2 + cube_hs + 0.003

    # Pre-compute spread placement targets on a wide ring over the disc.
    if n_cubes <= 1:
        place_targets_xy = [(disc_x, disc_y)]
    else:
        # Keep cube centers well inside the disc while maximizing separation.
        ring_radius = float(
            np.clip(
                float(env_u.DISC_RADIUS) - cube_hs - 0.012,
                0.05,
                0.075,
            )
        )
        start_angle = np.pi / 2.0
        place_targets_xy = []
        for i in range(n_cubes):
            a = start_angle + 2.0 * np.pi * i / n_cubes
            place_targets_xy.append(
                (
                    disc_x + ring_radius * np.cos(a),
                    disc_y + ring_radius * np.sin(a),
                )
            )

    print("=== GatherAndRecall Oracle MP ===")
    print(f"env_id: {env_id}")
    print(f"n_cubes: {n_cubes}")
    print(f"flash_color: {flash_color} ({color_names[flash_color]})")
    print(f"flash_trigger_count: {flash_trigger_count}")
    print(f"disc: [{disc_x:.3f}, {disc_y:.3f}]")
    print("placement targets:")
    for i, (px, py) in enumerate(place_targets_xy):
        print(f"  place_{i}: [{px:.3f}, {py:.3f}]")
    for i, (bx, by) in enumerate(buttons_xy):
        print(f"button_{i} ({color_names[i]}): [{bx:.3f}, {by:.3f}]")
    print()

    terminated = truncated = None

    # ---------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------
    def move_pose(p):
        if episode_done["value"]:
            return None, None, True, False, None
        res = planner.move_to_pose_with_screw(p, dry_run=False, refine_steps=0)
        return res

    def move_pose_retry(p, retries=3):
        """Try screw plan; on failure retry from a safe height first."""
        if episode_done["value"]:
            return None, None, True, False, None
        for attempt in range(retries):
            res = planner.move_to_pose_with_screw(p, dry_run=False, refine_steps=0)
            if res[0] is not None:
                return res
            if attempt < retries - 1 and not episode_done["value"]:
                # Retreat up and retry
                cur_tcp = env_u.agent.tcp.pose.p[0].detach().cpu().numpy()
                safe = sapien.Pose(
                    p=[float(cur_tcp[0]), float(cur_tcp[1]), z_lift],
                    q=tcp_q,
                )
                planner.move_to_pose_with_screw(safe, dry_run=False, refine_steps=0)
                print(f"    screw plan retry {attempt + 1}/{retries}")
        return res

    def hold_steps(k: int = 1):
        nonlocal terminated, truncated
        obs = reward = info = None
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

    def pick_and_place_cube(cube_idx):
        """Pick cube at cube_idx and place it on the disc."""
        cube = env_u.cubes[cube_idx]
        cube_pos = cube.pose.p[0].detach().cpu().numpy()
        cx, cy, cz = float(cube_pos[0]), float(cube_pos[1]), float(cube_pos[2])

        # Approach from well above to avoid collisions near table surface
        z_approach = max(cz + 0.14, z_lift - 0.02)
        z_grasp = cz + 0.003  # slightly above cube center for reliable grasp

        print(f"  [cube {cube_idx}] pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}]")

        # Open gripper
        planner.gripper_state = 1

        # Approach above cube
        if not episode_done["value"]:
            target = sapien.Pose(p=[cx, cy, z_approach], q=tcp_q)
            move_pose_retry(target)

        # Lower to grasp
        if not episode_done["value"]:
            target = sapien.Pose(p=[cx, cy, z_grasp], q=tcp_q)
            move_pose_retry(target)

        # Close gripper
        if not episode_done["value"]:
            planner.gripper_state = -1
            hold_steps(args.gripper_close_steps)

        # Lift
        if not episode_done["value"]:
            target = sapien.Pose(p=[cx, cy, z_lift], q=tcp_q)
            move_pose(target)

        place_x, place_y = place_targets_xy[cube_idx]

        # Move above placement target (avoid routing through disc center)
        if not episode_done["value"]:
            target = sapien.Pose(p=[place_x, place_y, z_lift], q=tcp_q)
            move_pose(target)

        # Lower onto disc
        if not episode_done["value"]:
            target = sapien.Pose(p=[place_x, place_y, z_place], q=tcp_q)
            move_pose_retry(target)

        # Release
        if not episode_done["value"]:
            planner.gripper_state = 1
            hold_steps(args.gripper_open_steps)

        # Lift away
        if not episode_done["value"]:
            target = sapien.Pose(p=[place_x, place_y, z_lift], q=tcp_q)
            move_pose(target)

        # Settle
        if not episode_done["value"]:
            hold_steps(args.settle_steps)

    # ---------------------------------------------------------------
    # Phase 1: Pick and place all cubes onto the disc
    # ---------------------------------------------------------------
    print("Phase 1: Moving cubes to disc...")
    for i in range(n_cubes):
        if episode_done["value"]:
            break
        pick_and_place_cube(i)

        # Check status
        info_now = env_u.evaluate()
        n_on = _to_int_scalar(info_now["n_on_disc"][0])
        flash_active = _to_bool_scalar(info_now["flash_active"][0])
        flash_triggered = _to_bool_scalar(info_now["flash_triggered"][0])
        print(f"  -> cubes on disc: {n_on}/{n_cubes}, flash_triggered={flash_triggered}, flash_active={flash_active}")

    # ---------------------------------------------------------------
    # Phase 2: Press the correct button
    # ---------------------------------------------------------------
    if not episode_done["value"]:
        target_btn = flash_color
        btn_x, btn_y = buttons_xy[target_btn]
        print(f"\nPhase 2: Pressing {color_names[target_btn]} button at [{btn_x:.3f}, {btn_y:.3f}]...")

        # Close gripper before pressing
        planner.gripper_state = -1
        hold_steps(3)

        # Move above button
        z_btn_approach = max(button_top_z + 0.10, z_lift - 0.02)
        z_btn_press = button_top_z - float(env_u.BUTTON_CAP_TRAVEL) * 0.6

        if not episode_done["value"]:
            target = sapien.Pose(p=[btn_x, btn_y, z_btn_approach], q=tcp_q)
            move_pose_retry(target)

        # Press down
        if not episode_done["value"]:
            target = sapien.Pose(p=[btn_x, btn_y, z_btn_press], q=tcp_q)
            move_pose_retry(target)

        # Hold press briefly
        if not episode_done["value"]:
            hold_steps(5)

        # Lift off
        if not episode_done["value"]:
            target = sapien.Pose(p=[btn_x, btn_y, z_btn_approach], q=tcp_q)
            move_pose(target)

    # ---------------------------------------------------------------
    # Report results
    # ---------------------------------------------------------------
    info_final = env_u.evaluate()
    elapsed_steps = get_elapsed()

    success = _to_bool_scalar(info_final["success"])
    failed = _to_bool_scalar(info_final["failed"])
    n_on_disc = _to_int_scalar(info_final["n_on_disc"][0])
    all_on_disc = _to_bool_scalar(info_final["all_on_disc"][0])
    pressed_btn = _to_int_scalar(info_final["pressed_button"][0])

    print()
    print("=== Results ===")
    print(f"env_id: {env_id}")
    print(f"n_cubes: {n_cubes}")
    print(f"elapsed_steps: {elapsed_steps} / {max_env_steps}")
    print(f"cubes_on_disc: {n_on_disc}/{n_cubes}")
    print(f"all_on_disc: {all_on_disc}")
    print(f"flash_color: {flash_color} ({color_names[flash_color]})")
    print(f"pressed_button: {pressed_btn} ({color_names[pressed_btn] if pressed_btn >= 0 else 'NONE'})")
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
