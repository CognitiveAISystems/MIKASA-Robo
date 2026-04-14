import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from mikasa_robo_suite.vla.dataset_collectors import (
    get_mikasa_robo_datasets_motion_planning as mp_collector,
)


def _to_scalar(x, default=None):
    if x is None:
        return default
    arr = np.asarray(mp_collector._to_numpy(x)).reshape(-1)
    if arr.size == 0:
        return default
    v = arr[0]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    return v.item() if hasattr(v, "item") else v


def _print_rollout_debug(info: dict):
    if not isinstance(info, dict):
        print("[debug] final info is not a dict")
        return
    keys = [
        "success",
        "checked_count",
        "active_battery_count",
        "found_working_count",
        "target_working_count",
        "all_checked",
        "action_stage",
        "new_insert_event",
        "new_return_event",
        "new_confirm_event",
    ]
    print("[debug] final rollout info:")
    for k in keys:
        if k in info:
            print(f"  - {k}: {_to_scalar(info[k])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Debug one BatteriesCheckerHard seed with the exact collector pipeline:\n"
            "planner(pd_joint_pos) -> ManiSkill replay(pd_ee_delta_pose) -> "
            "pd_ee rollout validation + video."
        )
    )
    parser.add_argument("--env-id", type=str, default="BatteriesCheckerHard-3-VLA-v0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store debug artifacts/video. Defaults to ./videos/<env>/collector_debug_seed_<seed>",
    )
    parser.add_argument(
        "--planner-save-video",
        type=int,
        default=0,
        help="Whether to save video in the planner stage (0/1).",
    )
    parser.add_argument(
        "--replay-video-fps",
        type=int,
        default=20,
        help="FPS for final pd_ee rollout debug video.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = args.env_id
    if not env_id.startswith("BatteriesCheckerHard-"):
        raise ValueError(f"This debug script supports only hard batteries envs. Got env_id={env_id!r}")

    mp_collector._configure_runtime_warning_filters()
    mp_collector._maybe_configure_vulkan_icd_env()
    from mani_skill.utils.wrappers.record import RecordEpisode

    project_root = Path(__file__).resolve().parents[4]
    planner_script = mp_collector.planner_script_for_env(env_id)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("videos") / env_id / f"collector_debug_seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[debug] env_id={env_id}")
    print(f"[debug] seed={args.seed}")
    print(f"[debug] planner_script={planner_script}")
    print(f"[debug] output_dir={output_dir}")

    trajectory_name = "trajectory"
    planner_cmd = [
        sys.executable,
        str(planner_script),
        "--env-id",
        env_id,
        "--seed",
        str(args.seed),
        "--save-trajectory",
        "1",
        "--save-video",
        str(int(args.planner_save_video)),
        "--trajectory-dir",
        str(artifacts_dir),
        "--trajectory-name",
        trajectory_name,
    ]
    env = dict(**os.environ)
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{prev_pythonpath}" if prev_pythonpath else str(project_root)
    mp_collector._configure_subprocess_env(env)

    ok, msg = mp_collector._run_subprocess(cmd=planner_cmd, cwd=project_root, env=env, tag="planner")
    if not ok:
        raise RuntimeError(f"[debug] planner failed:\n{msg}")

    raw_h5, raw_json = mp_collector._resolve_raw_trajectory_paths(
        run_dir=artifacts_dir, trajectory_name=trajectory_name
    )
    if raw_h5 is None:
        raise RuntimeError(f"[debug] planner produced no .h5 in {artifacts_dir}")
    if raw_json is None or not raw_json.exists():
        raise RuntimeError(f"[debug] missing planner json for {raw_h5}")
    print(f"[debug] raw_h5={raw_h5}")

    replay_h5, replay_json, replay_err = mp_collector._run_replay_trajectory(
        raw_h5=raw_h5,
        run_dir=artifacts_dir,
        project_root=project_root,
        use_discard_timeout=not mp_collector._is_batteries_env(env_id),
    )
    if replay_h5 is None:
        raise RuntimeError(f"[debug] replay failed: {replay_err}")
    print(f"[debug] replay_h5={replay_h5}")

    actions = mp_collector._load_last_traj_actions(replay_h5)
    print(f"[debug] replayed actions shape={actions.shape}")

    seed_candidates = [
        replay_json if replay_json is not None else Path(""),
        artifacts_dir / f"{trajectory_name}_meta.json",
        raw_json,
    ]
    episode_seed = int(args.seed)
    for seed_json in seed_candidates:
        if seed_json and seed_json.exists():
            parsed = mp_collector._extract_episode_seed_from_json(seed_json)
            if parsed is not None:
                episode_seed = int(parsed)
                break
    print(f"[debug] episode_seed={episode_seed}")

    rollout_env, continue_after_success = mp_collector._create_pd_ee_rollout_env(env_id)
    max_env_steps = int(gym.spec(env_id).max_episode_steps)
    rollout_env = RecordEpisode(
        rollout_env,
        output_dir=str(output_dir / "pd_ee_rollout"),
        save_trajectory=False,
        save_video=True,
        info_on_video=False,
        max_steps_per_video=max_env_steps,
        video_fps=int(args.replay_video_fps),
        source_type="motionplanning_debug",
        source_desc="collector-style pd_ee replay debug",
    )

    obs, info = rollout_env.reset(seed=episode_seed)
    mp_collector._validate_flatten_obs(obs)

    success_once = False
    final_info = info if isinstance(info, dict) else {}
    total_steps = 0
    done = False
    for action in actions:
        action_step = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = rollout_env.step(action_step)
        final_info = info if isinstance(info, dict) else final_info

        success_step = mp_collector._to_bool_scalar(info.get("success", False) if isinstance(info, dict) else False)
        terminated_step = mp_collector._to_bool_scalar(terminated)
        truncated_step = mp_collector._to_bool_scalar(truncated)
        success_once = success_once or success_step
        total_steps += 1

        if continue_after_success:
            done = terminated_step or truncated_step
        else:
            done = success_step or terminated_step or truncated_step
        if done:
            break

    rollout_env.close()

    result = {
        "env_id": env_id,
        "seed": int(args.seed),
        "episode_seed": int(episode_seed),
        "continue_after_success": bool(continue_after_success),
        "actions_len": int(actions.shape[0]),
        "rollout_steps": int(total_steps),
        "done": bool(done),
        "success_once": bool(success_once),
        "artifacts_dir": str(artifacts_dir),
        "raw_h5": str(raw_h5),
        "replay_h5": str(replay_h5),
    }
    with open(output_dir / "debug_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[debug] done={done}, success_once={success_once}, rollout_steps={total_steps}/{actions.shape[0]}")
    _print_rollout_debug(final_info)
    print(f"[debug] summary saved: {output_dir / 'debug_summary.json'}")
    print(f"[debug] video dir: {output_dir / 'pd_ee_rollout'}")


if __name__ == "__main__":
    main()
