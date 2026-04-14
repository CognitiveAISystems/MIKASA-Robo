#!/usr/bin/env python3
"""Build benchmark demo videos: general render + top/wrist side panel.

For each selected environment, the script runs one episode using either:
1) PPO checkpoint rollout, or
2) motion-planning script.

Then it creates a final mp4 with layout:
    [ general render | top (upper-right) / wrist (lower-right) ]

Typical usage:
    python utils/prepare_benchmark_demo_videos.py --overwrite
    python utils/prepare_benchmark_demo_videos.py --tasks RememberColor3-VLA-v0,RememberColor5-VLA-v0
    python utils/prepare_benchmark_demo_videos.py --policy-preference ppo_first --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper

import mikasa_robo_suite.vla.memory_envs  # noqa: F401  (register env ids)
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.dataset_collectors.get_mikasa_robo_datasets import (
    env_info,
    get_list_of_all_checkpoints_available,
)

DEFAULT_TASKS_FROM_DATA_DIR = Path("data_mikasa_robo/data_npz")
DEFAULT_OUTPUT_DIR = Path("videos/benchmark_demos")

# Motion-planning env set currently supported by existing planner scripts.
BATTERY_LEVELS = (3, 6, 9, 12, 15)
MOTION_DEFAULT_ENVS = {
    *(f"BatteriesCheckerEasy-{n}-VLA-v0" for n in BATTERY_LEVELS),
    *(f"BatteriesCheckerHard-{n}-VLA-v0" for n in BATTERY_LEVELS),
    "BlinkCountButtonPressEasy-VLA-v0",
    "BlinkCountButtonPressMedium-VLA-v0",
    "BlinkCountButtonPressHard-VLA-v0",
}


@dataclass
class EpisodeStreams:
    general_frames: List[np.ndarray]
    top_frames: List[np.ndarray]
    wrist_frames: List[np.ndarray]
    fps: float
    success: bool
    source_meta: Dict[str, Any]


@dataclass
class TaskResult:
    env_id: str
    policy: str
    final_video: str
    frames_written: int
    fps: float
    success: bool
    skipped: bool = False
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark demo videos with layout: general | (top over wrist)."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated env ids. Empty => auto-discover.",
    )
    parser.add_argument(
        "--tasks-from-data-dir",
        type=Path,
        default=DEFAULT_TASKS_FROM_DATA_DIR,
        help="Default source of benchmark env list (directory with per-task subfolders).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store composed videos and metadata.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("."),
        help="Root directory for oracle PPO checkpoints discovery.",
    )
    parser.add_argument(
        "--policy-preference",
        type=str,
        choices=("motion_first", "ppo_first"),
        default="motion_first",
        help="When both are available for an env, choose motion or PPO first.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Seed for rollout.")
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="gpu",
        choices=("gpu", "cpu"),
        help="ManiSkill sim backend for PPO rollouts.",
    )
    parser.add_argument(
        "--ppo-max-steps",
        type=int,
        default=None,
        help="Optional max steps override for PPO episode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing final videos.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rollout plan and exit.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep raw planner outputs in output-dir/intermediate.",
    )
    return parser.parse_args()


def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert RGB image (float/uint8) to contiguous uint8 RGB."""
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    x = x.astype(np.float32, copy=False)
    max_val = float(np.nanmax(x)) if x.size > 0 else 0.0
    if max_val <= 1.0 + 1e-5:
        x = x * 255.0
    x = np.clip(x, 0.0, 255.0).astype(np.uint8, copy=False)
    return np.ascontiguousarray(x)


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_bool_scalar(x: Any) -> bool:
    arr = to_numpy(x).reshape(-1)
    if arr.size == 0:
        return False
    return bool(arr[0])


def extract_single_frame(x: Any) -> np.ndarray:
    """Extract single RGB frame from batched/unbatched tensor/array."""
    arr = to_numpy(x)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected frame shape [H,W,3], got {arr.shape}")
    return to_uint8_rgb(arr)


def get_mp4_fps(mp4_path: Path, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6 or np.isnan(fps):
        return fallback
    return fps


def read_mp4_frames(mp4_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for reading: {mp4_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return frames


def write_mp4_frames(mp4_path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {mp4_path}")
    first = to_uint8_rgb(frames[0])
    h, w = first.shape[:2]
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {mp4_path}")
    try:
        for frame in frames:
            rgb = to_uint8_rgb(frame)
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def pick_camera_keys(camera_keys: Sequence[str]) -> Tuple[str, str]:
    if not camera_keys:
        raise ValueError("No cameras found.")
    keys = list(camera_keys)
    lowered = {k: k.lower() for k in keys}

    def pick(candidates: Sequence[str]) -> Optional[str]:
        for token in candidates:
            for key in keys:
                if token in lowered[key]:
                    return key
        return None

    wrist_key = pick(("wrist", "hand", "gripper"))
    top_key = pick(("top", "base", "front", "overhead"))

    if top_key is None:
        top_key = next((k for k in keys if k != wrist_key), keys[0])
    if wrist_key is None:
        wrist_key = next((k for k in keys if k != top_key), top_key)

    return top_key, wrist_key


def compose_frames(
    general_frames: Sequence[np.ndarray],
    top_frames: Sequence[np.ndarray],
    wrist_frames: Sequence[np.ndarray],
) -> List[np.ndarray]:
    n = min(len(general_frames), len(top_frames), len(wrist_frames))
    if n <= 0:
        raise ValueError(
            f"Cannot compose empty streams: "
            f"general={len(general_frames)}, top={len(top_frames)}, wrist={len(wrist_frames)}"
        )

    first_general = to_uint8_rgb(general_frames[0])
    g_h, g_w = first_general.shape[:2]
    side_w = max(1, g_w // 2)
    top_h = g_h // 2
    wrist_h = g_h - top_h

    out: List[np.ndarray] = []
    for idx in range(n):
        g = to_uint8_rgb(general_frames[idx])
        if g.shape[:2] != (g_h, g_w):
            g = cv2.resize(g, (g_w, g_h), interpolation=cv2.INTER_AREA)

        top = cv2.resize(to_uint8_rgb(top_frames[idx]), (side_w, top_h), interpolation=cv2.INTER_AREA)
        wrist = cv2.resize(
            to_uint8_rgb(wrist_frames[idx]),
            (side_w, wrist_h),
            interpolation=cv2.INTER_AREA,
        )

        cv2.putText(top, "top", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(wrist, "wrist", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        right = np.concatenate([top, wrist], axis=0)
        composed = np.concatenate([g, right], axis=1)
        out.append(np.ascontiguousarray(composed))
    return out


def motion_script_for_env(env_id: str) -> Optional[Path]:
    root = Path("mikasa_robo_suite/vla/utils/motion_planning")
    if env_id.startswith("BatteriesCheckerEasy-"):
        return root / "motion_planning_batteries_checker_easy.py"
    if env_id.startswith("BatteriesCheckerHard-"):
        return root / "motion_planning_batteries_checker_hard.py"
    if env_id.startswith("BlinkCountButtonPress"):
        return root / "motion_planning_blink_count_button_press.py"
    return None


def is_motion_supported(env_id: str) -> bool:
    return motion_script_for_env(env_id) is not None


def resolve_latest_checkpoints(ckpt_dir: Path) -> Dict[str, Path]:
    raw = get_list_of_all_checkpoints_available(ckpt_dir=str(ckpt_dir))
    best: Dict[str, Path] = {}
    for env_id, ckpt_str in raw:
        ckpt = Path(ckpt_str)
        if not ckpt.exists():
            continue
        if env_id not in best:
            best[env_id] = ckpt
            continue
        if ckpt.stat().st_mtime > best[env_id].stat().st_mtime:
            best[env_id] = ckpt
    return best


def discover_tasks(tasks_csv: str, tasks_from_data_dir: Path, checkpoint_map: Dict[str, Path]) -> List[str]:
    if tasks_csv.strip():
        return [t.strip() for t in tasks_csv.split(",") if t.strip()]

    tasks: set[str] = set()
    if tasks_from_data_dir.exists():
        tasks.update(p.name for p in tasks_from_data_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))
    tasks.update(checkpoint_map.keys())
    tasks.update(MOTION_DEFAULT_ENVS)
    return sorted(tasks)


def choose_policy(
    env_id: str,
    checkpoint_map: Dict[str, Path],
    policy_preference: str,
) -> str:
    has_motion = is_motion_supported(env_id)
    has_ppo = env_id in checkpoint_map

    if policy_preference == "motion_first":
        if has_motion:
            return "motion_planning"
        if has_ppo:
            return "ppo"
    elif policy_preference == "ppo_first":
        if has_ppo:
            return "ppo"
        if has_motion:
            return "motion_planning"

    raise ValueError(f"No available policy for {env_id}. motion_supported={has_motion}, ppo_checkpoint_found={has_ppo}")


def drop_video_text_wrappers(
    wrappers_list: Sequence[Tuple[Any, Dict[str, Any]]],
) -> List[Tuple[Any, Dict[str, Any]]]:
    """Remove wrappers that draw step/reward/reward_dict overlays."""
    blocked = {"RenderStepInfoWrapper", "RenderRewardInfoWrapper", "DebugRewardWrapper"}
    filtered: List[Tuple[Any, Dict[str, Any]]] = []
    for wrapper_class, wrapper_kwargs in wrappers_list:
        name = getattr(wrapper_class, "__name__", str(wrapper_class))
        if name in blocked:
            continue
        filtered.append((wrapper_class, wrapper_kwargs))
    return filtered


def run_motion_planning_episode(
    env_id: str,
    seed: int,
    run_dir: Path,
) -> EpisodeStreams:
    script = motion_script_for_env(env_id)
    if script is None:
        raise ValueError(f"Motion-planning script not found for env_id={env_id}")
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(script),
        "--env-id",
        env_id,
        "--seed",
        str(seed),
        "--save-video",
        "1",
        "--overlay-info",
        "0",
        "--save-trajectory",
        "1",
        "--trajectory-dir",
        str(run_dir),
        "--trajectory-name",
        "trajectory",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(f"Motion planner failed for {env_id}:\n{tail}")

    mp4_candidates = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4_candidates:
        raise FileNotFoundError(f"No mp4 was produced in {run_dir}")
    general_mp4 = mp4_candidates[-1]

    h5_candidates = sorted(run_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not h5_candidates:
        raise FileNotFoundError(f"No trajectory h5 was produced in {run_dir}")
    trajectory_h5 = h5_candidates[-1]

    general_frames = read_mp4_frames(general_mp4)
    fps = get_mp4_fps(general_mp4, fallback=30.0)
    top_frames, wrist_frames, top_key, wrist_key, success = load_camera_streams_from_h5(trajectory_h5)

    return EpisodeStreams(
        general_frames=general_frames,
        top_frames=top_frames,
        wrist_frames=wrist_frames,
        fps=fps,
        success=success,
        source_meta={
            "policy": "motion_planning",
            "script": str(script),
            "trajectory_h5": str(trajectory_h5),
            "general_mp4": str(general_mp4),
            "top_camera_key": top_key,
            "wrist_camera_key": wrist_key,
        },
    )


def load_camera_streams_from_h5(
    h5_path: Path,
) -> Tuple[List[np.ndarray], List[np.ndarray], str, str, bool]:
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {h5_path}")
        traj = traj_keys[0]
        obs_group = f[f"{traj}/obs"]

        # Format A: raw camera tree
        # traj_0/obs/sensor_data/<camera>/rgb
        if "sensor_data" in obs_group:
            sensor_group = obs_group["sensor_data"]
            camera_keys = list(sensor_group.keys())
            top_key, wrist_key = pick_camera_keys(camera_keys)

            top_arr = np.asarray(sensor_group[top_key]["rgb"])
            wrist_arr = np.asarray(sensor_group[wrist_key]["rgb"])
        # Format B: flattened RGB tensor from FlattenRGBDObservationWrapper
        # traj_0/obs/rgb with channels concatenated across cameras, e.g. [..., 6]
        elif "rgb" in obs_group:
            rgb_arr = np.asarray(obs_group["rgb"])
            if rgb_arr.ndim != 4 or rgb_arr.shape[-1] < 3:
                raise ValueError(f"Unexpected flattened rgb shape in {h5_path}: {rgb_arr.shape}")

            if rgb_arr.shape[-1] >= 6:
                top_arr = rgb_arr[..., :3]
                wrist_arr = rgb_arr[..., 3:6]
                top_key = "flattened_rgb_cam0"
                wrist_key = "flattened_rgb_cam1"
            else:
                top_arr = rgb_arr[..., :3]
                wrist_arr = rgb_arr[..., :3]
                top_key = "flattened_rgb"
                wrist_key = "flattened_rgb"
        else:
            raise ValueError(
                f"Unsupported trajectory obs layout in {h5_path}: "
                f"expected 'sensor_data' or 'rgb', got keys={list(obs_group.keys())}"
            )

        success_arr = np.asarray(f[f"{traj}/success"]) if f"{traj}/success" in f else np.array([], dtype=np.bool_)

    top_frames = [to_uint8_rgb(top_arr[i]) for i in range(top_arr.shape[0])]
    wrist_frames = [to_uint8_rgb(wrist_arr[i]) for i in range(wrist_arr.shape[0])]
    success = bool(success_arr.any()) if success_arr.size > 0 else False
    return top_frames, wrist_frames, top_key, wrist_key, success


def run_ppo_episode(
    env_id: str,
    checkpoint_path: Path,
    seed: int,
    sim_backend: str,
    ppo_max_steps: Optional[int],
) -> EpisodeStreams:
    # CPU inference is sufficient for one demo episode and avoids CUDA-runtime
    # issues on machines where driver/runtime availability is inconsistent.
    device = torch.device("cpu")

    wrappers_list, env_timeout = env_info(env_id)
    wrappers_list = drop_video_text_wrappers(wrappers_list)
    max_steps = int(ppo_max_steps) if ppo_max_steps is not None else int(env_timeout)

    chosen_sim_backend = sim_backend
    if chosen_sim_backend == "gpu" and not torch.cuda.is_available():
        print(f"[warn] CUDA is unavailable for {env_id}; fallback sim_backend='cpu' for PPO rollout.")
        chosen_sim_backend = "cpu"

    env_kwargs_state = dict(
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sim_backend=chosen_sim_backend,
        reward_mode="normalized_dense",
    )
    env_kwargs_rgb = dict(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        sim_backend=chosen_sim_backend,
        reward_mode="normalized_dense",
    )

    env_state = gym.make(env_id, num_envs=1, **env_kwargs_state)
    env_rgb = gym.make(env_id, num_envs=1, **env_kwargs_rgb)

    try:
        for wrapper_class, wrapper_kwargs in wrappers_list:
            env_state = wrapper_class(env_state, **wrapper_kwargs)
            env_rgb = wrapper_class(env_rgb, **wrapper_kwargs)

        env_state = FlattenRGBDObservationWrapper(
            env_state,
            rgb=False,
            depth=False,
            state=True,
            oracle=False,
            joints=False,
        )
        if isinstance(env_state.action_space, gym.spaces.Dict):
            env_state = FlattenActionSpaceWrapper(env_state)
        if isinstance(env_rgb.action_space, gym.spaces.Dict):
            env_rgb = FlattenActionSpaceWrapper(env_rgb)

        agent = AgentStateOnly(env_state).to(device)
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
        agent.eval()

        obs_state, _ = env_state.reset(seed=[seed])
        obs_rgb, info_rgb = env_rgb.reset(seed=[seed])
        del info_rgb

        camera_keys = list(obs_rgb["sensor_data"].keys())
        top_key, wrist_key = pick_camera_keys(camera_keys)

        general_frames: List[np.ndarray] = []
        top_frames: List[np.ndarray] = []
        wrist_frames: List[np.ndarray] = []

        episode_success = False
        for _ in range(max_steps):
            general_render = env_rgb.render()
            general_frames.append(extract_single_frame(general_render))

            top_frames.append(extract_single_frame(obs_rgb["sensor_data"][top_key]["rgb"]))
            wrist_frames.append(extract_single_frame(obs_rgb["sensor_data"][wrist_key]["rgb"]))

            with torch.no_grad():
                obs_state_dev: Dict[str, torch.Tensor] = {}
                for k, v in obs_state.items():
                    if torch.is_tensor(v):
                        obs_state_dev[k] = v.to(device)
                    else:
                        obs_state_dev[k] = torch.as_tensor(v, device=device)
                action = agent.get_action(obs_state_dev, deterministic=True)

            obs_state, _, _, _, _ = env_state.step(action)
            obs_rgb, _, term_rgb, trunc_rgb, info_rgb = env_rgb.step(action)

            success_now = to_bool_scalar(info_rgb.get("success", False))
            done_now = success_now or to_bool_scalar(term_rgb) or to_bool_scalar(trunc_rgb)
            episode_success = episode_success or success_now
            if done_now:
                break

        return EpisodeStreams(
            general_frames=general_frames,
            top_frames=top_frames,
            wrist_frames=wrist_frames,
            fps=30.0,
            success=episode_success,
            source_meta={
                "policy": "ppo",
                "checkpoint": str(checkpoint_path),
                "top_camera_key": top_key,
                "wrist_camera_key": wrist_key,
                "max_steps": max_steps,
                "sim_backend": chosen_sim_backend,
            },
        )
    finally:
        env_state.close()
        env_rgb.close()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_map = resolve_latest_checkpoints(args.ckpt_dir)
    tasks = discover_tasks(args.tasks, args.tasks_from_data_dir, checkpoint_map)
    if not tasks:
        raise RuntimeError("No tasks discovered. Provide --tasks explicitly.")

    results: List[TaskResult] = []
    plan: List[Tuple[str, str]] = []
    for env_id in tasks:
        try:
            policy = choose_policy(env_id, checkpoint_map, args.policy_preference)
            plan.append((env_id, policy))
        except ValueError as exc:
            print(f"[skip] {env_id}: {exc}")
            results.append(
                TaskResult(
                    env_id=env_id,
                    policy="unresolved",
                    final_video=str((final_dir / f"{env_id.replace('/', '_')}.mp4")),
                    frames_written=0,
                    fps=0.0,
                    success=False,
                    skipped=True,
                    error=str(exc),
                )
            )

    print("Planned tasks:")
    for env_id, policy in plan:
        if policy == "ppo":
            ckpt = checkpoint_map[env_id]
            print(f"  - {env_id}: PPO ({ckpt})")
        else:
            print(f"  - {env_id}: motion_planning")

    if args.dry_run:
        print("\nDry-run mode: no rollouts executed.")
        return

    for env_id, policy in plan:
        safe_name = env_id.replace("/", "_")
        final_video_path = final_dir / f"{safe_name}.mp4"

        if final_video_path.exists() and not args.overwrite:
            print(f"[skip] {env_id}: {final_video_path} already exists")
            results.append(
                TaskResult(
                    env_id=env_id,
                    policy=policy,
                    final_video=str(final_video_path),
                    frames_written=0,
                    fps=0.0,
                    success=False,
                    skipped=True,
                )
            )
            continue

        print(f"[run] {env_id} ({policy})")
        try:
            if policy == "motion_planning":
                if args.keep_intermediate:
                    run_dir = output_dir / "intermediate" / safe_name / "motion"
                    if run_dir.exists() and args.overwrite:
                        shutil.rmtree(run_dir)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    streams = run_motion_planning_episode(
                        env_id=env_id,
                        seed=int(args.seed),
                        run_dir=run_dir,
                    )
                else:
                    with tempfile.TemporaryDirectory(prefix=f"demo_{safe_name}_") as td:
                        streams = run_motion_planning_episode(
                            env_id=env_id,
                            seed=int(args.seed),
                            run_dir=Path(td),
                        )
            else:
                ckpt = checkpoint_map.get(env_id)
                if ckpt is None:
                    raise ValueError(f"PPO checkpoint not found for {env_id}")
                streams = run_ppo_episode(
                    env_id=env_id,
                    checkpoint_path=ckpt,
                    seed=int(args.seed),
                    sim_backend=args.sim_backend,
                    ppo_max_steps=args.ppo_max_steps,
                )

            composed_frames = compose_frames(
                general_frames=streams.general_frames,
                top_frames=streams.top_frames,
                wrist_frames=streams.wrist_frames,
            )
            write_mp4_frames(final_video_path, composed_frames, fps=streams.fps)

            task_meta_path = output_dir / "metadata" / f"{safe_name}.json"
            task_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(task_meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "env_id": env_id,
                        "policy": policy,
                        "final_video": str(final_video_path),
                        "frames_written": len(composed_frames),
                        "fps": float(streams.fps),
                        "success": bool(streams.success),
                        "source_meta": streams.source_meta,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            print(
                f"[ok] {env_id}: {final_video_path} "
                f"(frames={len(composed_frames)}, fps={streams.fps:.2f}, success={streams.success})"
            )
            results.append(
                TaskResult(
                    env_id=env_id,
                    policy=policy,
                    final_video=str(final_video_path),
                    frames_written=len(composed_frames),
                    fps=float(streams.fps),
                    success=bool(streams.success),
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {env_id}: {exc}")
            results.append(
                TaskResult(
                    env_id=env_id,
                    policy=policy,
                    final_video=str(final_video_path),
                    frames_written=0,
                    fps=0.0,
                    success=False,
                    error=str(exc),
                )
            )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    num_ok = sum(1 for r in results if (not r.skipped and r.error is None))
    num_err = sum(1 for r in results if r.error is not None)
    num_skip = sum(1 for r in results if r.skipped)
    print(f"\nDone. success={num_ok}, skipped={num_skip}, errors={num_err}. Summary: {summary_path}")


if __name__ == "__main__":
    main()
