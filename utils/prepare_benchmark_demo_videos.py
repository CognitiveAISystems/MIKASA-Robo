#!/usr/bin/env python3
"""Generate benchmark demo media for MIKASA-Robo VLA tasks.

For every task (by default from `mikasa_robo_vla_envs.csv`) this script:
1. Runs oracle rollout (PPO checkpoint and/or motion planning),
2. Retries with different seeds until success,
3. Composes output frame as: [general render | top over wrist],
4. Exports both:
   - web-friendly mp4 (H.264, yuv420p, faststart),
   - gif.

Notes:
- PPO path keeps env-specific wrappers, forcibly keeps step rendering, and removes reward render/debug wrappers.
- Motion-planning path uses each planner's `--overlay-info` flag
  (default `1` to keep wrapper-driven step/env overlays), while reward overlay is disabled via env flag.
- This script does not draw any custom step text; step is expected from `RenderStepInfoWrapper`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from mikasa_robo_suite.vla.utils.wrappers import RenderStepInfoWrapper

DEFAULT_TASKS_CSV = Path("mikasa_robo_vla_envs.csv")
DEFAULT_OUTPUT_DIR = Path("videos/benchmark_demos")
MOTION_ROOT = Path("mikasa_robo_suite/vla/utils/motion_planning")


@dataclass
class TaskSpec:
    env_id: str
    source_hint: Optional[str]
    configured: bool
    language_instruction: str


@dataclass
class EpisodeStreams:
    general_frames: List[np.ndarray]
    top_frames: List[np.ndarray]
    wrist_frames: List[np.ndarray]
    fps: float
    success: bool
    source_meta: Dict[str, Any]


@dataclass
class MotionEpisodeArtifacts:
    general_mp4: Path
    trajectory_h5: Path
    fps: float
    success: bool
    top_camera_key: str
    wrist_camera_key: str
    source_meta: Dict[str, Any]


@dataclass
class TaskResult:
    env_id: str
    source_hint: Optional[str]
    policy: str
    final_mp4: str
    final_gif: str
    frames_written: int
    fps: float
    success: bool
    attempts_tried: int
    seed_used: Optional[int]
    skipped: bool = False
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark demo media (H264 mp4 + gif) with layout: general | (top over wrist)."
    )
    parser.add_argument(
        "--tasks-csv",
        type=Path,
        default=DEFAULT_TASKS_CSV,
        help="CSV file with benchmark tasks, e.g. mikasa_robo_vla_envs.csv.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Optional comma-separated env ids. If empty, use tasks from --tasks-csv.",
    )
    parser.add_argument(
        "--skip-tasks",
        type=str,
        default="",
        help="Optional comma-separated env ids to skip.",
    )
    parser.add_argument(
        "--include-unconfigured",
        action="store_true",
        help="Include rows where CSV column 'Configured' is false.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store final media and metadata.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("."),
        help="Root directory used to discover oracle PPO checkpoints.",
    )
    parser.add_argument(
        "--policy-preference",
        type=str,
        choices=("source_first", "motion_first", "ppo_first"),
        default="source_first",
        help="How to prioritize policy type when both are available.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base seed for rollout attempts.")
    parser.add_argument(
        "--shellgame-seed",
        type=int,
        default=2026,
        help="Base seed used for ShellGame* tasks (attempt i uses shellgame-seed+i).",
    )
    parser.add_argument(
        "--max-attempts-per-task",
        type=int,
        default=8,
        help="Try seeds [seed, seed+1, ...] until success.",
    )
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
        "--motion-overlay-info",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Value passed to planner --overlay-info (0/1). "
            "Use 1 to enable wrapper-driven step/env overlays. "
            "Reward overlay is disabled by env flag in this script."
        ),
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=12,
        help="Target FPS for GIF export.",
    )
    parser.add_argument(
        "--gif-max-width",
        type=int,
        default=0,
        help="If >0, downscale GIF width to this value while preserving aspect ratio.",
    )
    parser.add_argument(
        "--h264-crf",
        type=int,
        default=20,
        help="H.264 quality factor (lower = better quality, larger files).",
    )
    parser.add_argument(
        "--h264-preset",
        type=str,
        default="medium",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
        help="H.264 encoding speed/quality preset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing final mp4/gif.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan and exit.",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate planner outputs in output-dir/intermediate.",
    )
    return parser.parse_args()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def parse_env_id_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_env_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_bool_scalar(x: Any) -> bool:
    arr = to_numpy(x).reshape(-1)
    if arr.size == 0:
        return False
    return bool(arr[0])


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


def ensure_hwc_rgb(x: Any) -> np.ndarray:
    """Convert input image tensor/array to [H, W, 3] uint8 RGB."""
    arr = to_numpy(x)

    # Strip singleton leading dims (e.g. [1,H,W,3], [1,1,H,W,3]).
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    # If still batched, take first element.
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims [H,W,C], got {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return to_uint8_rgb(arr)


def extract_single_frame(x: Any) -> np.ndarray:
    return ensure_hwc_rgb(x)


def get_mp4_fps(mp4_path: Path, fallback: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6 or np.isnan(fps):
        return fallback
    return fps


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_overrides:
        env.update({str(k): str(v) for k, v in env_overrides.items()})

    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").splitlines()[-60:])
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


def load_tasks_from_csv(tasks_csv: Path, include_unconfigured: bool) -> List[TaskSpec]:
    if not tasks_csv.exists():
        raise FileNotFoundError(f"Task CSV not found: {tasks_csv}")

    tasks: List[TaskSpec] = []
    with open(tasks_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            env_id = (row.get("name") or row.get("Name") or row.get("env_id") or "").strip()
            if not env_id:
                continue
            configured = parse_bool(row.get("Configured", True))
            if not configured and not include_unconfigured:
                continue
            source_hint = (row.get("Data Source") or row.get("source") or "").strip().upper() or None
            language_instruction = (row.get("language_instruction") or "").strip()
            tasks.append(
                TaskSpec(
                    env_id=env_id,
                    source_hint=source_hint,
                    configured=configured,
                    language_instruction=language_instruction,
                )
            )
    return tasks


def resolve_tasks(args: argparse.Namespace) -> List[TaskSpec]:
    csv_tasks = load_tasks_from_csv(args.tasks_csv, include_unconfigured=args.include_unconfigured)
    skip_envs_exact = set(parse_env_id_list(args.skip_tasks))
    skip_envs_norm = [normalize_env_token(x) for x in skip_envs_exact if normalize_env_token(x)]

    def is_skipped(env_id: str) -> bool:
        if env_id in skip_envs_exact:
            return True
        env_norm = normalize_env_token(env_id)
        if env_norm in skip_envs_norm:
            return True
        return any(tok in env_norm for tok in skip_envs_norm)

    if not args.tasks.strip():
        if not skip_envs_exact:
            return csv_tasks
        return [t for t in csv_tasks if not is_skipped(t.env_id)]

    requested = parse_env_id_list(args.tasks)
    by_name = {t.env_id: t for t in csv_tasks}
    by_norm = {normalize_env_token(t.env_id): t for t in csv_tasks}
    out: List[TaskSpec] = []
    for token in requested:
        token_norm = normalize_env_token(token)
        candidate: Optional[TaskSpec] = None

        if token in by_name:
            candidate = by_name[token]
        elif token_norm in by_norm:
            candidate = by_norm[token_norm]
        else:
            fuzzy_matches = [task for env_norm, task in by_norm.items() if token_norm and token_norm in env_norm]
            if len(fuzzy_matches) == 1:
                candidate = fuzzy_matches[0]
            elif len(fuzzy_matches) > 1:
                # Prefer shortest normalized id when multiple fuzzy matches exist.
                candidate = sorted(fuzzy_matches, key=lambda t: len(normalize_env_token(t.env_id)))[0]

        if candidate is None:
            candidate = TaskSpec(env_id=token, source_hint=None, configured=True, language_instruction="")

        if is_skipped(candidate.env_id):
            continue
        out.append(candidate)
    return out


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


def motion_script_for_env(env_id: str) -> Optional[Path]:
    if env_id.startswith("BatteriesCheckerEasy-"):
        return MOTION_ROOT / "motion_planning_batteries_checker_easy.py"
    if env_id.startswith("BatteriesCheckerHard-"):
        return MOTION_ROOT / "motion_planning_batteries_checker_hard.py"
    if env_id.startswith("BlinkCountButtonPress"):
        if "-Long-" in env_id:
            return MOTION_ROOT / "motion_planning_blink_count_button_press_long.py"
        return MOTION_ROOT / "motion_planning_blink_count_button_press.py"
    if env_id.startswith(("RememberColor", "RememberShape", "RememberShapeAndColor")) and "-Long-" in env_id:
        return MOTION_ROOT / "motion_planning_remember_long.py"
    if env_id.startswith(("BunchOfColors", "SeqOfColors", "ChainOfColors")):
        return MOTION_ROOT / "motion_planning_memory_capacity_colors.py"
    if env_id.startswith("ShellGameShuffle"):
        return MOTION_ROOT / "motion_planning_shell_game_shuffle.py"
    if env_id.startswith("TraceShapeSeq"):
        return MOTION_ROOT / "motion_planning_trace_shape_seq.py"
    if env_id.startswith("TraceShape"):
        return MOTION_ROOT / "motion_planning_trace_shape.py"
    if env_id.startswith("TimedTransfer"):
        return MOTION_ROOT / "motion_planning_timed_transfer.py"
    if env_id.startswith("GatherAndRecall"):
        return MOTION_ROOT / "motion_planning_gather_and_recall.py"
    if env_id.startswith("ShellGamePick"):
        return MOTION_ROOT / "motion_planning_shell_game_pick.py"
    return None


def has_motion_policy(env_id: str) -> bool:
    script = motion_script_for_env(env_id)
    return script is not None and script.exists()


def policy_candidates(
    env_id: str,
    source_hint: Optional[str],
    checkpoint_map: Dict[str, Path],
    policy_preference: str,
) -> List[str]:
    has_motion = has_motion_policy(env_id)
    has_ppo = env_id in checkpoint_map

    if not has_motion and not has_ppo:
        return []

    if policy_preference == "motion_first":
        order = ["motion_planning", "ppo"]
    elif policy_preference == "ppo_first":
        order = ["ppo", "motion_planning"]
    else:
        # source_first
        if source_hint == "MP":
            order = ["motion_planning", "ppo"]
        elif source_hint == "PPO":
            order = ["ppo", "motion_planning"]
        else:
            order = ["motion_planning", "ppo"]

    out: List[str] = []
    for p in order:
        if p == "motion_planning" and has_motion and p not in out:
            out.append(p)
        if p == "ppo" and has_ppo and p not in out:
            out.append(p)
    return out


def drop_reward_render_wrappers(
    wrappers_list: Sequence[Tuple[Any, Dict[str, Any]]],
) -> Tuple[List[Tuple[Any, Dict[str, Any]]], List[str]]:
    """Drop wrappers that render or expose reward-related debug information.

    This preserves step overlays and env-specific goal/progress overlays while
    removing reward visualizations.
    """

    filtered: List[Tuple[Any, Dict[str, Any]]] = []
    removed_names: List[str] = []
    for wrapper_class, wrapper_kwargs in wrappers_list:
        name = getattr(wrapper_class, "__name__", str(wrapper_class))
        normalized = name.replace("_", "").lower()
        if "reward" in normalized:
            removed_names.append(name)
            continue
        filtered.append((wrapper_class, wrapper_kwargs))
    return filtered, removed_names


def ensure_step_render_wrapper(
    wrappers_list: Sequence[Tuple[Any, Dict[str, Any]]],
) -> Tuple[List[Tuple[Any, Dict[str, Any]]], bool]:
    """Ensure RenderStepInfoWrapper is present exactly once."""
    out = list(wrappers_list)
    has_step = any(
        getattr(wrapper_class, "__name__", str(wrapper_class)).replace("_", "").lower() == "renderstepinfowrapper"
        for wrapper_class, _ in out
    )
    if has_step:
        return out, False
    out.append((RenderStepInfoWrapper, {}))
    return out, True


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


def load_motion_h5_metadata(h5_path: Path) -> Tuple[str, str, bool]:
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        if not traj_keys:
            raise ValueError(f"No traj_* groups found in {h5_path}")
        traj = traj_keys[0]
        obs_group = f[f"{traj}/obs"]

        if "sensor_data" in obs_group:
            camera_keys = list(obs_group["sensor_data"].keys())
            top_key, wrist_key = pick_camera_keys(camera_keys)
        elif "rgb" in obs_group:
            rgb_arr = obs_group["rgb"]
            if rgb_arr.ndim != 4 or rgb_arr.shape[-1] < 3:
                raise ValueError(f"Unexpected flattened rgb shape in {h5_path}: {rgb_arr.shape}")
            if rgb_arr.shape[-1] >= 6:
                top_key, wrist_key = "flattened_rgb_cam0", "flattened_rgb_cam1"
            else:
                top_key, wrist_key = "flattened_rgb", "flattened_rgb"
        else:
            raise ValueError(
                f"Unsupported trajectory obs layout in {h5_path}: "
                f"expected 'sensor_data' or 'rgb', got keys={list(obs_group.keys())}"
            )

        success_arr = np.asarray(f[f"{traj}/success"]) if f"{traj}/success" in f else np.array([], dtype=np.bool_)
        success = bool(success_arr.any()) if success_arr.size > 0 else False
    return top_key, wrist_key, success


def run_motion_planning_episode(
    env_id: str,
    seed: int,
    run_dir: Path,
    overlay_info: int,
) -> MotionEpisodeArtifacts:
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
        str(int(overlay_info)),
        "--save-trajectory",
        "1",
        "--trajectory-dir",
        str(run_dir),
        "--trajectory-name",
        "trajectory",
    ]
    run_cmd(
        cmd,
        cwd=Path.cwd(),
        env_overrides={
            # Keep wrapper-driven step/env overlays while suppressing reward text.
            "MIKASA_DISABLE_REWARD_OVERLAY": "1",
        },
    )

    mp4_candidates = sorted(run_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not mp4_candidates:
        raise FileNotFoundError(f"No mp4 was produced in {run_dir}")
    general_mp4 = mp4_candidates[-1]

    h5_candidates = sorted(run_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if not h5_candidates:
        raise FileNotFoundError(f"No trajectory h5 was produced in {run_dir}")
    trajectory_h5 = h5_candidates[-1]

    fps = get_mp4_fps(general_mp4, fallback=30.0)
    top_key, wrist_key, success = load_motion_h5_metadata(trajectory_h5)

    return MotionEpisodeArtifacts(
        general_mp4=general_mp4,
        trajectory_h5=trajectory_h5,
        fps=fps,
        success=success,
        top_camera_key=top_key,
        wrist_camera_key=wrist_key,
        source_meta={
            "policy": "motion_planning",
            "script": str(script),
            "general_mp4": str(general_mp4),
            "trajectory_h5": str(trajectory_h5),
            "top_camera_key": top_key,
            "wrist_camera_key": wrist_key,
            "overlay_info": int(overlay_info),
        },
    )


def run_ppo_episode(
    env_id: str,
    checkpoint_path: Path,
    seed: int,
    sim_backend: str,
    ppo_max_steps: Optional[int],
) -> EpisodeStreams:
    # CPU inference is enough for a single deterministic demo rollout.
    device = torch.device("cpu")

    try:
        wrappers_list, env_timeout = env_info(env_id)
    except ValueError:
        wrappers_list = []
        spec = gym.spec(env_id)
        env_timeout = int(spec.max_episode_steps)

    wrappers_list, removed_wrappers = drop_reward_render_wrappers(wrappers_list)
    wrappers_list, step_wrapper_added = ensure_step_render_wrapper(wrappers_list)
    wrapper_names = [getattr(wrapper_class, "__name__", str(wrapper_class)) for wrapper_class, _ in wrappers_list]
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
        obs_rgb, _ = env_rgb.reset(seed=[seed])

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
                # Keep terminal frame so the last success-transition step is visible.
                terminal_general = extract_single_frame(env_rgb.render())
                general_frames.append(terminal_general)
                try:
                    terminal_top = extract_single_frame(obs_rgb["sensor_data"][top_key]["rgb"])
                    terminal_wrist = extract_single_frame(obs_rgb["sensor_data"][wrist_key]["rgb"])
                except Exception:  # noqa: BLE001
                    if top_frames and wrist_frames:
                        terminal_top = top_frames[-1]
                        terminal_wrist = wrist_frames[-1]
                    else:
                        terminal_top = terminal_general
                        terminal_wrist = terminal_general
                top_frames.append(terminal_top)
                wrist_frames.append(terminal_wrist)
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
                "reward_wrappers_removed": removed_wrappers,
                "wrappers_applied": wrapper_names,
                "step_wrapper_added": bool(step_wrapper_added),
            },
        )
    finally:
        env_state.close()
        env_rgb.close()


def draw_panel_label(img: np.ndarray, text: str) -> None:
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)


def compose_single_frame(
    general_frame: np.ndarray,
    top_frame: np.ndarray,
    wrist_frame: np.ndarray,
) -> np.ndarray:
    g = to_uint8_rgb(general_frame)
    t = to_uint8_rgb(top_frame)
    w = to_uint8_rgb(wrist_frame)

    g_h, g_w = g.shape[:2]
    side_w = max(1, g_w // 2)
    top_h = g_h // 2
    wrist_h = g_h - top_h

    t = cv2.resize(t, (side_w, top_h), interpolation=cv2.INTER_AREA)
    w = cv2.resize(w, (side_w, wrist_h), interpolation=cv2.INTER_AREA)

    draw_panel_label(t, "top")
    draw_panel_label(w, "wrist")

    right = np.concatenate([t, w], axis=0)
    out = np.concatenate([g, right], axis=1)
    return np.ascontiguousarray(out)


def compose_frames(
    general_frames: Sequence[np.ndarray],
    top_frames: Sequence[np.ndarray],
    wrist_frames: Sequence[np.ndarray],
) -> List[np.ndarray]:
    n_general = len(general_frames)
    n_top = len(top_frames)
    n_wrist = len(wrist_frames)
    n = max(n_general, n_top, n_wrist)
    if n <= 0:
        raise ValueError(f"Cannot compose empty streams: general={n_general}, top={n_top}, wrist={n_wrist}")
    out: List[np.ndarray] = []
    for idx in range(n):
        g_idx = min(idx, n_general - 1)
        t_idx = min(idx, n_top - 1)
        w_idx = min(idx, n_wrist - 1)
        out.append(
            compose_single_frame(
                general_frame=general_frames[g_idx],
                top_frame=top_frames[t_idx],
                wrist_frame=wrist_frames[w_idx],
            )
        )
    return out


def write_raw_mp4_frames(mp4_path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
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


def compose_motion_to_raw_mp4(
    artifacts: MotionEpisodeArtifacts,
    out_raw_mp4: Path,
) -> int:
    cap = cv2.VideoCapture(str(artifacts.general_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open motion planner video: {artifacts.general_mp4}")

    out_raw_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[cv2.VideoWriter] = None
    frames_written = 0
    last_general_rgb: Optional[np.ndarray] = None

    try:
        with h5py.File(artifacts.trajectory_h5, "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
            if not traj_keys:
                raise ValueError(f"No traj_* groups found in {artifacts.trajectory_h5}")
            traj = traj_keys[0]
            obs_group = f[f"{traj}/obs"]

            if "sensor_data" in obs_group:
                sensor_group = obs_group["sensor_data"]
                top_ds = sensor_group[artifacts.top_camera_key]["rgb"]
                wrist_ds = sensor_group[artifacts.wrist_camera_key]["rgb"]
                n_side = int(min(top_ds.shape[0], wrist_ds.shape[0]))

                def get_side(i: int) -> Tuple[np.ndarray, np.ndarray]:
                    return np.asarray(top_ds[i]), np.asarray(wrist_ds[i])

            elif "rgb" in obs_group:
                rgb_ds = obs_group["rgb"]
                if rgb_ds.ndim != 4 or rgb_ds.shape[-1] < 3:
                    raise ValueError(f"Unexpected flattened rgb shape in {artifacts.trajectory_h5}: {rgb_ds.shape}")
                n_side = int(rgb_ds.shape[0])

                if rgb_ds.shape[-1] >= 6:

                    def get_side(i: int) -> Tuple[np.ndarray, np.ndarray]:
                        frame = np.asarray(rgb_ds[i])
                        return frame[..., :3], frame[..., 3:6]

                else:

                    def get_side(i: int) -> Tuple[np.ndarray, np.ndarray]:
                        frame = np.asarray(rgb_ds[i])
                        rgb = frame[..., :3]
                        return rgb, rgb

            else:
                raise ValueError(
                    f"Unsupported trajectory obs layout in {artifacts.trajectory_h5}: "
                    f"expected 'sensor_data' or 'rgb', got keys={list(obs_group.keys())}"
                )

            for i in range(n_side):
                ok, bgr = cap.read()
                if ok:
                    general_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    last_general_rgb = general_rgb
                elif last_general_rgb is not None:
                    # Some recorder setups end 1 frame earlier than side-camera stream.
                    # Repeat the last general frame to avoid cutting the terminal step.
                    general_rgb = last_general_rgb
                else:
                    break
                top_raw, wrist_raw = get_side(i)
                composed = compose_single_frame(
                    general_rgb,
                    ensure_hwc_rgb(top_raw),
                    ensure_hwc_rgb(wrist_raw),
                )

                if writer is None:
                    h, w = composed.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_raw_mp4),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(artifacts.fps),
                        (int(w), int(h)),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to open writer for {out_raw_mp4}")

                writer.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
                frames_written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_written <= 0:
        raise RuntimeError(f"No composed frames produced from {artifacts.general_mp4} + {artifacts.trajectory_h5}")
    return frames_written


def transcode_to_web_mp4(
    raw_mp4: Path,
    output_mp4: Path,
    crf: int,
    preset: str,
) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_mp4),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        str(output_mp4),
    ]
    run_cmd(cmd)


def write_gif_from_mp4(
    mp4_path: Path,
    gif_path: Path,
    gif_fps: int,
    gif_max_width: int,
) -> None:
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={max(1, int(gif_fps))}"]
    if gif_max_width > 0:
        vf_parts.append(f"scale={int(gif_max_width)}:-1:flags=lanczos:force_original_aspect_ratio=decrease")
    vf = ",".join(vf_parts)

    with tempfile.TemporaryDirectory(prefix="gif_palette_") as td:
        palette_path = Path(td) / "palette.png"
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_path),
                "-vf",
                f"{vf},palettegen",
                str(palette_path),
            ]
        )
        run_cmd(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(mp4_path),
                "-i",
                str(palette_path),
                "-lavfi",
                f"{vf}[x];[x][1:v]paletteuse",
                str(gif_path),
            ]
        )


def try_export_successful_episode(
    task: TaskSpec,
    policy: str,
    args: argparse.Namespace,
    checkpoint_map: Dict[str, Path],
    output_dir: Path,
) -> Tuple[TaskResult, Dict[str, Any]]:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    safe_name = task.env_id.replace("/", "_")
    final_mp4 = final_dir / f"{safe_name}.mp4"
    final_gif = final_dir / f"{safe_name}.gif"

    if final_mp4.exists() and final_gif.exists() and not args.overwrite:
        return (
            TaskResult(
                env_id=task.env_id,
                source_hint=task.source_hint,
                policy=policy,
                final_mp4=str(final_mp4),
                final_gif=str(final_gif),
                frames_written=0,
                fps=0.0,
                success=False,
                attempts_tried=0,
                seed_used=None,
                skipped=True,
            ),
            {},
        )

    if args.max_attempts_per_task <= 0:
        raise ValueError(f"--max-attempts-per-task must be > 0, got {args.max_attempts_per_task}")

    attempt_errors: List[str] = []
    attempts_tried = 0

    def rollout_seed_for_task(env_id: str, attempt_idx_local: int) -> int:
        if env_id.startswith("ShellGame"):
            return int(args.shellgame_seed) + int(attempt_idx_local)
        return int(args.seed) + int(attempt_idx_local)

    for attempt_idx in range(args.max_attempts_per_task):
        attempts_tried += 1
        seed = rollout_seed_for_task(task.env_id, attempt_idx)

        try:
            with tempfile.TemporaryDirectory(prefix=f"demo_{safe_name}_seed_{seed}_") as td:
                tmp_dir = Path(td)
                raw_composed_mp4 = tmp_dir / f"{safe_name}_composed_raw.mp4"

                if policy == "motion_planning":
                    if args.keep_intermediate:
                        run_dir = output_dir / "intermediate" / safe_name / f"motion_seed_{seed}"
                        if run_dir.exists() and args.overwrite:
                            shutil.rmtree(run_dir)
                        run_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        run_dir = tmp_dir / "motion"
                        run_dir.mkdir(parents=True, exist_ok=True)

                    artifacts = run_motion_planning_episode(
                        env_id=task.env_id,
                        seed=seed,
                        run_dir=run_dir,
                        overlay_info=int(args.motion_overlay_info),
                    )
                    if not artifacts.success:
                        attempt_errors.append(f"seed={seed}: planner rollout ended without success.")
                        continue

                    frames_written = compose_motion_to_raw_mp4(
                        artifacts=artifacts,
                        out_raw_mp4=raw_composed_mp4,
                    )
                    fps = float(artifacts.fps)
                    source_meta = artifacts.source_meta
                elif policy == "ppo":
                    ckpt = checkpoint_map.get(task.env_id)
                    if ckpt is None:
                        raise ValueError(f"PPO checkpoint not found for {task.env_id}")
                    streams = run_ppo_episode(
                        env_id=task.env_id,
                        checkpoint_path=ckpt,
                        seed=seed,
                        sim_backend=args.sim_backend,
                        ppo_max_steps=args.ppo_max_steps,
                    )
                    if not streams.success:
                        attempt_errors.append(f"seed={seed}: PPO rollout ended without success.")
                        continue

                    composed = compose_frames(
                        general_frames=streams.general_frames,
                        top_frames=streams.top_frames,
                        wrist_frames=streams.wrist_frames,
                    )
                    write_raw_mp4_frames(raw_composed_mp4, composed, fps=streams.fps)
                    frames_written = len(composed)
                    fps = float(streams.fps)
                    source_meta = streams.source_meta
                else:
                    raise ValueError(f"Unknown policy: {policy}")

                transcode_to_web_mp4(
                    raw_mp4=raw_composed_mp4,
                    output_mp4=final_mp4,
                    crf=int(args.h264_crf),
                    preset=str(args.h264_preset),
                )
                write_gif_from_mp4(
                    mp4_path=final_mp4,
                    gif_path=final_gif,
                    gif_fps=int(args.gif_fps),
                    gif_max_width=int(args.gif_max_width),
                )

                result = TaskResult(
                    env_id=task.env_id,
                    source_hint=task.source_hint,
                    policy=policy,
                    final_mp4=str(final_mp4),
                    final_gif=str(final_gif),
                    frames_written=frames_written,
                    fps=fps,
                    success=True,
                    attempts_tried=attempts_tried,
                    seed_used=seed,
                    skipped=False,
                )
                return result, source_meta
        except Exception as exc:  # noqa: BLE001
            attempt_errors.append(f"seed={seed}: {exc}")

    joined_errors = "\n".join(attempt_errors[-8:]) if attempt_errors else "Unknown error."
    return (
        TaskResult(
            env_id=task.env_id,
            source_hint=task.source_hint,
            policy=policy,
            final_mp4=str(final_mp4),
            final_gif=str(final_gif),
            frames_written=0,
            fps=0.0,
            success=False,
            attempts_tried=attempts_tried,
            seed_used=None,
            skipped=False,
            error=f"No successful rollout after {attempts_tried} attempts.\n{joined_errors}",
        ),
        {},
    )


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_map = resolve_latest_checkpoints(args.ckpt_dir)
    tasks = resolve_tasks(args)
    if not tasks:
        raise RuntimeError("No tasks resolved. Check --tasks or --tasks-csv.")

    plan: List[Tuple[TaskSpec, List[str]]] = []
    for task in tasks:
        candidates = policy_candidates(
            env_id=task.env_id,
            source_hint=task.source_hint,
            checkpoint_map=checkpoint_map,
            policy_preference=args.policy_preference,
        )
        plan.append((task, candidates))

    print("Planned tasks:")
    for task, candidates in plan:
        if not candidates:
            print(
                f"  - {task.env_id}: no policy (source={task.source_hint}, "
                f"has_motion={has_motion_policy(task.env_id)}, has_ppo={task.env_id in checkpoint_map})"
            )
        else:
            print(f"  - {task.env_id}: {', '.join(candidates)} (source={task.source_hint})")

    if args.dry_run:
        print("\nDry-run mode: no rollouts executed.")
        return

    results: List[TaskResult] = []
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for task, candidates in plan:
        if not candidates:
            results.append(
                TaskResult(
                    env_id=task.env_id,
                    source_hint=task.source_hint,
                    policy="unresolved",
                    final_mp4=str((output_dir / "final" / f"{task.env_id.replace('/', '_')}.mp4")),
                    final_gif=str((output_dir / "final" / f"{task.env_id.replace('/', '_')}.gif")),
                    frames_written=0,
                    fps=0.0,
                    success=False,
                    attempts_tried=0,
                    seed_used=None,
                    error="No available policy (neither motion-planning script nor PPO checkpoint found).",
                )
            )
            print(f"[error] {task.env_id}: no available policy.")
            continue

        task_done = False
        last_failure_result: Optional[TaskResult] = None
        for policy in candidates:
            print(f"[run] {task.env_id} with {policy} (up to {args.max_attempts_per_task} attempts)")
            result, source_meta = try_export_successful_episode(
                task=task,
                policy=policy,
                args=args,
                checkpoint_map=checkpoint_map,
                output_dir=output_dir,
            )

            if result.skipped:
                print(f"[skip] {task.env_id}: media already exists and --overwrite is not set.")
                results.append(result)
                task_done = True
                break

            if result.error is None and result.success:
                meta_path = metadata_dir / f"{task.env_id.replace('/', '_')}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "env_id": task.env_id,
                            "source_hint": task.source_hint,
                            "policy": policy,
                            "final_mp4": result.final_mp4,
                            "final_gif": result.final_gif,
                            "frames_written": result.frames_written,
                            "fps": result.fps,
                            "success": result.success,
                            "attempts_tried": result.attempts_tried,
                            "seed_used": result.seed_used,
                            "source_meta": source_meta,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(
                    f"[ok] {task.env_id}: {result.final_mp4}, {result.final_gif} "
                    f"(frames={result.frames_written}, fps={result.fps:.2f}, seed={result.seed_used})"
                )
                results.append(result)
                task_done = True
                break

            print(f"[warn] {task.env_id} via {policy} failed:\n{result.error}")
            last_failure_result = result

        if not task_done:
            last_err = (
                last_failure_result.error
                if last_failure_result is not None and last_failure_result.error is not None
                else "All candidate policies failed."
            )
            last_attempts = (
                last_failure_result.attempts_tried
                if last_failure_result is not None
                else int(args.max_attempts_per_task)
            )
            results.append(
                TaskResult(
                    env_id=task.env_id,
                    source_hint=task.source_hint,
                    policy=",".join(candidates),
                    final_mp4=str((output_dir / "final" / f"{task.env_id.replace('/', '_')}.mp4")),
                    final_gif=str((output_dir / "final" / f"{task.env_id.replace('/', '_')}.gif")),
                    frames_written=0,
                    fps=0.0,
                    success=False,
                    attempts_tried=last_attempts,
                    seed_used=None,
                    error=last_err,
                )
            )
            print(f"[error] {task.env_id}: all candidate policies failed.")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)

    num_ok = sum(1 for r in results if (not r.skipped and r.error is None))
    num_err = sum(1 for r in results if r.error is not None)
    num_skip = sum(1 for r in results if r.skipped)
    print(f"\nDone. success={num_ok}, skipped={num_skip}, errors={num_err}. Summary: {summary_path}")


if __name__ == "__main__":
    main()
