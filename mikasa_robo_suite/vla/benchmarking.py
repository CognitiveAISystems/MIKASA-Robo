"""Reusable evaluation utilities for the MIKASA-Robo-VLA benchmark.

The canonical benchmark runs one seeded episode stream per task with
``num_envs=1``. Policies may emit one action or a chunk of actions per forward
pass; the chunk queue is recreated for every episode before stepping the env.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch

import mikasa_robo_suite.vla.memory_envs  # noqa: F401 - registers VLA env IDs
from mikasa_robo_suite.vla.utils.apply_wrappers import apply_mikasa_vla_wrappers

START_SEED = 4242424242
NUM_EPISODES_PER_TASK = 50
OBS_MODE = "rgb"
CONTROL_MODE = "pd_ee_delta_pose"
REWARD_MODE = "normalized_dense"
WRAPPER_CHAIN = "apply_mikasa_vla_wrappers(include_overlays=False)"
SUPPORTED_SPLITS = ("short", "medium", "long", "all")

JsonDict = Dict[str, Any]
ProgressCallback = Callable[[str], None]
EpisodeCallback = Callable[[int, "EpisodeResult"], None]
TaskStartCallback = Callable[[int, "BenchmarkTask"], None]
TaskDoneCallback = Callable[[int, JsonDict], None]


# ---------------------------------------------------------------------------
# Video composition utilities (same layout as prepare_benchmark_demo_videos.py)
# ---------------------------------------------------------------------------

def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.dtype == np.uint8:
        return np.ascontiguousarray(x)
    x = x.astype(np.float32, copy=False)
    max_val = float(np.nanmax(x)) if x.size > 0 else 0.0
    if max_val <= 1.0 + 1e-5:
        x = x * 255.0
    return np.clip(x, 0.0, 255.0).astype(np.uint8)


def _ensure_hwc_rgb(x: Any) -> np.ndarray:
    """Strip batch dimensions and return a (H, W, 3) uint8 RGB array."""
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected [H,W,C] image, got shape {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return _to_uint8_rgb(arr)


def _compose_frame(
    general: np.ndarray,
    top: np.ndarray,
    wrist: np.ndarray,
) -> np.ndarray:
    """Compose one output frame with the same layout as prepare_benchmark_demo_videos.py.

    Layout::

        ┌──────────────────┬──────────┐
        │                  │  top     │
        │  general render  ├──────────┤
        │                  │  wrist   │
        └──────────────────┴──────────┘

    The right panel occupies half the width of the general render; top and
    wrist each take half its height.
    """
    import cv2  # lazy import — not needed for metric-only runs

    g = _to_uint8_rgb(general)
    t = _to_uint8_rgb(top)
    w = _to_uint8_rgb(wrist)

    g_h, g_w = g.shape[:2]
    side_w = max(1, g_w // 2)
    top_h = g_h // 2
    wrist_h = g_h - top_h

    t = cv2.resize(t, (side_w, top_h), interpolation=cv2.INTER_AREA)
    w = cv2.resize(w, (side_w, wrist_h), interpolation=cv2.INTER_AREA)

    def _label(img: np.ndarray, text: str) -> None:
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    _label(t, "top")
    _label(w, "wrist")

    right = np.concatenate([t, w], axis=0)
    return np.ascontiguousarray(np.concatenate([g, right], axis=1))


def _write_video(path: Path, frames: Sequence[np.ndarray], fps: float) -> None:
    """Write frames to an mp4 file using cv2, then try to transcode to H.264."""
    import cv2  # lazy import

    if not frames:
        return

    first = _to_uint8_rgb(frames[0])
    h, w = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)

    raw_path = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {raw_path}")
    try:
        for frame in frames:
            rgb = _to_uint8_rgb(frame)
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    # Transcode to web-friendly H.264 with ffmpeg if available.
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-crf", "20", "-preset", "medium",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        raw_path.unlink(missing_ok=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg not available — keep raw mp4 under the intended path
        raw_path.rename(path)


# ---------------------------------------------------------------------------
# Repository helpers
# ---------------------------------------------------------------------------

def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_task_csv_path() -> Path:
    return repository_root() / "mikasa_robo_vla_envs.csv"


def make_run_dir(base_dir: Union[str, Path]) -> Path:
    """Create and return a timestamped subdirectory under *base_dir*.

    Each call produces a unique directory named ``YYYY-MM-DD_HH-MM-SS`` so
    successive runs never overwrite each other.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkTask:
    """Metadata for one task row in ``mikasa_robo_vla_envs.csv``."""

    env_id: str
    split: str
    memory_type: str
    max_episode_steps: int
    language_instruction: str
    data_source: str

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "BenchmarkTask":
        return cls(
            env_id=row["Name"],
            split=_canonical_split(row["Horizon Split"]),
            memory_type=row["Memory Type"],
            max_episode_steps=int(row["Max Length"]),
            language_instruction=row.get("language_instruction", ""),
            data_source=row.get("Data Source", ""),
        )


@dataclass(frozen=True)
class BenchmarkConfig:
    """Execution settings for canonical task and split evaluation."""

    start_seed: int = START_SEED
    n_episodes: int = NUM_EPISODES_PER_TASK
    obs_mode: str = OBS_MODE
    control_mode: str = CONTROL_MODE
    reward_mode: str = REWARD_MODE
    sim_backend: str = "gpu"
    include_overlays: bool = False
    # When True, each episode is recorded as a composed video:
    #   [general render | top camera over wrist camera]
    # Videos go to output_dir/videos/<env_id>/episode_NNNN.mp4.
    save_videos: bool = False

    def __post_init__(self) -> None:
        if self.n_episodes <= 0:
            raise ValueError(f"n_episodes must be > 0, got {self.n_episodes}")
        if self.start_seed < 0:
            raise ValueError(f"start_seed must be >= 0, got {self.start_seed}")


@dataclass(frozen=True)
class EpisodeResult:
    success_once: bool
    episode_return: float
    n_steps: int
    seed: int


class ChunkPolicy(Protocol):
    """Minimal policy interface consumed by the benchmark runner."""

    chunk_size: int

    def forward(self, obs: Mapping[str, Any]) -> Union[torch.Tensor, np.ndarray]:
        """Return one action or a chunk of actions for ``num_envs=1``."""


# ---------------------------------------------------------------------------
# Task loading and selection
# ---------------------------------------------------------------------------

def _canonical_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SUPPORTED_SPLITS:
        raise ValueError(f"Unknown split={split!r}. Expected one of {SUPPORTED_SPLITS}.")
    return normalized


def load_benchmark_tasks(csv_path: Optional[Union[str, Path]] = None) -> List[BenchmarkTask]:
    path = Path(csv_path) if csv_path is not None else default_task_csv_path()
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [BenchmarkTask.from_csv_row(row) for row in csv.DictReader(csv_file, delimiter=";")]


def select_benchmark_tasks(
    *,
    split: Optional[str] = None,
    env_ids: Optional[Sequence[str]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> List[BenchmarkTask]:
    """Select tasks from the benchmark CSV.

    Accepted call forms:

    ``split="short"`` / ``"medium"`` / ``"long"``
        All tasks in that horizon split.

    ``split="all"``
        All 90 canonical benchmark tasks.

    ``env_ids=["EnvA-VLA-v0", ...]``
        Exactly those tasks by ID.  IDs absent from the CSV are accepted as
        custom tasks (``split="custom"``, ``memory_type="Unknown"``).
    """
    tasks = load_benchmark_tasks(csv_path)
    task_by_id = {task.env_id: task for task in tasks}
    selected_split = _canonical_split(split) if split is not None else None

    if env_ids is not None:
        if selected_split is not None:
            raise ValueError("Pass either split=... or env_ids=[...], not both.")
        result: List[BenchmarkTask] = []
        for env_id in env_ids:
            if env_id in task_by_id:
                result.append(task_by_id[env_id])
            else:
                result.append(BenchmarkTask(
                    env_id=env_id, split="custom", memory_type="Unknown",
                    max_episode_steps=0, language_instruction="", data_source="custom",
                ))
        return result

    if selected_split is None:
        raise ValueError("Pass either split=... or env_ids=[...].")
    if selected_split == "all":
        return tasks
    return [task for task in tasks if task.split == selected_split]


# ---------------------------------------------------------------------------
# Git commit
# ---------------------------------------------------------------------------

def benchmark_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root(), check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------

def make_benchmark_env(
    env_id: str,
    config: BenchmarkConfig,
) -> gym.Env:
    """Create the canonical single-env evaluation environment.

    When ``config.save_videos`` is True, ``render_mode="rgb_array"`` is used
    so that ``env.render()`` returns the 3D scene view needed for the composed
    video layout.  Task overlays are enabled in that case so videos carry debug
    information (step counter, task-specific progress, etc.).
    """
    # For video recording we need env.render() to return the 3D scene and we
    # want overlay text in the render.  For metric-only runs we skip rendering.
    render_mode = "rgb_array" if config.save_videos else "all"
    include_overlays = config.include_overlays or config.save_videos

    make_kwargs: JsonDict = {
        "num_envs": 1,
        "obs_mode": config.obs_mode,
        "control_mode": config.control_mode,
        "render_mode": render_mode,
        "reward_mode": config.reward_mode,
    }
    make_kwargs["sim_backend"] = config.sim_backend

    env = gym.make(env_id, **make_kwargs)
    return apply_mikasa_vla_wrappers(env, include_overlays=include_overlays)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _first_scalar(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.detach().reshape(-1)[0].cpu().item()
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item()


def _policy_chunk_size(policy: ChunkPolicy) -> int:
    chunk_size = int(getattr(policy, "chunk_size", 1))
    if chunk_size <= 0:
        raise ValueError(f"policy.chunk_size must be > 0, got {chunk_size}")
    return chunk_size


def _chunk_actions(
    action_chunk: Union[torch.Tensor, np.ndarray],
    action_shape: Tuple[int, ...],
) -> Iterable[torch.Tensor]:
    chunk = action_chunk if torch.is_tensor(action_chunk) else torch.as_tensor(action_chunk)
    if tuple(chunk.shape) == action_shape:
        chunk = chunk.unsqueeze(0)
    elif chunk.ndim == len(action_shape) + 2 and chunk.shape[0] == 1 and tuple(chunk.shape[2:]) == action_shape:
        chunk = chunk.squeeze(0)
    elif chunk.ndim != len(action_shape) + 1 or tuple(chunk.shape[1:]) != action_shape:
        raise ValueError(
            f"Policy forward must return action shape {action_shape}, "
            f"chunk shape (K, {action_shape}), or batch-one shape "
            f"(1, K, {action_shape}); got {tuple(chunk.shape)}."
        )
    if chunk.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk.")
    for action in chunk:
        yield action


def _batch_action_for_single_env(action: torch.Tensor, env: gym.Env) -> torch.Tensor:
    action_shape = tuple(env.action_space.shape)
    action = action if torch.is_tensor(action) else torch.as_tensor(action)
    target_device = getattr(env.unwrapped, "device", None)
    if target_device is not None:
        action = action.to(device=target_device)
    action = action.to(dtype=torch.float32)
    if tuple(action.shape) == action_shape:
        return action.unsqueeze(0)
    if tuple(action.shape) == (1, *action_shape):
        return action
    raise ValueError(f"Expected one action with shape {action_shape}; got {tuple(action.shape)}.")


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: gym.Env,
    policy: ChunkPolicy,
    episode_seed: int,
    *,
    collect_video: bool = False,
) -> Tuple[EpisodeResult, Optional[List[np.ndarray]]]:
    """Run one seeded episode.

    Returns ``(EpisodeResult, frames_or_None)``.  When *collect_video* is
    True, *frames* is a list of composed ``(H, W*1.5, 3)`` uint8 arrays — one
    per simulator step — in the same layout used by
    ``utils/prepare_benchmark_demo_videos.py``:

    - Left panel  : ``env.render()`` — the 3D scene view with overlays.
    - Right panel : top camera (upper half) + wrist camera (lower half).
    """
    obs, _ = env.reset(seed=episode_seed)
    action_shape = tuple(env.action_space.shape)
    max_steps = getattr(env, "max_episode_steps", None)
    if max_steps is None:
        raise RuntimeError("Wrapped env does not expose max_episode_steps.")

    # Fresh queue per episode — truncated chunks never leak into the next episode.
    action_queue: Deque[torch.Tensor] = deque()
    success_once = False
    total_return = 0.0
    n_steps = 0
    frames: Optional[List[np.ndarray]] = [] if collect_video else None

    for _ in range(int(max_steps)):
        # Collect frame BEFORE the step (captures the state the policy sees).
        if collect_video and frames is not None:
            general = env.render()
            rgb_obs = obs.get("rgb") if isinstance(obs, dict) else None
            if rgb_obs is not None:
                rgb_np = rgb_obs.detach().cpu().numpy() if torch.is_tensor(rgb_obs) else np.asarray(rgb_obs)
                # rgb shape is (1, H, W, 6): first 3 ch = top, last 3 = wrist
                top = rgb_np[0, :, :, :3]
                wrist = rgb_np[0, :, :, 3:6]
            else:
                # Fallback: duplicate general render when camera data unavailable
                g_np = _ensure_hwc_rgb(general)
                top = g_np
                wrist = g_np
            frames.append(_compose_frame(_ensure_hwc_rgb(general), top, wrist))

        if not action_queue:
            action_queue.extend(_chunk_actions(policy.forward(obs), action_shape))

        action = _batch_action_for_single_env(action_queue.popleft(), env)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps += 1
        total_return += float(_first_scalar(reward, default=0.0))
        success_once = success_once or bool(_first_scalar(info.get("success"), default=False))
        done = bool(_first_scalar(terminated, default=False)) or bool(_first_scalar(truncated, default=False))
        if done:
            # Capture the terminal frame so the final state is visible.
            if collect_video and frames is not None:
                general = env.render()
                rgb_obs = obs.get("rgb") if isinstance(obs, dict) else None
                if rgb_obs is not None:
                    rgb_np = rgb_obs.detach().cpu().numpy() if torch.is_tensor(rgb_obs) else np.asarray(rgb_obs)
                    top = rgb_np[0, :, :, :3]
                    wrist = rgb_np[0, :, :, 3:6]
                else:
                    g_np = _ensure_hwc_rgb(general)
                    top = g_np
                    wrist = g_np
                frames.append(_compose_frame(_ensure_hwc_rgb(general), top, wrist))
            break

    return (
        EpisodeResult(
            success_once=success_once,
            episode_return=total_return,
            n_steps=n_steps,
            seed=episode_seed,
        ),
        frames,
    )


# ---------------------------------------------------------------------------
# Task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task: BenchmarkTask,
    policy: ChunkPolicy,
    config: BenchmarkConfig,
    *,
    model: Optional[Mapping[str, Any]] = None,
    commit: Optional[str] = None,
    video_dir: Optional[Union[str, Path]] = None,
    episode_callback: Optional[EpisodeCallback] = None,
) -> JsonDict:
    """Evaluate one task and return its canonical JSON-compatible result dict."""
    env = make_benchmark_env(task.env_id, config)
    episodes: List[EpisodeResult] = []
    try:
        for i in range(config.n_episodes):
            result, frames = run_episode(
                env, policy, config.start_seed + i,
                collect_video=(video_dir is not None),
            )
            episodes.append(result)
            if episode_callback is not None:
                episode_callback(i, result)
            if frames and video_dir is not None:
                vpath = Path(video_dir) / f"episode_{i:04d}.mp4"
                _write_video(vpath, frames, fps=30.0)
    finally:
        env.close()

    successes = [ep.success_once for ep in episodes]
    returns = [ep.episode_return for ep in episodes]
    return {
        "env_id": task.env_id,
        "split": task.split.title(),
        "memory_type": task.memory_type,
        "start_seed": config.start_seed,
        "n_episodes": config.n_episodes,
        "successes": successes,
        "returns": returns,
        "sr": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "benchmark_commit": commit,
        "control_mode": config.control_mode,
        "obs_mode": config.obs_mode,
        "wrapper_chain": WRAPPER_CHAIN,
        "action_chunk_size": _policy_chunk_size(policy),
        "model": dict(model or {}),
        "episode_lengths": [ep.n_steps for ep in episodes],
        "episode_seeds": [ep.seed for ep in episodes],
    }


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def summarize_task_results(task_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    if not task_results:
        raise ValueError("Cannot summarize an empty result set.")

    splits = {str(result["split"]) for result in task_results}
    summary_split = next(iter(splits)) if len(splits) == 1 else "Custom"
    per_task_sr = {str(result["env_id"]): float(result["sr"]) for result in task_results}
    per_task_return = {str(result["env_id"]): float(result["mean_return"]) for result in task_results}
    sr_by_memory_type: Dict[str, List[float]] = defaultdict(list)
    for result in task_results:
        sr_by_memory_type[str(result["memory_type"])].append(float(result["sr"]))

    return {
        "split": summary_split,
        "sr_split": float(np.mean(list(per_task_sr.values()))),
        "sr_per_memory_type": {
            mt: float(np.mean(vals)) for mt, vals in sorted(sr_by_memory_type.items())
        },
        "tasks": list(per_task_sr),
        "per_task_sr": per_task_sr,
        "per_task_mean_return": per_task_return,
    }


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def write_benchmark_results(
    output_dir: Union[str, Path],
    task_results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_path = Path(output_dir)
    for result in task_results:
        _write_json(output_path / f"{result['env_id']}.json", result)
    _write_json(output_path / "summary.json", summary)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    tasks: Sequence[BenchmarkTask],
    policy: ChunkPolicy,
    config: Optional[BenchmarkConfig] = None,
    *,
    output_dir: Optional[Union[str, Path]] = None,
    model: Optional[Mapping[str, Any]] = None,
    progress: Optional[ProgressCallback] = None,
    task_start_callback: Optional[TaskStartCallback] = None,
    episode_callback: Optional[EpisodeCallback] = None,
    task_done_callback: Optional[TaskDoneCallback] = None,
    initial_results: Optional[List[JsonDict]] = None,
) -> Tuple[List[JsonDict], JsonDict]:
    """Evaluate selected tasks; persist JSON results after every finished task.

    After each task completes, its per-task JSON is written and
    ``summary.json`` is updated so a partial run is always readable.

    When ``config.save_videos`` is True, each episode is recorded as a
    composed video (``general render | top / wrist cameras``) at
    ``output_dir/videos/<env_id>/episode_NNNN.mp4``.

    Pass ``initial_results`` to resume a partial run — those results are
    included in every intermediate ``summary.json`` write.
    """
    if not tasks:
        raise ValueError("No benchmark tasks selected.")
    config = config or BenchmarkConfig()
    commit = benchmark_commit()
    output_path = Path(output_dir) if output_dir is not None else None

    # Seed with already-completed results so summary writes stay consistent.
    results: List[JsonDict] = list(initial_results) if initial_results else []
    summary: JsonDict = {}

    for index, task in enumerate(tasks, start=1):
        if progress is not None:
            progress(f"[{index}/{len(tasks)}] {task.env_id}  ({config.n_episodes} episodes)")
        if task_start_callback is not None:
            task_start_callback(index - 1, task)

        video_dir = None
        if config.save_videos and output_path is not None:
            video_dir = output_path / "videos" / task.env_id

        result = evaluate_task(
            task, policy, config,
            model=model, commit=commit, video_dir=video_dir,
            episode_callback=episode_callback,
        )
        results.append(result)
        if task_done_callback is not None:
            task_done_callback(index - 1, result)

        if output_path is not None:
            _write_json(output_path / f"{result['env_id']}.json", result)
            summary = summarize_task_results(results)
            _write_json(output_path / "summary.json", summary)

    if not summary:
        summary = summarize_task_results(results)
    return results, summary


# ---------------------------------------------------------------------------
# Rich terminal UI  (requires `rich`)
# ---------------------------------------------------------------------------

class RichBenchmarkUI:
    """Live rich terminal display with dual progress bars and a live results table.

    Usage::

        with RichBenchmarkUI(tasks, config.n_episodes) as ui:
            results, summary = evaluate_benchmark(
                tasks, policy, config,
                task_start_callback=ui.on_task_start,
                episode_callback=ui.on_episode_done,
                task_done_callback=ui.on_task_done,
            )
    """

    def __init__(
        self,
        tasks: Sequence[BenchmarkTask],
        n_episodes: int,
        *,
        initial_results: Optional[List[JsonDict]] = None,
    ) -> None:
        try:
            import rich  # noqa: F401
        except ImportError as exc:
            raise ImportError("Install `rich` to use RichBenchmarkUI: pip install rich") from exc

        self._tasks = list(tasks)
        self._n_episodes = n_episodes
        self._completed: List[JsonDict] = list(initial_results) if initial_results else []
        self._current_idx: int = -1
        self._cur_ep: int = 0
        self._cur_sr: float = 0.0
        self._cur_ret: float = 0.0
        self._live: Any = None
        self._overall_progress: Any = None
        self._episode_progress: Any = None
        self._overall_tid: Any = None
        self._episode_tid: Any = None
        self._console: Any = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "RichBenchmarkUI":
        from rich.console import Console
        from rich.live import Live
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._console = Console()
        self._overall_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
        )
        self._episode_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}[/bold green]"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            console=self._console,
        )
        self._overall_tid = self._overall_progress.add_task(
            f"Tasks  [0/{len(self._tasks)}]", total=len(self._tasks)
        )
        self._episode_tid = self._episode_progress.add_task(
            "Episodes", total=self._n_episodes
        )
        self._live = Live(
            self._build_renderable(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._live is not None:
            self._live.__exit__(*args)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_table(self) -> Any:
        from rich.table import Table

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim blue",
            row_styles=["", "dim"],
            expand=True,
            show_edge=True,
        )
        table.add_column("Task", style="white", no_wrap=True, ratio=4)
        table.add_column("Split", justify="center", ratio=1)
        table.add_column("Memory Type", justify="center", ratio=2)
        table.add_column("Episodes", justify="right", ratio=1)
        table.add_column("SR", justify="right", ratio=1)
        table.add_column("Avg Return", justify="right", ratio=1)

        for r in self._completed:
            sr = float(r["sr"])
            sr_str = self._sr_colored(sr)
            table.add_row(
                r["env_id"],
                r["split"],
                r["memory_type"],
                str(r["n_episodes"]),
                sr_str,
                f"{r['mean_return']:.4f}",
            )

        # In-progress row
        if self._current_idx >= 0 and self._cur_ep > 0:
            task = self._tasks[self._current_idx]
            sr_str = self._sr_colored(self._cur_sr)
            table.add_row(
                f"[bold]{task.env_id}[/bold] [dim italic](running…)[/dim italic]",
                task.split.title(),
                task.memory_type,
                f"[yellow]{self._cur_ep}/{self._n_episodes}[/yellow]",
                sr_str,
                f"[yellow]{self._cur_ret:.4f}[/yellow]",
            )

        return table

    @staticmethod
    def _sr_colored(sr: float) -> str:
        if sr >= 0.7:
            color = "bright_green"
        elif sr >= 0.4:
            color = "yellow"
        else:
            color = "red"
        return f"[{color}]{sr:.1%}[/{color}]"

    def _build_renderable(self) -> Any:
        from rich.console import Group
        from rich.panel import Panel
        from rich.rule import Rule

        done = len(self._completed)
        total = len(self._tasks)
        header = f"[bold white]MIKASA-Robo-VLA Benchmark[/bold white]  [dim]{done}/{total} tasks[/dim]"

        return Panel(
            Group(
                self._overall_progress,
                self._episode_progress,
                Rule(style="dim blue"),
                self._build_table(),
            ),
            title=header,
            border_style="blue",
            padding=(0, 1),
        )

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._build_renderable())

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_task_start(self, task_idx: int, task: BenchmarkTask) -> None:
        self._current_idx = task_idx
        self._cur_ep = 0
        self._cur_sr = 0.0
        self._cur_ret = 0.0
        done = len(self._completed)
        self._overall_progress.update(
            self._overall_tid,
            description=f"Tasks  [{done}/{len(self._tasks)}]  {task.env_id}",
        )
        self._episode_progress.reset(self._episode_tid, total=self._n_episodes, completed=0)
        self._episode_progress.update(
            self._episode_tid,
            description=f"Episodes  {task.env_id}",
        )
        self._refresh()

    def on_episode_done(self, episode_idx: int, result: EpisodeResult) -> None:
        n = episode_idx + 1
        self._cur_ep = n
        self._cur_sr += (float(result.success_once) - self._cur_sr) / n
        self._cur_ret += (result.episode_return - self._cur_ret) / n
        self._episode_progress.update(self._episode_tid, completed=n)
        self._refresh()

    def on_task_done(self, task_idx: int, result_dict: JsonDict) -> None:
        self._completed.append(result_dict)
        self._current_idx = -1
        self._cur_ep = 0
        done = len(self._completed)
        self._overall_progress.update(
            self._overall_tid,
            completed=done,
            description=f"Tasks  [{done}/{len(self._tasks)}]",
        )
        self._refresh()
