#!/usr/bin/env python3
"""Convert RLDS (TFDS episodic format) datasets to LeRobotDataset v3 format."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

# Import the naming helper directly, bypassing mikasa_robo_suite package __init__
# so this converter can run in a lightweight venv without sapien/mani_skill.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mikasa_robo_suite" / "vla" / "utils"))
from dataset_naming import env_id_to_dataset_name  # noqa: E402

# Keep tfds builds quiet on local/offline machines.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TFDS_DISABLE_GCS", "1")


@dataclass
class TaskPaths:
    task_id: str
    rlds_task_dir: Path
    rlds_version_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RLDS task dataset(s) in data_mikasa_robo/data_rlds/* to LeRobotDataset v3."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--task",
        type=str,
        help=(
            "Convert one RLDS task id. Accepts either gym env_id "
            "('RememberColor3-VLA-v0') or snake_case dataset name ('remember_color_3_vla_v0')."
        ),
    )
    scope.add_argument(
        "--all",
        action="store_true",
        help="Convert all task directories under --rlds-root.",
    )

    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path to MIKASA-Robo root. Defaults to auto-detect from this script.",
    )
    parser.add_argument(
        "--rlds-root",
        type=str,
        default="data_mikasa_robo/data_rlds",
        help="Relative path from repo root containing RLDS task directories.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=str,
        default="data_mikasa_robo/data_lerobot",
        help="Relative path from repo root where LeRobot datasets are written.",
    )
    parser.add_argument(
        "--repo-id-template",
        type=str,
        default="{task}",
        help=("Template for LeRobot repo_id. Use {task} placeholder. Examples: '{task}' or 'avanturist322/{task}'."),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="RLDS version folder (default: latest semver subdir, e.g., 1.0.0).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="TFDS split name to read from RLDS task dataset.",
    )
    parser.add_argument("--fps", type=int, default=10, help="LeRobot dataset FPS.")
    parser.add_argument(
        "--robot-type",
        type=str,
        default="mikasa_robo",
        help="LeRobot robot_type metadata.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap for number of converted episodes per task.",
    )
    parser.add_argument(
        "--overwrite-dest",
        action="store_true",
        help="Remove existing output directory for each task before conversion.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip tasks whose output directory already exists instead of failing.",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Store image frames directly instead of encoding MP4 videos.",
    )
    return parser.parse_args()


def repo_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    return Path(__file__).resolve().parents[2]


def repo_id_to_relpath(repo_id: str) -> Path:
    return Path(*repo_id.split("/"))


def semver_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")
    return int(major), int(minor), int(patch)


def resolve_version_dir(task_dir: Path, forced_version: str | None) -> Path:
    if forced_version:
        version_dir = task_dir / forced_version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version dir not found: {version_dir}")
        return version_dir

    semver_dirs = [p for p in task_dir.iterdir() if p.is_dir() and p.name.count(".") == 2]
    semver_dirs = [p for p in semver_dirs if all(chunk.isdigit() for chunk in p.name.split("."))]
    if not semver_dirs:
        raise FileNotFoundError(f"No semver version dirs found in {task_dir}. Expected e.g. {task_dir / '1.0.0'}")
    semver_dirs.sort(key=lambda p: semver_key(p.name))
    return semver_dirs[-1]


def discover_tasks(
    rlds_root: Path,
    task: str | None,
    all_tasks: bool,
    forced_version: str | None,
) -> list[TaskPaths]:
    if task:
        dataset_name = env_id_to_dataset_name(task)
        task_dir = rlds_root / dataset_name
        if not task_dir.exists():
            raise FileNotFoundError(f"Task not found in RLDS root: {task_dir}")
        return [
            TaskPaths(
                task_id=dataset_name,
                rlds_task_dir=task_dir,
                rlds_version_dir=resolve_version_dir(task_dir, forced_version),
            )
        ]

    if not all_tasks:
        raise ValueError("Either --task or --all must be provided.")

    tasks: list[TaskPaths] = []
    for task_dir in sorted(rlds_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        try:
            version_dir = resolve_version_dir(task_dir, forced_version)
        except FileNotFoundError:
            continue
        tasks.append(
            TaskPaths(
                task_id=task_dir.name,
                rlds_task_dir=task_dir,
                rlds_version_dir=version_dir,
            )
        )
    if not tasks:
        raise FileNotFoundError(f"No RLDS tasks discovered in {rlds_root}")
    return tasks


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return normalize_text(value.item())
        if value.size == 1:
            return normalize_text(value.reshape(-1)[0])
    return str(value)


def to_uint8_hwc(image: Any) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims (H,W,C), got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, 0.0, 1.0)
        return (clipped * 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def dict_leaf_length(data: dict[str, Any]) -> int:
    first_value = next(iter(data.values()))
    if isinstance(first_value, dict):
        return dict_leaf_length(first_value)
    return len(first_value)


def index_nested(data: Any, idx: int) -> Any:
    if isinstance(data, dict):
        return {k: index_nested(v, idx) for k, v in data.items()}
    return data[idx]


def iter_episode_steps(steps_obj: Any) -> Iterable[dict[str, Any]]:
    if hasattr(steps_obj, "as_numpy_iterator"):
        yield from steps_obj.as_numpy_iterator()
        return

    if isinstance(steps_obj, dict):
        length = dict_leaf_length(steps_obj)
        for i in range(length):
            yield index_nested(steps_obj, i)
        return

    yield from steps_obj


def infer_features(first_step: dict[str, Any], include_wrist: bool) -> dict[str, dict[str, Any]]:
    top = to_uint8_hwc(first_step["observation"]["image"])
    proprio = np.asarray(first_step["observation"]["proprio"], dtype=np.float32)
    action = np.asarray(first_step["action"], dtype=np.float32)

    features: dict[str, dict[str, Any]] = {
        "observation.images.top": {
            "dtype": "video",
            "shape": tuple(top.shape),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": tuple(proprio.shape),
        },
        "action": {
            "dtype": "float32",
            "shape": tuple(action.shape),
        },
    }
    if include_wrist:
        wrist = to_uint8_hwc(first_step["observation"]["wrist_image"])
        features["observation.images.wrist"] = {
            "dtype": "video",
            "shape": tuple(wrist.shape),
            "names": ["height", "width", "channel"],
        }
    return features


def create_lerobot_dataset(
    repo_id: str,
    root: Path,
    fps: int,
    robot_type: str,
    features: dict[str, dict[str, Any]],
    use_videos: bool,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    attempts: list[dict[str, Any]] = [
        {
            "repo_id": repo_id,
            "root": root,
            "fps": fps,
            "robot_type": robot_type,
            "features": features,
            "use_videos": use_videos,
        },
        {
            "repo_id": repo_id,
            "root": root,
            "fps": fps,
            "robot_type": robot_type,
            "features": features,
        },
    ]

    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return LeRobotDataset.create(**kwargs)
        except TypeError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def save_episode_compat(dataset: Any) -> None:
    try:
        dataset.save_episode(parallel_encoding=False)
    except TypeError:
        dataset.save_episode()


def finalize_dataset(dataset: Any) -> None:
    if hasattr(dataset, "finalize"):
        dataset.finalize()
        return
    if hasattr(dataset, "consolidate"):
        dataset.consolidate()
        return
    raise AttributeError("LeRobot dataset object does not expose finalize() or consolidate().")


def convert_one_task(
    task_paths: TaskPaths,
    output_root: Path,
    repo_id_template: str,
    split: str,
    fps: int,
    robot_type: str,
    use_videos: bool,
    overwrite_dest: bool,
    skip_existing: bool,
    max_episodes: int | None,
) -> bool:
    import tensorflow_datasets as tfds

    task = task_paths.task_id
    repo_id = repo_id_template.format(task=task)
    if not repo_id.strip():
        raise ValueError("repo_id from --repo-id-template is empty")

    destination_dir = output_root / repo_id_to_relpath(repo_id)
    if destination_dir.exists():
        if overwrite_dest:
            shutil.rmtree(destination_dir)
        elif skip_existing:
            print(f"[{task}] skip: destination already exists -> {destination_dir}")
            return False
        else:
            raise FileExistsError(
                f"Destination already exists: {destination_dir}. "
                "Use --overwrite-dest to replace or --skip-existing to skip."
            )

    builder = tfds.builder_from_directory(str(task_paths.rlds_version_dir))
    ds = builder.as_dataset(split=split)
    if max_episodes is not None:
        ds = ds.take(max_episodes)

    lerobot_dataset = None
    include_wrist = True
    episodes_written = 0
    steps_written = 0
    latest_instruction = task

    episodes_iter = tfds.as_numpy(ds)
    for episode in tqdm(episodes_iter, desc=f"[{task}] episodes", unit="ep"):
        steps = iter_episode_steps(episode["steps"])
        try:
            first_step = next(steps)
        except StopIteration:
            continue

        if lerobot_dataset is None:
            include_wrist = "wrist_image" in first_step["observation"]
            features = infer_features(first_step, include_wrist=include_wrist)
            try:
                lerobot_dataset = create_lerobot_dataset(
                    repo_id=repo_id,
                    root=destination_dir,
                    fps=fps,
                    robot_type=robot_type,
                    features=features,
                    use_videos=use_videos,
                )
            except ValueError as exc:
                if "/" not in repo_id:
                    fallback_repo_id = f"local/{repo_id}"
                    print(
                        f"[{task}] repo_id='{repo_id}' not accepted by current lerobot. "
                        f"Falling back to '{fallback_repo_id}'."
                    )
                    repo_id = fallback_repo_id
                    destination_dir = output_root / repo_id_to_relpath(repo_id)
                    if destination_dir.exists():
                        if overwrite_dest:
                            shutil.rmtree(destination_dir)
                        elif skip_existing:
                            print(f"[{task}] skip: fallback destination already exists -> {destination_dir}")
                            return False
                        else:
                            raise FileExistsError(
                                f"Destination already exists: {destination_dir}. "
                                "Use --overwrite-dest to replace or --skip-existing to skip."
                            )
                    lerobot_dataset = create_lerobot_dataset(
                        repo_id=repo_id,
                        root=destination_dir,
                        fps=fps,
                        robot_type=robot_type,
                        features=features,
                        use_videos=use_videos,
                    )
                else:
                    raise exc

        def write_step(step: dict[str, Any]) -> None:
            nonlocal steps_written, latest_instruction
            if "language_instruction" in step:
                txt = normalize_text(step["language_instruction"]).strip()
                if txt:
                    latest_instruction = txt

            frame = {
                "observation.images.top": to_uint8_hwc(step["observation"]["image"]),
                "observation.state": np.asarray(step["observation"]["proprio"], dtype=np.float32),
                "action": np.asarray(step["action"], dtype=np.float32),
                "task": latest_instruction or task,
            }
            if include_wrist and "wrist_image" in step["observation"]:
                frame["observation.images.wrist"] = to_uint8_hwc(step["observation"]["wrist_image"])

            lerobot_dataset.add_frame(frame)
            steps_written += 1

        write_step(first_step)
        for step in steps:
            write_step(step)

        save_episode_compat(lerobot_dataset)
        episodes_written += 1

    if lerobot_dataset is None:
        raise RuntimeError(f"No episodes written for task '{task}'. Check split='{split}' and RLDS data content.")

    finalize_dataset(lerobot_dataset)

    source_metadata = task_paths.rlds_version_dir / "metadata.json"
    if source_metadata.exists() and destination_dir.exists():
        shutil.copy2(source_metadata, destination_dir / "source_rlds_metadata.json")

    print(f"[{task}] done: episodes={episodes_written}, steps={steps_written}, output={destination_dir}")
    return True


def main() -> None:
    args = parse_args()
    repo_root = repo_root_from_args(args.repo_root)
    rlds_root = (repo_root / args.rlds_root).resolve()
    output_root = (repo_root / args.lerobot_root).resolve()

    if "{task}" not in args.repo_id_template:
        raise ValueError("--repo-id-template must include '{task}' placeholder")
    if not rlds_root.exists():
        raise FileNotFoundError(f"RLDS root not found: {rlds_root}")

    tasks = discover_tasks(
        rlds_root=rlds_root,
        task=args.task,
        all_tasks=args.all,
        forced_version=args.version,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if args.overwrite_dest and args.skip_existing:
        raise ValueError("--overwrite-dest and --skip-existing are mutually exclusive.")

    converted: list[str] = []
    skipped: list[str] = []
    for tp in tasks:
        did_convert = convert_one_task(
            task_paths=tp,
            output_root=output_root,
            repo_id_template=args.repo_id_template,
            split=args.split,
            fps=args.fps,
            robot_type=args.robot_type,
            use_videos=not args.no_videos,
            overwrite_dest=args.overwrite_dest,
            skip_existing=args.skip_existing,
            max_episodes=args.max_episodes,
        )
        (converted if did_convert else skipped).append(tp.task_id)

    print(f"Summary: converted={len(converted)}, skipped={len(skipped)}, total={len(tasks)}")
    if skipped:
        print("Skipped tasks:")
        for t in skipped:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
