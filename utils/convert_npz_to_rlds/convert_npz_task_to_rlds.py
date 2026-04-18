#!/usr/bin/env python3
"""Convert one MIKASA .npz task directory into RLDS artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import the naming helper directly, bypassing mikasa_robo_suite package __init__
# so this converter can run in a lightweight venv without sapien/mani_skill.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mikasa_robo_suite" / "vla" / "utils"))
from dataset_naming import env_id_to_dataset_name  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MIKASA-Robo-VLA task from .npz episodes to RLDS.")
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Task folder name under data_mikasa_robo/data_npz "
            "(gym env_id like 'RememberColor3-VLA-v0' or snake_case 'remember_color_3_vla_v0')."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Path to MIKASA-Robo root. Defaults to auto-detect from script location.",
    )
    parser.add_argument(
        "--npz-root",
        default="data_mikasa_robo/data_npz",
        help="Relative path (from repo root) to source .npz task directories.",
    )
    parser.add_argument(
        "--rlds-root",
        default="data_mikasa_robo/data_rlds",
        help="Relative path (from repo root) for converted RLDS output.",
    )
    parser.add_argument(
        "--builder-dir",
        default="utils/convert_npz_to_rlds/rlds_dataset_builder/mikasa_dataset",
        help="Relative path (from repo root) to TFDS dataset builder directory.",
    )
    parser.add_argument(
        "--dataset-name",
        default="mikasa_dataset",
        help="TFDS dataset folder name generated under ~/tensorflow_datasets.",
    )
    parser.add_argument(
        "--tfds-data-dir",
        default=None,
        help=(
            "Directory for TFDS build artifacts. "
            "If omitted, a per-run temporary directory is used (safe for parallel runs)."
        ),
    )
    parser.add_argument(
        "--overwrite-dest",
        action="store_true",
        help="Remove existing destination directory for this task before copy.",
    )
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"+ (cd {cwd} && {' '.join(cmd)})")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    dataset_name = env_id_to_dataset_name(args.task)
    npz_root = repo_root / args.npz_root
    task_dir = npz_root / dataset_name
    builder_dir = repo_root / args.builder_dir
    version_name = "1.0.0"
    destination_task_dir = repo_root / args.rlds_root / dataset_name
    destination_version_dir = destination_task_dir / version_name
    source_metadata = task_dir / "metadata.json"
    destination_metadata = destination_version_dir / "metadata.json"

    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")
    if not source_metadata.exists():
        raise FileNotFoundError(f"Task metadata.json not found: {source_metadata}")
    if not builder_dir.exists():
        raise FileNotFoundError(f"Builder dir not found: {builder_dir}")

    env = os.environ.copy()
    env["MIKASA_TASK_NAME"] = dataset_name
    try:
        env["MIKASA_NPZ_ROOT"] = str(npz_root.relative_to(repo_root))
    except ValueError:
        env["MIKASA_NPZ_ROOT"] = str(npz_root)

    # Use per-run TFDS output dir by default to avoid collisions in parallel runs.
    temp_data_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.tfds_data_dir:
        tfds_data_dir = Path(args.tfds_data_dir).expanduser().resolve()
        tfds_data_dir.mkdir(parents=True, exist_ok=True)
    else:
        safe_task = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in dataset_name)
        temp_data_dir_obj = tempfile.TemporaryDirectory(prefix=f"mikasa_tfds_{safe_task}_")
        tfds_data_dir = Path(temp_data_dir_obj.name)

    tfds_dataset_dir = tfds_data_dir / args.dataset_name
    source_version_dir = tfds_dataset_dir / version_name
    env["TFDS_DATA_DIR"] = str(tfds_data_dir)

    try:
        # Use the same interpreter as this script so `tfds` matches `uv run --project ...` deps.
        # Calling bare `tfds` uses PATH and can pick another venv (e.g. convert_rlds_to_lerobot)
        # where tensorflow_datasets imports apache_beam but beam is not installed.
        run(
            [
                sys.executable,
                "-m",
                "tensorflow_datasets.scripts.cli.main",
                "build",
                "--overwrite",
                f"--data_dir={tfds_data_dir}",
            ],
            cwd=builder_dir,
            env=env,
        )

        if not source_version_dir.exists():
            raise FileNotFoundError(
                f"TFDS output version dir not found: {source_version_dir}. tfds build may have failed."
            )

        if destination_task_dir.exists() and args.overwrite_dest:
            shutil.rmtree(destination_task_dir)
        elif destination_task_dir.exists() and not args.overwrite_dest:
            raise FileExistsError(
                f"Destination already exists: {destination_task_dir}. Use --overwrite-dest to replace it."
            )

        destination_task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_version_dir, destination_version_dir)
        shutil.copy2(source_metadata, destination_metadata)
    finally:
        if temp_data_dir_obj is not None:
            temp_data_dir_obj.cleanup()

    print(f"RLDS dataset written to: {destination_task_dir}")
    print(f"Metadata copied to: {destination_metadata}")


if __name__ == "__main__":
    main()
