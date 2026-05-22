import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from mikasa_robo_suite.vla.utils.dataset_naming import env_id_to_dataset_name  # noqa: E402

DATA_NPZ_DIRNAME = "data_npz"
BATCHED_TMP_SUBDIR = "_batched"


"""
python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id BlinkCountButtonPressEasy-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000

python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id TraceShapeHard-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000

python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id TraceShapeSeqHard-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000

python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
  --env-id GatherAndRecall9-VLA-v0 \
  --path-to-save-data data_mikasa_robo \
  --num-train-data 250 \
  --max-attempts 5000
"""


BATTERIES_AND_BLINK_ENVS = {
    "BatteriesCheckerEasy-3-VLA-v0",
    "BatteriesCheckerEasy-6-VLA-v0",
    "BatteriesCheckerHard-3-VLA-v0",
    "BatteriesCheckerHard-6-VLA-v0",
    "BlinkCountButtonPressEasy-VLA-v0",
    "BlinkCountButtonPressMedium-VLA-v0",
    "BlinkCountButtonPressHard-VLA-v0",
}

BLINK_LONG_ENVS = {
    "BlinkCountButtonPressEasy-Long-VLA-v0",
    "BlinkCountButtonPressMedium-Long-VLA-v0",
    "BlinkCountButtonPressHard-Long-VLA-v0",
}

REMEMBER_LONG_ENVS = {
    "RememberColor3-Long-VLA-v0",
    "RememberColor5-Long-VLA-v0",
    "RememberColor9-Long-VLA-v0",
    "RememberShape3-Long-VLA-v0",
    "RememberShape5-Long-VLA-v0",
    "RememberShape9-Long-VLA-v0",
    "RememberShapeAndColor3x2-Long-VLA-v0",
    "RememberShapeAndColor3x3-Long-VLA-v0",
    "RememberShapeAndColor5x3-Long-VLA-v0",
}

MEMORY_CAPACITY_COLORS_ENVS = {
    "BunchOfColors3-VLA-v0",
    "BunchOfColors5-VLA-v0",
    "BunchOfColors7-VLA-v0",
    "SeqOfColors3-VLA-v0",
    "SeqOfColors5-VLA-v0",
    "SeqOfColors7-VLA-v0",
    "ChainOfColors3-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors7-VLA-v0",
    "BunchOfColors3-Long-VLA-v0",
    "BunchOfColors5-Long-VLA-v0",
    "BunchOfColors7-Long-VLA-v0",
    "SeqOfColors3-Long-VLA-v0",
    "SeqOfColors5-Long-VLA-v0",
    "SeqOfColors7-Long-VLA-v0",
    "ChainOfColors3-Long-VLA-v0",
    "ChainOfColors5-Long-VLA-v0",
    "ChainOfColors7-Long-VLA-v0",
}

SHELL_GAME_SHUFFLE_LONG_ENVS = {
    "ShellGameShuffleTouch-Long-VLA-v0",
    "ShellGameShuffleColorLampTouch-Long-VLA-v0",
}

TRACE_SHAPE_ENVS = {
    "TraceShapeEasy-VLA-v0",
    "TraceShapeMedium-VLA-v0",
    "TraceShapeHard-VLA-v0",
}

TRACE_SHAPE_SEQ_ENVS = {
    "TraceShapeSeqEasy-VLA-v0",
    "TraceShapeSeqMedium-VLA-v0",
    "TraceShapeSeqHard-VLA-v0",
}

TIMED_TRANSFER_ENVS = {
    "TimedTransferEasy-VLA-v0",
    "TimedTransferMedium-VLA-v0",
    "TimedTransferHard-VLA-v0",
    "TimedTransferEasy-Long-VLA-v0",
    "TimedTransferMedium-Long-VLA-v0",
    "TimedTransferHard-Long-VLA-v0",
}

GATHER_AND_RECALL_ENVS = {
    "GatherAndRecall1-VLA-v0",
    "GatherAndRecall3-VLA-v0",
    "GatherAndRecall5-VLA-v0",
    "GatherAndRecall7-VLA-v0",
    "GatherAndRecall9-VLA-v0",
}

SUPPORTED_ENVS = (
    BATTERIES_AND_BLINK_ENVS
    | BLINK_LONG_ENVS
    | REMEMBER_LONG_ENVS
    | MEMORY_CAPACITY_COLORS_ENVS
    | SHELL_GAME_SHUFFLE_LONG_ENVS
    | TRACE_SHAPE_ENVS
    | TRACE_SHAPE_SEQ_ENVS
    | TIMED_TRANSFER_ENVS
    | GATHER_AND_RECALL_ENVS
)

LANG_BATTERIES_EASY = (
    "Find all working batteries by inserting each one into the socket, observing the lamp result, "
    "and then pressing the button to confirm."
)
LANG_BATTERIES_HARD = (
    "Find all working batteries by inserting each one into the socket, observing the lamp result, "
    "returning it from the socket to its initial slot, and then pressing the button to confirm."
)
LANG_BLINK = (
    "Count how many times the blue lamp blinks, press the red button exactly that many times when "
    "the red lamp turns green, then press the black button to submit your answer."
)
LANG_REMEMBER_COLOR = "Observe the cube's color, wait, then touch the cube of the same color."
LANG_REMEMBER_SHAPE = "Observe the object's shape, wait, then touch the object of the same shape."
LANG_REMEMBER_SHAPE_AND_COLOR = (
    "Observe the object's shape and color, wait, then touch the object of the same shape and color."
)
LANG_MEMORY_COLORS_ANY_ORDER = (
    "Observe which colored cubes appear during the cue, wait, then touch all of them in any order "
    "and press the center button."
)
LANG_MEMORY_COLORS_SAME_ORDER = (
    "Observe which colored cubes appear during the cue, wait, then touch all of them in the same order "
    "as the cubes were shown and press the center button."
)
LANG_SHELL_GAME_SHUFFLE_TOUCH = (
    "Observe which cup hides the ball, track the cups as they shuffle, then touch the correct cup."
)
LANG_SHELL_GAME_SHUFFLE_COLOR_LAMP = (
    "Observe which color is under each cup, track the cups as they shuffle, then touch the cup matching the lamp color."
)
LANG_TRACE_SHAPE = (
    "Watch the red cube trace a shape on the table. When the lamp turns green, pick up the green cube and trace exactly "
    "the same shape."
)
LANG_TRACE_SHAPE_SEQ = (
    "Watch the red cube trace a sequence of shapes. When the lamp turns green, pick up the green cube and trace the same "
    "sequence in order. After finishing all shapes, press the button to submit your answer."
)
LANG_GATHER_AND_RECALL = (
    "Move all cubes onto the disc. A lamp will briefly flash red, green, or blue while you work. "
    "After all cubes are placed, press the button matching the flash color."
)


def _build_fallback_language() -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    for env_id in BATTERIES_AND_BLINK_ENVS:
        if env_id.startswith("BatteriesCheckerEasy-"):
            mapping[env_id] = LANG_BATTERIES_EASY
        elif env_id.startswith("BatteriesCheckerHard-"):
            mapping[env_id] = LANG_BATTERIES_HARD
        elif env_id.startswith("BlinkCountButtonPress"):
            mapping[env_id] = LANG_BLINK

    for env_id in BLINK_LONG_ENVS:
        mapping[env_id] = LANG_BLINK

    for env_id in REMEMBER_LONG_ENVS:
        if env_id.startswith("RememberColor"):
            mapping[env_id] = LANG_REMEMBER_COLOR
        elif env_id.startswith("RememberShapeAndColor"):
            mapping[env_id] = LANG_REMEMBER_SHAPE_AND_COLOR
        elif env_id.startswith("RememberShape"):
            mapping[env_id] = LANG_REMEMBER_SHAPE

    for env_id in MEMORY_CAPACITY_COLORS_ENVS:
        if env_id.startswith("ChainOfColors"):
            mapping[env_id] = LANG_MEMORY_COLORS_SAME_ORDER
        else:
            mapping[env_id] = LANG_MEMORY_COLORS_ANY_ORDER

    mapping["ShellGameShuffleTouch-Long-VLA-v0"] = LANG_SHELL_GAME_SHUFFLE_TOUCH
    mapping["ShellGameShuffleColorLampTouch-Long-VLA-v0"] = LANG_SHELL_GAME_SHUFFLE_COLOR_LAMP

    for env_id in TRACE_SHAPE_ENVS:
        mapping[env_id] = LANG_TRACE_SHAPE
    for env_id in TRACE_SHAPE_SEQ_ENVS:
        mapping[env_id] = LANG_TRACE_SHAPE_SEQ

    # Matches TimedTransferVLABaseEnv._lang_instruction with per-env DELAY_STEPS.
    mapping["TimedTransferEasy-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 100 of that count."
    )
    mapping["TimedTransferMedium-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 150 of that count."
    )
    mapping["TimedTransferHard-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 200 of that count."
    )
    mapping["TimedTransferEasy-Long-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 300 of that count."
    )
    mapping["TimedTransferMedium-Long-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 500 of that count."
    )
    mapping["TimedTransferHard-Long-VLA-v0"] = (
        "When the white lamp turns green, start counting steps from that exact moment. "
        "Move the blue cube from the green disc to the red disc exactly on step 1000 of that count."
    )

    for env_id in GATHER_AND_RECALL_ENVS:
        mapping[env_id] = LANG_GATHER_AND_RECALL

    return mapping


FALLBACK_LANGUAGE = _build_fallback_language()

_missing_language_envs = sorted(SUPPORTED_ENVS - set(FALLBACK_LANGUAGE.keys()))
_extra_language_envs = sorted(set(FALLBACK_LANGUAGE.keys()) - SUPPORTED_ENVS)
if _missing_language_envs or _extra_language_envs:
    raise RuntimeError(
        "FALLBACK_LANGUAGE must cover exactly SUPPORTED_ENVS. "
        f"missing={_missing_language_envs}, extra={_extra_language_envs}"
    )

# Tri-state capability cache for ManiSkill replay --discard-timeout in this process:
# None  -> unknown yet
# True  -> supported
# False -> known-broken (UnboundLocalError on `truncated`)
_REPLAY_DISCARD_TIMEOUT_SUPPORTED: Optional[bool] = None

ENV_ID_ALIASES = {
    "TimedTransfeHard-VLA-v0": "TimedTransferHard-VLA-v0",
    "TimedTransfeHard-Long-VLA-v0": "TimedTransferHard-Long-VLA-v0",
}

EPISODE_FILE_RE = re.compile(r"^train_data_(\d+)\.npz$")
LOG_SEED_RE = re.compile(r"seed=(\d+)")


def npz_layout_roots(path_to_save_data: str) -> Tuple[Path, Path]:
    base = Path(path_to_save_data)
    npz_root = base / DATA_NPZ_DIRNAME
    batched_root = npz_root / BATCHED_TMP_SUBDIR
    return npz_root, batched_root


def _list_episode_files_with_indices(save_dir: Path) -> list[tuple[int, Path]]:
    indexed_files: list[tuple[int, Path]] = []
    for path in save_dir.glob("train_data_*.npz"):
        match = EPISODE_FILE_RE.match(path.name)
        if match is None:
            continue
        indexed_files.append((int(match.group(1)), path))
    indexed_files.sort(key=lambda item: item[0])
    return indexed_files


def _sorted_episode_files(save_dir: Path) -> list[Path]:
    return [path for _, path in _list_episode_files_with_indices(save_dir)]


def _validate_contiguous_episode_indices(indexed_files: list[tuple[int, Path]], env_id: str) -> None:
    if not indexed_files:
        return

    indices = [idx for idx, _ in indexed_files]
    if indices[0] != 0:
        raise RuntimeError(
            f"Found existing files for {env_id} but first index is {indices[0]}, expected 0. "
            "Refusing to resume from a non-standard index layout."
        )

    expected_last = indices[-1]
    expected = list(range(expected_last + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        missing_preview = missing[:10]
        raise RuntimeError(
            f"Non-contiguous train_data indices for {env_id} in {indexed_files[0][1].parent}. "
            f"Missing indices (first 10): {missing_preview}"
        )


def _extract_episode_seed_from_npz(npz_path: Path) -> Optional[int]:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "episode_seed" not in data:
                return None
            return int(np.asarray(data["episode_seed"]).reshape(-1)[0])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to read episode_seed from {npz_path}: {e}") from e


def _infer_max_seed_from_logs(project_root: Path, env_id: str) -> Optional[int]:
    logs_root = project_root / "logs"
    if not logs_root.exists():
        return None

    max_seed: Optional[int] = None
    for log_path in logs_root.rglob(f"{env_id}.log"):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for match in LOG_SEED_RE.finditer(text):
            seed = int(match.group(1))
            if max_seed is None or seed > max_seed:
                max_seed = seed
    return max_seed


def _infer_resume_seed(
    save_dir: Path,
    env_id: str,
    requested_start_seed: int,
    project_root: Path,
) -> tuple[int, Optional[int], Optional[int], int]:
    indexed_files = _list_episode_files_with_indices(save_dir)
    _validate_contiguous_episode_indices(indexed_files=indexed_files, env_id=env_id)

    saved_successful = len(indexed_files)
    last_saved_seed = _extract_episode_seed_from_npz(indexed_files[-1][1]) if indexed_files else None
    max_logged_seed = _infer_max_seed_from_logs(project_root=project_root, env_id=env_id)

    next_seed_candidates = [int(requested_start_seed)]
    if last_saved_seed is not None:
        next_seed_candidates.append(last_saved_seed + 1)
    if max_logged_seed is not None:
        next_seed_candidates.append(max_logged_seed + 1)

    next_attempt_seed = max(next_seed_candidates)
    return saved_successful, last_saved_seed, max_logged_seed, next_attempt_seed


def _maybe_configure_vulkan_icd_env(target_env: Optional[dict] = None) -> Optional[str]:
    env = target_env if target_env is not None else os.environ
    if env.get("VK_ICD_FILENAMES"):
        return env["VK_ICD_FILENAMES"]

    # Prefer explicit NVIDIA ICD file if present.
    candidates = [
        "/etc/vulkan/icd.d/nvidia_icd.json",
        "/usr/share/vulkan/icd.d/nvidia_icd.json",
        "/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json",
    ]
    for path in candidates:
        if Path(path).exists():
            env["VK_ICD_FILENAMES"] = path
            return path
    return None


def _configure_runtime_warning_filters():
    # User requested to hide warnings during dataset collection.
    warnings.simplefilter("ignore")

    # Reduce noisy third-party logs.
    logging.getLogger("mani_skill").setLevel(logging.ERROR)
    logging.getLogger("gymnasium").setLevel(logging.ERROR)
    logging.getLogger("gymnasium.core").setLevel(logging.ERROR)


def _configure_subprocess_env(env: dict):
    _maybe_configure_vulkan_icd_env(env)
    # Hide warnings in planner/replay subprocesses as requested.
    prev_pw = env.get("PYTHONWARNINGS", "")
    env["PYTHONWARNINGS"] = "ignore" if not prev_pw else f"{prev_pw},ignore"


def planner_script_for_env(env_id: str) -> Path:
    root = Path(__file__).resolve().parents[1] / "utils" / "motion_planning"
    if env_id.startswith("BatteriesCheckerEasy-"):
        return root / "motion_planning_batteries_checker_easy.py"
    if env_id.startswith("BatteriesCheckerHard-"):
        return root / "motion_planning_batteries_checker_hard.py"
    if env_id in BLINK_LONG_ENVS:
        return root / "motion_planning_blink_count_button_press_long.py"
    if env_id.startswith("BlinkCountButtonPress"):
        return root / "motion_planning_blink_count_button_press.py"
    if env_id in REMEMBER_LONG_ENVS:
        return root / "motion_planning_remember_long.py"
    if env_id in MEMORY_CAPACITY_COLORS_ENVS:
        return root / "motion_planning_memory_capacity_colors.py"
    if env_id in SHELL_GAME_SHUFFLE_LONG_ENVS:
        return root / "motion_planning_shell_game_shuffle.py"
    if env_id in TRACE_SHAPE_ENVS:
        return root / "motion_planning_trace_shape.py"
    if env_id in TRACE_SHAPE_SEQ_ENVS:
        return root / "motion_planning_trace_shape_seq.py"
    if env_id in TIMED_TRANSFER_ENVS:
        return root / "motion_planning_timed_transfer.py"
    if env_id in GATHER_AND_RECALL_ENVS:
        return root / "motion_planning_gather_and_recall.py"
    raise ValueError(f"Unsupported env for motion planning: {env_id}")


def load_language_commands() -> Dict[str, str]:
    # Keep language commands self-contained in this script; no external markdown dependency.
    return dict(FALLBACK_LANGUAGE)


def _long_to_short_env_id(env_id: str) -> Optional[str]:
    if "-Long-" not in env_id:
        return None
    return env_id.replace("-Long-", "-", 1)


def resolve_language_instruction(env_id: str, language_map: Dict[str, str]) -> Optional[str]:
    if env_id in language_map:
        return language_map[env_id]

    short_env_id = _long_to_short_env_id(env_id)
    if short_env_id and short_env_id in language_map:
        return language_map[short_env_id]

    if env_id in FALLBACK_LANGUAGE:
        return FALLBACK_LANGUAGE[env_id]
    if short_env_id and short_env_id in FALLBACK_LANGUAGE:
        return FALLBACK_LANGUAGE[short_env_id]
    return None


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _to_bool_scalar(x) -> bool:
    arr = _to_numpy(x).reshape(-1)
    return bool(arr[0]) if arr.size > 0 else False


def _to_int_scalar(x, default: int = -1) -> int:
    arr = _to_numpy(x).reshape(-1)
    return int(arr[0]) if arr.size > 0 else int(default)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    rgb = rgb.astype(np.float32, copy=False)
    max_val = float(np.nanmax(rgb)) if rgb.size > 0 else 0.0
    if max_val <= 1.0 + 1e-5:
        rgb = rgb * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8, copy=False)


def _extract_language_instruction(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return str(value[0])

    arr = _to_numpy(value)
    if arr.size == 0:
        return None
    item = arr.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _extract_episode_seed_from_json(json_path: Path) -> Optional[int]:
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if isinstance(data, dict):
        if "actual_seed" in data:
            return int(data["actual_seed"])
        if "episode_seed" in data:
            return int(data["episode_seed"])
        episodes = data.get("episodes")
        if isinstance(episodes, list) and len(episodes) > 0:
            last = episodes[-1]
            if isinstance(last, dict) and "episode_seed" in last:
                return int(last["episode_seed"])
    return None


def _run_subprocess(cmd, cwd: Path, env: dict, tag: str) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
            env=env,
        )
        return True, proc.stdout
    except subprocess.CalledProcessError as e:
        stderr_tail = ""
        if e.stderr:
            stderr_tail = "\n" + "\n".join(e.stderr.strip().splitlines()[-40:])
        return False, f"[{tag}] exit={e.returncode}{stderr_tail}"


def _resolve_raw_trajectory_paths(run_dir: Path, trajectory_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    preferred_h5 = run_dir / f"{trajectory_name}.h5"
    if preferred_h5.exists():
        raw_h5 = preferred_h5
    else:
        h5s = sorted(run_dir.glob("*.h5"), key=lambda p: p.stat().st_mtime)
        if not h5s:
            return None, None
        raw_h5 = h5s[-1]

    preferred_json = raw_h5.with_suffix(".json")
    if preferred_json.exists():
        return raw_h5, preferred_json

    jsons = sorted(
        [p for p in run_dir.glob("*.json") if not p.name.endswith("_meta.json")],
        key=lambda p: p.stat().st_mtime,
    )
    raw_json = jsons[-1] if jsons else None
    return raw_h5, raw_json


def _h5_has_traj_groups(h5_path: Path) -> bool:
    try:
        with h5py.File(h5_path, "r") as f:
            return any(k.startswith("traj_") for k in f.keys())
    except Exception:
        return False


def _run_replay_trajectory(
    raw_h5: Path,
    run_dir: Path,
    project_root: Path,
    use_discard_timeout: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
    global _REPLAY_DISCARD_TIMEOUT_SUPPORTED
    pre_h5 = set(run_dir.glob("*.h5"))
    pre_json = set(run_dir.glob("*.json"))

    # Replay runs in a clean subprocess and must import custom env registries first,
    # otherwise gym.make(<...-VLA-v0>) fails with NameNotFound.
    replay_bootstrap = (
        "import logging, sys, warnings; "
        "warnings.simplefilter('ignore'); "
        "logging.getLogger('mani_skill').setLevel(logging.ERROR); "
        "logging.getLogger('gymnasium').setLevel(logging.ERROR); "
        "logging.getLogger('gymnasium.core').setLevel(logging.ERROR); "
        "import mikasa_robo_suite.vla.memory_envs; "
        "from mani_skill.trajectory.replay_trajectory import main, parse_args; "
        "main(parse_args(sys.argv[1:]))"
    )

    base_cmd = [
        sys.executable,
        "-c",
        replay_bootstrap,
        "--traj-path",
        str(raw_h5),
        "--target-control-mode",
        "pd_ee_delta_pose",
        "--obs-mode",
        "none",
        "--save-traj",
        "--allow-failure",
        "--num-procs",
        "1",
    ]

    env = dict(**os.environ)
    prev_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{prev_pythonpath}" if prev_pythonpath else str(project_root)
    _configure_subprocess_env(env)

    if not use_discard_timeout or _REPLAY_DISCARD_TIMEOUT_SUPPORTED is False:
        ok, msg = _run_subprocess(
            cmd=base_cmd,
            cwd=project_root,
            env=env,
            tag="replay_no_discard_timeout",
        )
        if not ok:
            return None, None, msg
    else:
        # First try (or known-supported): keep timeout filtering in replay stage.
        cmd = [*base_cmd, "--discard-timeout"]
        ok, msg = _run_subprocess(cmd=cmd, cwd=project_root, env=env, tag="replay")
        if not ok:
            # Robust fallback: if replay with --discard-timeout fails for any reason,
            # retry once without --discard-timeout.
            _REPLAY_DISCARD_TIMEOUT_SUPPORTED = False
            fallback_ok, fallback_msg = _run_subprocess(
                cmd=base_cmd,
                cwd=project_root,
                env=env,
                tag="replay_no_discard_timeout",
            )
            if not fallback_ok:
                return (
                    None,
                    None,
                    f"{msg}\n[fallback replay without --discard-timeout failed] {fallback_msg}",
                )
        else:
            _REPLAY_DISCARD_TIMEOUT_SUPPORTED = True

    post_h5 = set(run_dir.glob("*.h5"))
    post_json = set(run_dir.glob("*.json"))

    new_h5 = sorted(list(post_h5 - pre_h5), key=lambda p: p.stat().st_mtime)
    fallback_h5 = sorted(
        [p for p in post_h5 if p != raw_h5 and "pd_ee_delta_pose" in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    replay_candidates = []
    for p in [*new_h5, *fallback_h5]:
        if p not in replay_candidates:
            replay_candidates.append(p)

    replay_h5 = None
    # Prefer newest valid replay output that actually contains traj_* groups.
    for p in reversed(replay_candidates):
        if _h5_has_traj_groups(p):
            replay_h5 = p
            break

    if replay_h5 is None:
        if not replay_candidates:
            return None, None, "[replay] no converted .h5 trajectory file created"
        names = ", ".join(sorted([c.name for c in replay_candidates]))
        return None, None, f"[replay] converted .h5 produced, but none contain traj_* groups: {names}"

    replay_json = replay_h5.with_suffix(".json")
    if not replay_json.exists():
        new_json = sorted(list(post_json - pre_json), key=lambda p: p.stat().st_mtime)
        replay_json = new_json[-1] if new_json else None

    return replay_h5, replay_json, None


def _load_last_traj_actions(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted([k for k in f.keys() if k.startswith("traj_")], key=lambda x: int(x.split("_")[-1]))
        if not traj_keys:
            raise RuntimeError(f"No trajectory groups found in {h5_path}")
        traj = f[traj_keys[-1]]
        if "actions" not in traj:
            raise RuntimeError(f"Missing actions in trajectory: {h5_path}")

        actions = np.asarray(traj["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(
                f"Expected replayed pd_ee_delta_pose actions with shape (T,7), got {actions.shape} from {h5_path}"
            )
        return np.clip(actions, -1.0, 1.0)


def _validate_flatten_obs(obs: dict):
    if not isinstance(obs, dict):
        raise RuntimeError(f"Expected dict observation, got {type(obs).__name__}")
    if "rgb" not in obs or "proprio" not in obs:
        raise RuntimeError("Expected observation keys 'rgb' and 'proprio' from FlattenRGBDObservationWrapper")


def _is_batteries_env(env_id: str) -> bool:
    return env_id.startswith("BatteriesChecker")


def _rollout_pd_ee_episode(
    env,
    continue_after_success: bool,
    episode_seed: int,
    actions: np.ndarray,
    language_fallback: str,
) -> Optional[Dict[str, np.ndarray]]:
    # Kept for backward compatibility of call sites; rollout now always stops on success.
    _ = continue_after_success
    rgb_steps = []
    proprio_steps = []
    reward_steps = []
    success_steps = []
    done_steps = []

    obs, info = env.reset(seed=int(episode_seed))
    _validate_flatten_obs(obs)
    language_instruction = _extract_language_instruction(
        info.get("language_instruction") if isinstance(info, dict) else None
    )
    if not language_instruction:
        language_instruction = language_fallback

    success_once = False
    for action in actions:
        _validate_flatten_obs(obs)
        rgb_steps.append(_to_numpy(obs["rgb"])[0])
        proprio_step = _to_numpy(obs["proprio"])
        if proprio_step.ndim != 2 or proprio_step.shape[0] != 1 or proprio_step.shape[1] != 7:
            raise RuntimeError(
                f"Expected proprio observation shape (1, 7) for eef xyz+rpy+gripper proprio, got {proprio_step.shape}."
            )
        proprio_steps.append(proprio_step[0].astype(np.float32, copy=False))

        action_step = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action_step)

        if not language_instruction and isinstance(info, dict):
            maybe_lang = _extract_language_instruction(info.get("language_instruction"))
            if maybe_lang:
                language_instruction = maybe_lang

        reward_scalar = float(_to_numpy(reward).reshape(-1)[0])
        success_step = _to_bool_scalar(info.get("success", False) if isinstance(info, dict) else False)
        success_once_step = _to_bool_scalar(
            info.get("success_once", success_step) if isinstance(info, dict) else success_step
        )
        terminated_step = _to_bool_scalar(terminated)
        truncated_step = _to_bool_scalar(truncated)
        # Stop the rollout as soon as success is reached.
        done_step = success_step or terminated_step or truncated_step

        reward_steps.append(np.float32(reward_scalar))
        success_steps.append(np.int32(1 if success_step else 0))
        done_steps.append(np.int32(1 if done_step else 0))
        success_once = success_once or success_once_step

        if done_step:
            break

    if len(reward_steps) == 0:
        return None
    if not success_once:
        return None

    ep_len = len(reward_steps)
    rgb = _to_uint8_rgb(np.stack(rgb_steps, axis=0))
    proprio = np.stack(proprio_steps, axis=0).astype(np.float32, copy=False)
    action = np.asarray(actions[:ep_len], dtype=np.float32)
    reward = np.asarray(reward_steps, dtype=np.float32)
    success = np.asarray(success_steps, dtype=np.int32)
    done = np.asarray(done_steps, dtype=np.int32)

    episode_data = {
        "rgb": rgb,
        "proprio": proprio,
        "action": action,
        "reward": reward,
        "success": success,
        "done": done,
        "language_instruction": np.array(language_instruction or language_fallback, dtype=np.str_),
        "success_once": np.array(True, dtype=np.bool_),
        "episode_length": np.array(ep_len, dtype=np.int32),
        "episode_seed": np.array(int(episode_seed), dtype=np.int64),
    }
    return episode_data


def _create_pd_ee_rollout_env(env_id: str):
    _configure_runtime_warning_filters()
    _maybe_configure_vulkan_icd_env()

    import gymnasium as gym

    import mikasa_robo_suite.vla.memory_envs  # noqa: F401
    from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
    from mikasa_robo_suite.vla.utils.wrappers import (
        ConvertJointsToEEFXyzRpyGripperWrapper,
        StateOnlyTensorToDictWrapper,
    )

    make_kwargs = dict(
        id=env_id,
        num_envs=1,
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        reward_mode="normalized_dense",
    )
    try:
        env = gym.make(**make_kwargs)
    except RuntimeError as e:
        # In CPU-only contexts SAPIEN may fail on default CUDA device.
        if 'failed to find device "cuda"' not in str(e).lower():
            raise
        env = gym.make(**make_kwargs, sim_backend="cpu")

    # Keep rollout semantics uniform: terminate immediately on success for all tasks.
    continue_after_success = False

    env = StateOnlyTensorToDictWrapper(env)
    env = FlattenRGBDObservationWrapper(
        env,
        rgb=True,
        depth=False,
        state=False,
        oracle=False,
        joints=True,
    )
    env = ConvertJointsToEEFXyzRpyGripperWrapper(env)
    return env, continue_after_success


def _validate_removed_conversion_args(args):
    checks = [
        ("--ee-pos-scale", args.ee_pos_scale is None),
        ("--ee-rot-scale", args.ee_rot_scale is None),
        ("--auto-calibrate-pos-scale", int(args.auto_calibrate_pos_scale) == 1),
        ("--calibration-seed", int(args.calibration_seed) == 0),
        ("--calibration-probe-action", abs(float(args.calibration_probe_action) - 0.5) < 1e-12),
        ("--calibration-repeats", int(args.calibration_repeats) == 3),
        ("--calibration-mode", str(args.calibration_mode) == "scalar"),
        ("--split-large-steps", int(args.split_large_steps) == 1),
        ("--validate-pd-ee-replay", int(args.validate_pd_ee_replay) == 1),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        opts = ", ".join(failed)
        raise ValueError(
            "Deprecated conversion options are no longer supported in motion-planning collector: "
            f"{opts}. These options have been removed; ManiSkill replay is used instead."
        )


def collect_batched_motion_planning(
    env_id: str,
    path_to_save_data: str,
    num_train_data: int,
    max_attempts: int,
    seed: int,
):
    alias_env_id = ENV_ID_ALIASES.get(env_id)
    if alias_env_id is not None:
        print(f"[env-id alias] {env_id} -> {alias_env_id}")
        env_id = alias_env_id

    if env_id not in SUPPORTED_ENVS:
        raise ValueError(f"Unsupported env_id={env_id}. Supported: {sorted(SUPPORTED_ENVS)}")

    planner_script = planner_script_for_env(env_id)
    if not planner_script.exists():
        raise FileNotFoundError(f"Planner script not found: {planner_script}")

    project_root = Path(__file__).resolve().parents[3]

    language_map = load_language_commands()
    language_instruction = resolve_language_instruction(env_id, language_map)
    if not language_instruction:
        raise ValueError(
            f"Missing language command for env_id={env_id}. "
            "No exact match, no Long->short fallback, and no FALLBACK_LANGUAGE entry."
        )

    dataset_name = env_id_to_dataset_name(env_id)
    _, batched_root = npz_layout_roots(path_to_save_data)
    save_dir = batched_root / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix=f"mp_{env_id}_"))

    existing_saved, last_saved_seed, max_logged_seed, next_attempt_seed = _infer_resume_seed(
        save_dir=save_dir,
        env_id=env_id,
        requested_start_seed=seed,
        project_root=project_root,
    )
    if existing_saved > num_train_data:
        raise RuntimeError(
            f"Found {existing_saved} existing trajectories in {save_dir}, "
            f"but requested num_train_data={num_train_data}. Increase --num-train-data or clean this directory."
        )

    print(
        f"Collecting {num_train_data} successful episodes for {env_id} "
        f"via motion planning + ManiSkill replay from {planner_script} (start_seed={seed})"
    )
    if existing_saved > 0 or max_logged_seed is not None:
        print(
            "[resume] "
            f"existing_saved={existing_saved}, "
            f"last_saved_seed={last_saved_seed}, "
            f"max_logged_seed={max_logged_seed}, "
            f"next_attempt_seed={next_attempt_seed}"
        )

    saved_successful = existing_saved
    attempted = 0
    progress = tqdm(total=num_train_data, desc="Successful episodes", unit="ep")
    if saved_successful > 0:
        progress.update(saved_successful)
        progress.set_postfix(attempted=attempted)
    rollout_env = None
    continue_after_success = False

    try:
        rollout_env, continue_after_success = _create_pd_ee_rollout_env(env_id)
        while saved_successful < num_train_data:
            if attempted >= max_attempts:
                break

            attempt_seed = next_attempt_seed
            next_attempt_seed += 1
            attempted += 1

            run_dir = temp_root / f"seed_{attempt_seed:06d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            trajectory_name = "trajectory"

            planner_cmd = [
                sys.executable,
                str(planner_script),
                "--env-id",
                env_id,
                "--seed",
                str(attempt_seed),
                "--save-trajectory",
                "1",
                "--save-video",
                "0",
                "--trajectory-dir",
                str(run_dir),
                "--trajectory-name",
                trajectory_name,
            ]

            env = dict(**os.environ)
            prev_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{project_root}:{prev_pythonpath}" if prev_pythonpath else str(project_root)
            _configure_subprocess_env(env)

            ok, msg = _run_subprocess(cmd=planner_cmd, cwd=project_root, env=env, tag="planner")
            if not ok:
                print(f"[attempt={attempted}] planner failed for seed={attempt_seed}: {msg}")
                continue

            raw_h5, raw_json = _resolve_raw_trajectory_paths(run_dir=run_dir, trajectory_name=trajectory_name)
            if raw_h5 is None:
                print(f"[attempt={attempted}] planner produced no .h5 trajectory in {run_dir}")
                continue
            if raw_json is None or not raw_json.exists():
                print(f"[attempt={attempted}] replay requires trajectory json next to h5, missing for {raw_h5}")
                continue

            replay_h5, replay_json, replay_err = _run_replay_trajectory(
                raw_h5=raw_h5,
                run_dir=run_dir,
                project_root=project_root,
                use_discard_timeout=not _is_batteries_env(env_id),
            )
            if replay_h5 is None:
                print(f"[attempt={attempted}] replay failed for seed={attempt_seed}: {replay_err}")
                continue

            try:
                actions = _load_last_traj_actions(replay_h5)
            except Exception as e:  # noqa: BLE001
                print(f"[attempt={attempted}] invalid replayed trajectory for seed={attempt_seed}: {e}")
                continue

            seed_candidates = [
                replay_json if replay_json is not None else Path(""),
                run_dir / f"{trajectory_name}_meta.json",
                raw_json,
            ]
            episode_seed = attempt_seed
            for seed_json in seed_candidates:
                if seed_json and seed_json.exists():
                    parsed = _extract_episode_seed_from_json(seed_json)
                    if parsed is not None:
                        episode_seed = int(parsed)
                        break

            if episode_seed != attempt_seed:
                print(
                    f"[attempt={attempted}] seed remap after planner/replay: "
                    f"requested={attempt_seed} actual={episode_seed}"
                )

            try:
                episode_data = _rollout_pd_ee_episode(
                    env=rollout_env,
                    continue_after_success=continue_after_success,
                    episode_seed=episode_seed,
                    actions=actions,
                    language_fallback=language_instruction,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[attempt={attempted}] pd_ee rollout failed for seed={episode_seed}: {e}")
                continue

            if episode_data is None:
                print(f"[attempt={attempted}] seed={episode_seed} replay rollout not successful, skip")
                continue

            out_file = save_dir / f"train_data_{saved_successful}.npz"
            np.savez(out_file, **episode_data)
            saved_successful += 1
            progress.update(1)
            progress.set_postfix(attempted=attempted)
    finally:
        if rollout_env is not None:
            rollout_env.close()
        progress.close()
        shutil.rmtree(temp_root, ignore_errors=True)

    if saved_successful < num_train_data:
        raise RuntimeError(
            f"Could not collect enough successful episodes: saved={saved_successful}, "
            f"target={num_train_data}, attempted={attempted}, max_attempts={max_attempts}"
        )

    print(
        f"Saved {saved_successful} successful episodes "
        f"(existing_before_resume={existing_saved}, attempted_this_run={attempted}) to {save_dir}"
    )


def collect_unbatched_data_from_batched(env_id: str, path_to_save_data: str):
    dataset_name = env_id_to_dataset_name(env_id)
    npz_root, batched_root = npz_layout_roots(path_to_save_data)
    batched_dir = batched_root / dataset_name
    unbatched_dir = npz_root / dataset_name
    unbatched_dir.mkdir(parents=True, exist_ok=True)

    batch_files = _sorted_episode_files(batched_dir)
    if not batch_files:
        batch_files = sorted(batched_dir.glob("train_data_*.npz"))
    print(f"Unbatching {batched_dir}, {len(batch_files)} files")

    episode_lengths = []
    success_once_list = []
    reward_sums = []
    seeds = []

    traj_idx = 0
    for batch_file in tqdm(batch_files):
        episode = np.load(batch_file, allow_pickle=True)
        data = {key: episode[key] for key in episode.keys()}

        success_once = bool(data.get("success_once", np.array(False)).reshape(-1)[0])
        if not success_once:
            continue

        out_file = unbatched_dir / f"train_data_{traj_idx}.npz"
        np.savez(out_file, **data)

        ep_len = int(data.get("episode_length", np.array(data["action"].shape[0])).reshape(-1)[0])
        ep_reward_sum = float(np.asarray(data["reward"], dtype=np.float32).sum())
        ep_seed = int(data.get("episode_seed", np.array(-1)).reshape(-1)[0])

        episode_lengths.append(ep_len)
        success_once_list.append(True)
        reward_sums.append(ep_reward_sum)
        seeds.append(ep_seed)
        traj_idx += 1

    metadata = {
        "env_id": env_id,
        "num_episodes": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "success_once": success_once_list,
        "reward_sums": reward_sums,
        "episode_seeds": seeds,
        "batched_source_dir": str(batched_dir),
        "unbatched_dir": str(unbatched_dir),
    }
    metadata_path = unbatched_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for batch_file in batch_files:
        try:
            batch_file.unlink()
        except FileNotFoundError:
            pass


def maybe_remove_empty_batched_dir(env_id: str, path_to_save_data: str):
    dataset_name = env_id_to_dataset_name(env_id)
    _, batched_root = npz_layout_roots(path_to_save_data)
    batched_dir = batched_root / dataset_name
    if batched_dir.exists() and batched_dir.is_dir() and not any(batched_dir.iterdir()):
        batched_dir.rmdir()
        print(f"Deleted empty batched dir: {batched_dir}")
    if batched_root.exists() and batched_root.is_dir() and not any(batched_root.iterdir()):
        batched_root.rmdir()
        print(f"Deleted empty temporary batched root: {batched_root}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--path-to-save-data", type=str, default="data_mikasa_robo")
    parser.add_argument("--num-train-data", type=int, default=250)
    parser.add_argument("--max-attempts", type=int, default=5000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Start seed for motion-planning rollouts. Attempt i uses seed+i.",
    )

    # Deprecated legacy options kept for CLI compatibility only.
    parser.add_argument("--ee-pos-scale", type=float, default=None)
    parser.add_argument("--ee-rot-scale", type=float, default=None)
    parser.add_argument("--auto-calibrate-pos-scale", type=int, default=1)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--calibration-probe-action", type=float, default=0.5)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--calibration-mode", type=str, default="scalar")
    parser.add_argument("--split-large-steps", type=int, default=1)
    parser.add_argument("--validate-pd-ee-replay", type=int, default=1)
    return parser.parse_args()


def main():
    _configure_runtime_warning_filters()
    _maybe_configure_vulkan_icd_env()
    args = parse_args()
    _validate_removed_conversion_args(args)

    collect_batched_motion_planning(
        env_id=args.env_id,
        path_to_save_data=args.path_to_save_data,
        num_train_data=args.num_train_data,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )
    collect_unbatched_data_from_batched(
        env_id=args.env_id,
        path_to_save_data=args.path_to_save_data,
    )
    maybe_remove_empty_batched_dir(
        env_id=args.env_id,
        path_to_save_data=args.path_to_save_data,
    )


if __name__ == "__main__":
    main()

# Example:
# python mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets_motion_planning.py \
#   --env-id BatteriesCheckerEasy-3-VLA-v0 --path-to-save-data data_mikasa_robo --num-train-data 250 --seed 123
