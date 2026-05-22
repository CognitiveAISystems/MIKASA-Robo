import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import mikasa_robo_suite.vla.memory_envs  # noqa: F401
from baselines.ppo.ppo_memtasks import AgentStateOnly, FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.utils.dataset_naming import env_id_to_dataset_name
from mikasa_robo_suite.vla.utils.wrappers import *

DATA_NPZ_DIRNAME = "data_npz"
BATCHED_TMP_SUBDIR = "_batched"
MIN_EPISODE_LENGTH_TO_SAVE = 10
DEFAULT_BATCH_SIZE = 10
BATCH_SIZE_OVERRIDE_BY_ENV = {
    "ShellGamePick-VLA-v0": 1,
}


def _env_flag_is_true(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


LOG_EPISODE_LENGTHS = _env_flag_is_true("MIKASA_LOG_EPISODE_LENGTHS", "0")


SHELL_GAME_CORE_ENVS = {
    "ShellGameTouch-VLA-v0",
    "ShellGamePush-VLA-v0",
    "ShellGamePick-VLA-v0",
}

INTERCEPT_ENVS = {
    "InterceptSlow-VLA-v0",
    "InterceptMedium-VLA-v0",
    "InterceptFast-VLA-v0",
    "InterceptGrabSlow-VLA-v0",
    "InterceptGrabMedium-VLA-v0",
    "InterceptGrabFast-VLA-v0",
}

ROTATE_LENIENT_ENVS = {
    "RotateLenientPos-VLA-v0",
    "RotateLenientPosNeg-VLA-v0",
}

ROTATE_STRICT_ENVS = {
    "RotateStrictPos-VLA-v0",
    "RotateStrictPosNeg-VLA-v0",
}

COLOR_MEMORY_ENVS = {
    "RememberColor3-VLA-v0",
    "RememberColor5-VLA-v0",
    "RememberColor9-VLA-v0",
    "FindImposterColor3-VLA-v0",
    "FindImposterColor5-VLA-v0",
    "FindImposterColor9-VLA-v0",
}

SHAPE_MEMORY_ENVS = {
    "RememberShape3-VLA-v0",
    "RememberShape5-VLA-v0",
    "RememberShape9-VLA-v0",
    "FindImposterShape3-VLA-v0",
    "FindImposterShape5-VLA-v0",
    "FindImposterShape9-VLA-v0",
}

SHAPE_COLOR_MEMORY_ENVS = {
    "RememberShapeAndColor3x2-VLA-v0",
    "RememberShapeAndColor3x3-VLA-v0",
    "RememberShapeAndColor5x3-VLA-v0",
    "FindImposterShapeAndColor3x2-VLA-v0",
    "FindImposterShapeAndColor3x3-VLA-v0",
    "FindImposterShapeAndColor5x3-VLA-v0",
}

MEMORY_CAPACITY_ENVS = {
    "BunchOfColors3-VLA-v0",
    "BunchOfColors5-VLA-v0",
    "BunchOfColors7-VLA-v0",
    "SeqOfColors3-VLA-v0",
    "SeqOfColors5-VLA-v0",
    "SeqOfColors7-VLA-v0",
    "ChainOfColors3-VLA-v0",
    "ChainOfColors5-VLA-v0",
    "ChainOfColors7-VLA-v0",
}

SHELL_GAME_LAMP_ENVS = {
    "ShellGameShuffleColorLampTouch-VLA-v0",
    "ShellGameColorLampTouch-VLA-v0",
}

EPISODE_TIMEOUT_BY_ENV = {
    "ShellGameTouch-VLA-v0": 30,
    "ShellGamePush-VLA-v0": 30,
    "ShellGamePick-VLA-v0": 30,
    "InterceptSlow-VLA-v0": 60,
    "InterceptMedium-VLA-v0": 60,
    "InterceptFast-VLA-v0": 60,
    "InterceptGrabSlow-VLA-v0": 60,
    "InterceptGrabMedium-VLA-v0": 60,
    "InterceptGrabFast-VLA-v0": 60,
    "RotateLenientPos-VLA-v0": 60,
    "RotateLenientPosNeg-VLA-v0": 60,
    "RotateStrictPos-VLA-v0": 90,
    "RotateStrictPosNeg-VLA-v0": 90,
    "TakeItBack-VLA-v0": 60,
    "RememberColor3-VLA-v0": 25,
    "RememberColor5-VLA-v0": 25,
    "RememberColor9-VLA-v0": 25,
    "RememberShape3-VLA-v0": 25,
    "RememberShape5-VLA-v0": 25,
    "RememberShape9-VLA-v0": 25,
    "RememberShapeAndColor3x2-VLA-v0": 25,
    "RememberShapeAndColor3x3-VLA-v0": 25,
    "RememberShapeAndColor5x3-VLA-v0": 25,
    "BunchOfColors3-VLA-v0": 100,
    "BunchOfColors5-VLA-v0": 100,
    "BunchOfColors7-VLA-v0": 100,
    "SeqOfColors3-VLA-v0": 100,
    "SeqOfColors5-VLA-v0": 100,
    "SeqOfColors7-VLA-v0": 100,
    "ChainOfColors3-VLA-v0": 100,
    "ChainOfColors5-VLA-v0": 100,
    "ChainOfColors7-VLA-v0": 100,
    "ShellGameShuffleTouch-VLA-v0": 60,
    "ShellGameShuffleColorLampTouch-VLA-v0": 60,
    "ShellGameColorLampTouch-VLA-v0": 30,
    "FindImposterColor3-VLA-v0": 25,
    "FindImposterColor5-VLA-v0": 25,
    "FindImposterColor9-VLA-v0": 25,
    "FindImposterShape3-VLA-v0": 25,
    "FindImposterShape5-VLA-v0": 25,
    "FindImposterShape9-VLA-v0": 25,
    "FindImposterShapeAndColor3x2-VLA-v0": 25,
    "FindImposterShapeAndColor3x3-VLA-v0": 25,
    "FindImposterShapeAndColor5x3-VLA-v0": 40,
}


def env_info(env_id: str):
    if env_id not in EPISODE_TIMEOUT_BY_ENV:
        raise ValueError(f"Unknown environment: {env_id}. Supported envs: {sorted(EPISODE_TIMEOUT_BY_ENV.keys())}")

    if env_id in SHELL_GAME_CORE_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in INTERCEPT_ENVS:
        wrappers_list = [
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in ROTATE_LENIENT_ENVS or env_id in ROTATE_STRICT_ENVS:
        wrappers_list = [
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (RotateRenderAngleInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id == "TakeItBack-VLA-v0":
        wrappers_list = [
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in COLOR_MEMORY_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RememberColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in SHAPE_MEMORY_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RememberShapeInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in SHAPE_COLOR_MEMORY_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RememberShapeAndColorInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in MEMORY_CAPACITY_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (MemoryCapacityInfoWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id == "ShellGameShuffleTouch-VLA-v0":
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    elif env_id in SHELL_GAME_LAMP_ENVS:
        wrappers_list = [
            (CurriculumPhaseNoopActionWrapper, {}),
            (RenderStepInfoWrapper, {}),
            (ShellGameRenderCupInfoWrapper, {}),
            (RenderRewardInfoWrapper, {}),
            (DebugRewardWrapper, {}),
        ]
    else:
        raise ValueError(f"Unknown environment: {env_id}")

    wrappers_list.insert(0, (StateOnlyTensorToDictWrapper, {}))
    return wrappers_list, EPISODE_TIMEOUT_BY_ENV[env_id]


def _to_uint8_rgb_batch(rgb_batch: torch.Tensor) -> np.ndarray:
    rgb_np = rgb_batch.detach().cpu().numpy()
    if rgb_np.dtype == np.uint8:
        return rgb_np

    rgb_np = rgb_np.astype(np.float32, copy=False)
    max_val = float(np.nanmax(rgb_np)) if rgb_np.size > 0 else 0.0
    if max_val <= 1.0 + 1e-5:
        rgb_np = rgb_np * 255.0
    return np.clip(rgb_np, 0.0, 255.0).astype(np.uint8, copy=False)


def npz_layout_roots(path_to_save_data: str) -> tuple[Path, Path]:
    base = Path(path_to_save_data)
    npz_root = base / DATA_NPZ_DIRNAME
    batched_root = npz_root / BATCHED_TMP_SUBDIR
    return npz_root, batched_root


def _extract_language_instruction_single(info: dict) -> str:
    if "language_instruction" not in info:
        raise KeyError("language_instruction is missing in env info.")

    language = info["language_instruction"]
    if isinstance(language, str):
        return language

    if isinstance(language, np.ndarray):
        if language.ndim == 0:
            return str(language.item())
        return str(language.reshape(-1)[0])

    if isinstance(language, (list, tuple)):
        if not language:
            raise ValueError("language_instruction list is empty.")
        return str(language[0])

    if torch.is_tensor(language):
        if language.ndim == 0:
            return str(language.item())
        return str(language.detach().cpu().reshape(-1)[0].item())

    return str(language)


def _to_scalar_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if torch.is_tensor(x):
        return bool(x.reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return bool(np.asarray(x).reshape(-1)[0])
    return bool(x)


def _to_bool_batch(x, batch_size: int) -> np.ndarray:
    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 0:
        return np.full((batch_size,), bool(arr.item()), dtype=np.bool_)

    arr = arr.reshape(-1)
    if arr.shape[0] == 1:
        return np.full((batch_size,), bool(arr[0]), dtype=np.bool_)

    if arr.shape[0] != batch_size:
        raise ValueError(f"Expected batch dimension {batch_size}, got {arr.shape[0]} for bool batch")

    return arr.astype(np.bool_, copy=False)


def _extract_language_instruction_batch(info: dict, batch_size: int) -> list[str]:
    if "language_instruction" not in info:
        raise KeyError("language_instruction is missing in env info.")

    language = info["language_instruction"]
    if isinstance(language, str):
        return [language for _ in range(batch_size)]

    if torch.is_tensor(language):
        language = language.detach().cpu().numpy()

    if isinstance(language, np.ndarray):
        if language.ndim == 0:
            return [str(language.item()) for _ in range(batch_size)]
        flat = language.reshape(-1)
        if flat.shape[0] == 1:
            return [str(flat[0]) for _ in range(batch_size)]
        if flat.shape[0] != batch_size:
            raise ValueError(f"Expected batch dimension {batch_size}, got {flat.shape[0]} for language_instruction")
        return [str(x) for x in flat]

    if isinstance(language, (list, tuple)):
        if len(language) == 1:
            return [str(language[0]) for _ in range(batch_size)]
        if len(language) != batch_size:
            raise ValueError(f"Expected batch dimension {batch_size}, got {len(language)} for language_instruction")
        return [str(x) for x in language]

    # Fallback: same value for every env.
    return [str(language) for _ in range(batch_size)]


def get_list_of_all_checkpoints_available(ckpt_dir: str = "."):
    oracle_checkpoints_dir = os.path.join(ckpt_dir, "oracle_checkpoints")

    if not os.path.exists(oracle_checkpoints_dir):
        raise FileNotFoundError(f"Directory {oracle_checkpoints_dir} does not exist.")

    # Merge checkpoints from both layouts:
    # 1) legacy: oracle_checkpoints/<env_id>/.../final_success_ckpt.pt
    # 2) normalized: oracle_checkpoints/ppo_memtasks/state/normalized_dense/<env_id>/.../final_success_ckpt.pt
    # Keep one checkpoint per env, preferring newer mtime.
    latest_by_env: dict[str, tuple[float, str]] = {}

    def register_checkpoint(env_id: str, ckpt_path: str):
        if not env_id:
            return
        try:
            mtime = os.path.getmtime(ckpt_path)
        except OSError:
            return

        prev = latest_by_env.get(env_id)
        if prev is None or mtime >= prev[0]:
            latest_by_env[env_id] = (mtime, ckpt_path)

    normalized_dense_dir = os.path.join(
        oracle_checkpoints_dir,
        "ppo_memtasks",
        "state",
        "normalized_dense",
    )
    if os.path.isdir(normalized_dense_dir):
        for env_dir in os.listdir(normalized_dense_dir):
            env_path = os.path.join(normalized_dense_dir, env_dir)
            if not os.path.isdir(env_path):
                continue
            for root, _, files in os.walk(env_path):
                if "final_success_ckpt.pt" in files:
                    register_checkpoint(env_dir, os.path.join(root, "final_success_ckpt.pt"))

    skip_top_level = {"ppo_memtasks"}
    for env_dir in os.listdir(oracle_checkpoints_dir):
        if env_dir in skip_top_level:
            continue
        env_path = os.path.join(oracle_checkpoints_dir, env_dir)
        if not os.path.isdir(env_path):
            continue
        for root, _, files in os.walk(env_path):
            if "final_success_ckpt.pt" in files:
                register_checkpoint(env_dir, os.path.join(root, "final_success_ckpt.pt"))

    return [[env_id, ckpt_path] for env_id, (_, ckpt_path) in sorted(latest_by_env.items())]


def collect_batched_data_from_ckpt(
    env_id: str = "ShellGameTouch-VLA-v0",
    checkpoint_path: Optional[str] = None,
    path_to_save_data: str = "data_mikasa_robo",
    num_train_data: int = 250,
):
    """Collect episodes in batches; keep a batch only if all episodes are successful."""

    target_successful_episodes = num_train_data
    batch_size = BATCH_SIZE_OVERRIDE_BY_ENV.get(env_id, DEFAULT_BATCH_SIZE)
    if target_successful_episodes <= 0:
        raise ValueError(f"num_train_data must be > 0, got {target_successful_episodes}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size} for env_id={env_id}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_kwargs_state = dict(
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )

    env_kwargs_rgb = dict(
        obs_mode="rgb",
        control_mode="pd_ee_delta_pose",
        render_mode="all",
        sim_backend="gpu",
        reward_mode="normalized_dense",
    )

    env_state = gym.make(env_id, num_envs=batch_size, **env_kwargs_state)
    env_rgb = gym.make(env_id, num_envs=batch_size, **env_kwargs_rgb)

    wrappers_list, episode_timeout = env_info(env_id)

    for wrapper_class, wrapper_kwargs in wrappers_list:
        env_state = wrapper_class(env_state, **wrapper_kwargs)

    env_state = FlattenRGBDObservationWrapper(
        env_state,
        rgb=False,
        depth=False,
        state=True,
        oracle=False,
        joints=False,
    )

    for wrapper_class, wrapper_kwargs in wrappers_list:
        env_rgb = wrapper_class(env_rgb, **wrapper_kwargs)

    env_rgb = FlattenRGBDObservationWrapper(
        env_rgb,
        rgb=True,
        depth=False,
        state=False,
        oracle=False,
        joints=True,
    )
    env_rgb = ConvertJointsToEEFXyzRpyGripperWrapper(env_rgb)

    if isinstance(env_state.action_space, gym.spaces.Dict):
        env_state = FlattenActionSpaceWrapper(env_state)
    if isinstance(env_rgb.action_space, gym.spaces.Dict):
        env_rgb = FlattenActionSpaceWrapper(env_rgb)

    env_state = ManiSkillVectorEnv(
        env_state,
        batch_size,
        ignore_terminations=True,
        record_metrics=True,
    )
    env_rgb = ManiSkillVectorEnv(
        env_rgb,
        batch_size,
        ignore_terminations=True,
        record_metrics=True,
    )

    agent = AgentStateOnly(env_state).to(device)
    agent.load_state_dict(torch.load(checkpoint_path, map_location=device))
    agent.eval()

    dataset_name = env_id_to_dataset_name(env_id)
    _, batched_root = npz_layout_roots(path_to_save_data)
    save_dir = batched_root / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Collecting {target_successful_episodes} successful episodes for {env_id} "
        f"(timeout={episode_timeout}, batch_size={batch_size}, "
        f"keep_only_full_success_batches=True, min_episode_len={MIN_EPISODE_LENGTH_TO_SAVE}, "
        f"log_episode_lengths={int(LOG_EPISODE_LENGTHS)})"
    )

    saved_successful = 0
    attempted_batches = 0
    accepted_batches = 0
    skipped_short_episodes = 0

    progress = tqdm(total=target_successful_episodes, desc="Successful episodes", unit="ep")
    while saved_successful < target_successful_episodes:
        remaining_needed = target_successful_episodes - saved_successful
        seeds = [attempted_batches * batch_size + i for i in range(batch_size)]
        attempted_batches += 1

        obs_state, _ = env_state.reset(seed=seeds)
        obs_rgb, _ = env_rgb.reset(seed=seeds)

        rgb_steps = []
        proprio_steps = []
        action_steps = []
        reward_steps = []
        success_steps = []
        done_steps = []

        language_by_env: list[Optional[str]] = [None for _ in range(batch_size)]
        success_once = np.zeros((batch_size,), dtype=np.bool_)
        first_done_step = np.full((batch_size,), -1, dtype=np.int32)

        total_steps = 0
        for step_idx in range(episode_timeout):
            rgb_steps.append(_to_uint8_rgb_batch(obs_rgb["rgb"]))
            proprio_np = obs_rgb["proprio"].detach().cpu().numpy().astype(np.float32, copy=False)
            if proprio_np.ndim != 2 or proprio_np.shape[0] != batch_size or proprio_np.shape[1] != 7:
                raise RuntimeError(
                    "Expected proprio observation shape (batch_size, 7) for "
                    "eef xyz+rpy+gripper proprio, got "
                    f"{proprio_np.shape}."
                )
            proprio_steps.append(proprio_np)

            with torch.no_grad():
                for key, value in obs_state.items():
                    obs_state[key] = value.to(device)
                action = agent.get_action(obs_state, deterministic=True)

            action_np = action.detach().cpu().numpy().astype(np.float32, copy=False)
            # Clip to [-1, 1] to match what the controller actually receives
            # (PDEEPoseController clips internally; store the effective action).
            action_np = np.clip(action_np, -1.0, 1.0)
            obs_state, _, _, _, _ = env_state.step(action)
            obs_rgb, reward_rgb, term_rgb, trunc_rgb, info_rgb = env_rgb.step(action)

            batch_language = _extract_language_instruction_batch(info_rgb, batch_size)
            for i, lang in enumerate(batch_language):
                if language_by_env[i] is None:
                    language_by_env[i] = lang

            reward_np = reward_rgb.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
            success_np = _to_bool_batch(info_rgb["success"], batch_size)
            term_np = _to_bool_batch(term_rgb, batch_size)
            trunc_np = _to_bool_batch(trunc_rgb, batch_size)
            done_np = np.logical_or(np.logical_or(term_np, trunc_np), success_np)

            success_once = np.logical_or(success_once, success_np)

            just_finished = np.logical_and(done_np, first_done_step < 0)
            if np.any(just_finished):
                first_done_step[just_finished] = step_idx + 1

            action_steps.append(action_np)
            reward_steps.append(reward_np)
            success_steps.append(success_np.astype(np.int32, copy=False))
            done_steps.append(done_np.astype(np.int32, copy=False))

            total_steps = step_idx + 1
            if np.all(first_done_step >= 0):
                break

        episode_lengths = np.where(first_done_step > 0, first_done_step, total_steps).astype(np.int32, copy=False)
        max_to_take = min(batch_size, remaining_needed)
        batch_all_success = bool(np.all(success_once))

        if LOG_EPISODE_LENGTHS:
            for env_idx in range(batch_size):
                ep_len = int(episode_lengths[env_idx])
                ep_success = bool(success_once[env_idx])
                is_short = ep_len < MIN_EPISODE_LENGTH_TO_SAVE

                if not ep_success:
                    reason = "unsuccessful"
                    saved_flag = 0
                elif is_short:
                    reason = "too_short"
                    saved_flag = 0
                elif env_idx >= max_to_take:
                    reason = "extra_for_target"
                    saved_flag = 0
                elif not batch_all_success:
                    reason = "batch_not_full_success"
                    saved_flag = 0
                else:
                    reason = "saved"
                    saved_flag = 1

                print(
                    f"[EPISODE] env={env_id} batch_attempt={attempted_batches} idx={env_idx} "
                    f"seed={seeds[env_idx]} length={ep_len} success={int(ep_success)} "
                    f"saved={saved_flag} reason={reason}",
                    flush=True,
                )

        if not batch_all_success:
            progress.set_postfix(attempted_batches=attempted_batches, accepted_batches=accepted_batches)
            continue

        # Convert step-major lists into arrays: (T, B, ...)
        rgb_arr = np.stack(rgb_steps, axis=0)
        proprio_arr = np.stack(proprio_steps, axis=0)
        action_arr = np.stack(action_steps, axis=0)
        reward_arr = np.stack(reward_steps, axis=0)
        success_arr = np.stack(success_steps, axis=0)
        done_arr = np.stack(done_steps, axis=0)

        episodes_in_this_batch = []
        for env_idx in range(max_to_take):
            ep_len = int(episode_lengths[env_idx])
            if ep_len < MIN_EPISODE_LENGTH_TO_SAVE:
                skipped_short_episodes += 1
                continue

            language_instruction = language_by_env[env_idx]
            if language_instruction is None:
                raise RuntimeError("Failed to capture language_instruction for at least one env in batch.")

            episode_data = {
                "rgb": rgb_arr[:ep_len, env_idx].astype(np.uint8, copy=False),
                "proprio": proprio_arr[:ep_len, env_idx].astype(np.float32, copy=False),
                "action": action_arr[:ep_len, env_idx].astype(np.float32, copy=False),
                "reward": reward_arr[:ep_len, env_idx].astype(np.float32, copy=False),
                "success": success_arr[:ep_len, env_idx].astype(np.int32, copy=False),
                "done": done_arr[:ep_len, env_idx].astype(np.int32, copy=False),
                "language_instruction": np.array(language_instruction, dtype=np.str_),
                "success_once": np.array(True, dtype=np.bool_),
                "episode_length": np.array(ep_len, dtype=np.int32),
                "episode_seed": np.array(seeds[env_idx], dtype=np.int64),
            }
            episodes_in_this_batch.append(episode_data)

        if len(episodes_in_this_batch) == 0:
            progress.set_postfix(
                attempted_batches=attempted_batches,
                accepted_batches=accepted_batches,
                skipped_short=skipped_short_episodes,
            )
            continue

        accepted_batches += 1
        batch_payload = np.array(episodes_in_this_batch, dtype=object)
        batch_file = save_dir / f"train_data_batch_{accepted_batches - 1}.npz"
        np.savez(
            batch_file,
            episode_data=batch_payload,
            batch_size=np.array(len(episodes_in_this_batch), dtype=np.int32),
            all_success_once=np.array(True, dtype=np.bool_),
        )

        saved_successful += len(episodes_in_this_batch)
        progress.update(len(episodes_in_this_batch))
        progress.set_postfix(
            attempted_batches=attempted_batches,
            accepted_batches=accepted_batches,
            skipped_short=skipped_short_episodes,
        )

    progress.close()
    env_state.close()
    env_rgb.close()

    print(
        f"Saved {saved_successful} successful episodes "
        f"(attempted_batches={attempted_batches}, accepted_batches={accepted_batches}, "
        f"skipped_short={skipped_short_episodes}) to {save_dir}"
    )


def collect_unbatched_data_from_batched(
    env_id: str = "ShellGameTouch-VLA-v0",
    path_to_save_data: str = "data_mikasa_robo",
):
    dataset_name = env_id_to_dataset_name(env_id)
    npz_root, batched_root = npz_layout_roots(path_to_save_data)
    batched_dir = batched_root / dataset_name
    unbatched_dir = npz_root / dataset_name
    unbatched_dir.mkdir(parents=True, exist_ok=True)

    batch_files = sorted(batched_dir.glob("train_data_*.npz"))
    print(f"Unbatching {batched_dir}, {len(batch_files)} files")

    episode_lengths = []
    success_once_list = []
    reward_sums = []
    seeds = []

    traj_idx = 0
    for batch_file in tqdm(batch_files):
        npz_obj = np.load(batch_file, allow_pickle=True)
        data = {key: npz_obj[key] for key in npz_obj.keys()}

        if "episode_data" in data:
            episodes = list(np.asarray(data["episode_data"], dtype=object).reshape(-1))
            for episode_data in episodes:
                success_once = bool(np.asarray(episode_data.get("success_once", np.array(False))).reshape(-1)[0])
                if not success_once:
                    continue

                ep_len = int(
                    np.asarray(episode_data.get("episode_length", np.array(episode_data["action"].shape[0]))).reshape(
                        -1
                    )[0]
                )
                if ep_len < MIN_EPISODE_LENGTH_TO_SAVE:
                    continue

                out_file = unbatched_dir / f"train_data_{traj_idx}.npz"
                np.savez(out_file, **episode_data)

                ep_reward_sum = float(np.asarray(episode_data["reward"], dtype=np.float32).sum())
                ep_seed = int(np.asarray(episode_data.get("episode_seed", np.array(-1))).reshape(-1)[0])

                episode_lengths.append(ep_len)
                success_once_list.append(True)
                reward_sums.append(ep_reward_sum)
                seeds.append(ep_seed)
                traj_idx += 1
        else:
            # Backward compatibility with old format (one episode per .npz file)
            success_once = bool(data.get("success_once", np.array(False)).reshape(-1)[0])
            if not success_once:
                continue

            ep_len = int(data.get("episode_length", np.array(data["action"].shape[0])).reshape(-1)[0])
            if ep_len < MIN_EPISODE_LENGTH_TO_SAVE:
                continue

            out_file = unbatched_dir / f"train_data_{traj_idx}.npz"
            np.savez(out_file, **data)

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

    # Remove only batched files here. Caller can delete dir if it becomes empty.
    for batch_file in batch_files:
        try:
            batch_file.unlink()
        except FileNotFoundError:
            pass


def maybe_remove_empty_batched_dir(env_id: str, path_to_save_data: str = "data_mikasa_robo"):
    dataset_name = env_id_to_dataset_name(env_id)
    _, batched_root = npz_layout_roots(path_to_save_data)
    batched_dir = batched_root / dataset_name
    if batched_dir.exists() and batched_dir.is_dir() and not any(batched_dir.iterdir()):
        batched_dir.rmdir()
        print(f"Deleted empty batched dir: {batched_dir}")
    if batched_root.exists() and batched_root.is_dir() and not any(batched_root.iterdir()):
        batched_root.rmdir()
        print(f"Deleted empty temporary batched root: {batched_root}")


def collect_for_env(
    env_id: str,
    checkpoint: str,
    path_to_save_data: str,
    num_train_data: int,
):
    collect_batched_data_from_ckpt(
        env_id=env_id,
        checkpoint_path=checkpoint,
        path_to_save_data=path_to_save_data,
        num_train_data=num_train_data,
    )

    collect_unbatched_data_from_batched(
        env_id=env_id,
        path_to_save_data=path_to_save_data,
    )

    maybe_remove_empty_batched_dir(env_id=env_id, path_to_save_data=path_to_save_data)


@dataclass
class Args:
    env_id: Optional[str] = "ShellGameTouch-VLA-v0"
    path_to_save_data: str = "data_mikasa_robo"
    ckpt_dir: str = "."
    num_train_data: int = 250


if __name__ == "__main__":
    args = tyro.cli(Args)

    checkpoints = get_list_of_all_checkpoints_available(ckpt_dir=args.ckpt_dir)
    checkpoint_map = {env: ckpt for env, ckpt in checkpoints}

    if args.env_id not in checkpoint_map:
        available = sorted(checkpoint_map.keys())
        raise ValueError(f"Checkpoint for env_id={args.env_id} not found. Available: {available}")

    print(f"Collecting data for {args.env_id} from {checkpoint_map[args.env_id]}")
    collect_for_env(
        env_id=args.env_id,
        checkpoint=checkpoint_map[args.env_id],
        path_to_save_data=args.path_to_save_data,
        num_train_data=args.num_train_data,
    )

# python3 mikasa_robo_suite/vla/dataset_collectors/get_mikasa_robo_datasets.py --env-id=ShellGameTouch-VLA-v0 --path-to-save-data=data_mikasa_robo --ckpt-dir=. --num-train-data=250
