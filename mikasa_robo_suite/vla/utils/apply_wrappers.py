"""Apply the canonical wrapper stack for MIKASA-Robo-VLA environments.

Most VLA tasks share the same outer observation wrappers, but some tasks need
an extra curriculum no-op wrapper or task-specific render overlays. This module
keeps those differences in :data:`VLA_WRAPPER_CONFIGS`.

To support a new VLA env:

1. Register the env itself with Gym/ManiSkill.
2. Add one entry to ``VLA_WRAPPER_CONFIGS`` below.
3. Pick ``overlays`` for the text/debug information rendered on videos.
4. Set ``curriculum_wrapper`` only when the train-data rollout stack freezes
   actions during the env's cue/empty phase.

Example::

    "MyTask-VLA-v0": VLAWrapperConfig(
        overlays=COLOR_MEMORY_OVERLAYS,
        curriculum_wrapper=CurriculumPhaseNoopActionWrapper,
    ),

The wrapper order is inner to outer:

1. ``StateOnlyTensorToDictWrapper`` for every env.
2. Optional ``config.curriculum_wrapper``.
3. Optional ``config.overlays`` when ``include_overlays=True``.
4. ``FlattenRGBDObservationWrapper`` for every env.
5. ``ConvertJointsToEEFXyzRpyGripperWrapper`` for every env.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
from mani_skill.utils import gym_utils

from baselines.ppo.ppo_memtasks import FlattenRGBDObservationWrapper
from mikasa_robo_suite.vla.utils.wrappers import (
    ConvertJointsToEEFXyzRpyGripperWrapper,
    CurriculumPhaseNoopActionWrapper,
    MemoryCapacityInfoWrapper,
    RememberColorInfoWrapper,
    RememberShapeAndColorInfoWrapper,
    RememberShapeInfoWrapper,
    RenderPressProgressInfoWrapper,
    RenderRewardInfoWrapper,
    RenderStepInfoWrapper,
    RenderTimedTransferInfoWrapper,
    RenderTraceShapeDebugWrapper,
    RenderWorkingBatteriesInfoWrapper,
    RotateRenderAngleInfoWrapper,
    ShellGameRenderCupInfoWrapper,
    StateOnlyTensorToDictWrapper,
)

__all__ = [
    "apply_mikasa_vla_wrappers",
    "MIKASA_VLA_ENV_IDS",
    "VLA_WRAPPER_CONFIGS",
    "VLAWrapperConfig",
]


@dataclass(frozen=True)
class VLAWrapperConfig:
    """Task-specific choices used by :func:`apply_mikasa_vla_wrappers`."""

    overlays: tuple[type, ...] = ()
    curriculum_wrapper: Optional[type] = None
    expected_control_mode: Optional[str] = None


# Render overlays are applied in this order, inner to outer. Reuse one of these
# chains in VLA_WRAPPER_CONFIGS when a new task renders the same annotations.
SHELL_GAME_OVERLAYS = (
    RenderStepInfoWrapper,
    ShellGameRenderCupInfoWrapper,
    RenderRewardInfoWrapper,
)
INTERCEPT_OVERLAYS = (
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
ROTATE_OVERLAYS = (
    # RotateRenderAngleInfoWrapper expects the uint8 frame produced by reward.
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
    RotateRenderAngleInfoWrapper,
)
COLOR_MEMORY_OVERLAYS = (
    RememberColorInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
SHAPE_MEMORY_OVERLAYS = (
    RememberShapeInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
SHAPE_COLOR_MEMORY_OVERLAYS = (
    RememberShapeAndColorInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
MEMORY_CAPACITY_OVERLAYS = (
    MemoryCapacityInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
BATTERIES_CHECKER_OVERLAYS = (
    RenderWorkingBatteriesInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
BLINK_COUNT_OVERLAYS = (
    RenderPressProgressInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
TRACE_SHAPE_OVERLAYS = (
    RenderTraceShapeDebugWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
TIMED_TRANSFER_OVERLAYS = (
    RenderTimedTransferInfoWrapper,
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)
STEP_REWARD_OVERLAYS = (
    RenderStepInfoWrapper,
    RenderRewardInfoWrapper,
)


# Add a new supported env here. Each entry describes every task-specific choice
# that apply_mikasa_vla_wrappers needs for that env.
VLA_WRAPPER_CONFIGS: dict[str, VLAWrapperConfig] = {
    # Shell game.
    "ShellGameTouch-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "ShellGamePush-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "ShellGameShuffleTouch-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "ShellGameShuffleColorLampTouch-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "ShellGameColorLampTouch-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "ShellGameShuffleTouch-Long-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS),
    "ShellGameShuffleColorLampTouch-Long-VLA-v0": VLAWrapperConfig(SHELL_GAME_OVERLAYS),
    # Intercept, rotate, and TakeItBack.
    "InterceptSlow-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "InterceptMedium-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "InterceptFast-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "InterceptGrabSlow-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "InterceptGrabMedium-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "InterceptGrabFast-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    "RotateLenientPos-VLA-v0": VLAWrapperConfig(ROTATE_OVERLAYS),
    "RotateLenientPosNeg-VLA-v0": VLAWrapperConfig(ROTATE_OVERLAYS),
    "RotateStrictPos-VLA-v0": VLAWrapperConfig(ROTATE_OVERLAYS),
    "RotateStrictPosNeg-VLA-v0": VLAWrapperConfig(ROTATE_OVERLAYS),
    "TakeItBack-VLA-v0": VLAWrapperConfig(INTERCEPT_OVERLAYS),
    # Remember color.
    "RememberColor3-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberColor5-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberColor9-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberColor3-Long-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS),
    "RememberColor5-Long-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS),
    "RememberColor9-Long-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS),
    # Remember shape.
    "RememberShape3-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShape5-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShape9-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShape3-Long-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS),
    "RememberShape5-Long-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS),
    "RememberShape9-Long-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS),
    # Remember shape and color.
    "RememberShapeAndColor3x2-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShapeAndColor3x3-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShapeAndColor5x3-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "RememberShapeAndColor3x2-Long-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS),
    "RememberShapeAndColor3x3-Long-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS),
    "RememberShapeAndColor5x3-Long-VLA-v0": VLAWrapperConfig(SHAPE_COLOR_MEMORY_OVERLAYS),
    # Find imposter.
    "FindImposterColor3-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterColor5-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterColor9-VLA-v0": VLAWrapperConfig(COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterShape3-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterShape5-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterShape9-VLA-v0": VLAWrapperConfig(SHAPE_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper),
    "FindImposterShapeAndColor3x2-VLA-v0": VLAWrapperConfig(
        SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper
    ),
    "FindImposterShapeAndColor3x3-VLA-v0": VLAWrapperConfig(
        SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper
    ),
    "FindImposterShapeAndColor5x3-VLA-v0": VLAWrapperConfig(
        SHAPE_COLOR_MEMORY_OVERLAYS, CurriculumPhaseNoopActionWrapper
    ),
    # Bunch/sequence/chain memory capacity.
    "BunchOfColors3-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "BunchOfColors5-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "BunchOfColors7-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "BunchOfColors3-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "BunchOfColors5-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "BunchOfColors7-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors3-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors5-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors7-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors3-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors5-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "SeqOfColors7-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors3-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors5-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors7-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors3-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors5-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    "ChainOfColors7-Long-VLA-v0": VLAWrapperConfig(MEMORY_CAPACITY_OVERLAYS),
    # Motion-planning datasets with task-specific overlays.
    "BatteriesCheckerEasy-3-VLA-v0": VLAWrapperConfig(BATTERIES_CHECKER_OVERLAYS),
    "BatteriesCheckerEasy-6-VLA-v0": VLAWrapperConfig(BATTERIES_CHECKER_OVERLAYS),
    "BatteriesCheckerHard-3-VLA-v0": VLAWrapperConfig(BATTERIES_CHECKER_OVERLAYS),
    "BatteriesCheckerHard-6-VLA-v0": VLAWrapperConfig(BATTERIES_CHECKER_OVERLAYS),
    # The pd_joint_pos hold-action wrappers in some motion-planning oracle
    # scripts are upstream of replay. Validation follows the unfiltered pd-ee
    # rollout that exports the published train data.
    "BlinkCountButtonPressEasy-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "BlinkCountButtonPressMedium-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "BlinkCountButtonPressHard-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "BlinkCountButtonPressEasy-Long-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "BlinkCountButtonPressMedium-Long-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "BlinkCountButtonPressHard-Long-VLA-v0": VLAWrapperConfig(BLINK_COUNT_OVERLAYS),
    "TraceShapeEasy-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TraceShapeMedium-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TraceShapeHard-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TraceShapeSeqEasy-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TraceShapeSeqMedium-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TraceShapeSeqHard-VLA-v0": VLAWrapperConfig(TRACE_SHAPE_OVERLAYS),
    "TimedTransferEasy-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "TimedTransferMedium-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "TimedTransferHard-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "TimedTransferEasy-Long-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "TimedTransferMedium-Long-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "TimedTransferHard-Long-VLA-v0": VLAWrapperConfig(TIMED_TRANSFER_OVERLAYS),
    "GatherAndRecall1-VLA-v0": VLAWrapperConfig(STEP_REWARD_OVERLAYS),
    "GatherAndRecall3-VLA-v0": VLAWrapperConfig(STEP_REWARD_OVERLAYS),
    "GatherAndRecall5-VLA-v0": VLAWrapperConfig(STEP_REWARD_OVERLAYS),
    "GatherAndRecall7-VLA-v0": VLAWrapperConfig(STEP_REWARD_OVERLAYS),
    "GatherAndRecall9-VLA-v0": VLAWrapperConfig(STEP_REWARD_OVERLAYS),
}

MIKASA_VLA_ENV_IDS = tuple(VLA_WRAPPER_CONFIGS)


def _resolve_env_id(env: gym.Env) -> str:
    spec = getattr(env, "spec", None)
    if spec is None:
        spec = getattr(env.unwrapped, "spec", None)
    if spec is None or spec.id is None:
        raise ValueError("Cannot resolve env_id. Pass an env created via `gym.make(...)`.")
    return spec.id


def _warn_if_control_mode_differs(env: gym.Env, env_id: str, config: VLAWrapperConfig) -> None:
    expected = config.expected_control_mode
    actual = getattr(env.unwrapped, "control_mode", None)
    if expected is None or actual is None or actual == expected:
        return

    warnings.warn(
        f"{env_id} was collected with control_mode={expected!r}, but this env "
        f"was created with control_mode={actual!r}. The configured curriculum "
        "wrapper may not match the action space.",
        RuntimeWarning,
        stacklevel=3,
    )


def apply_mikasa_vla_wrappers(env: gym.Env, *, include_overlays: bool = True) -> gym.Env:
    """Apply the configured MIKASA-Robo-VLA wrapper stack.

    Call this immediately after ``gym.make``. Add support for another env by
    adding its :class:`VLAWrapperConfig` to :data:`VLA_WRAPPER_CONFIGS`.

    The returned env exposes ``env.max_episode_steps`` with the configured
    ManiSkill horizon, including a ``gym.make(..., max_episode_steps=...)``
    override.
    """
    env_id = _resolve_env_id(env)
    max_episode_steps = gym_utils.find_max_episode_steps_value(env)

    try:
        config = VLA_WRAPPER_CONFIGS[env_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown env_id={env_id!r}. Add it to VLA_WRAPPER_CONFIGS or use one of: {sorted(VLA_WRAPPER_CONFIGS)}"
        ) from exc

    _warn_if_control_mode_differs(env, env_id, config)

    env = StateOnlyTensorToDictWrapper(env)
    if config.curriculum_wrapper is not None:
        env = config.curriculum_wrapper(env)

    if include_overlays:
        for wrapper_cls in config.overlays:
            env = wrapper_cls(env)

    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=False, oracle=False, joints=True)
    env = ConvertJointsToEEFXyzRpyGripperWrapper(env)
    env.max_episode_steps = max_episode_steps
    return env
