"""Canonical mapping from gym env_id to dataset folder name.

RLDS/LeRobot tooling require dataset directories to be lowercase snake_case,
while gym envs are registered in CamelCase with dashes (e.g. ``BatteriesCheckerHard-3-VLA-v0``).
This module converts between the two so dataset artifacts are always written
to snake_case directories while env registration stays untouched.
"""

from __future__ import annotations

import re

__all__ = [
    "env_id_to_dataset_name",
    "is_dataset_name",
]


_CAMEL_BOUNDARY_1 = re.compile(r"([a-z])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")
# Split letter→digit only when the letter is preceded by another letter, so tokens
# like ``v0`` stay intact while ``Color3`` becomes ``Color_3``.
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])([A-Za-z])([0-9])")


def _token_to_snake(token: str) -> str:
    token = _CAMEL_BOUNDARY_1.sub(r"\1_\2", token)
    token = _CAMEL_BOUNDARY_2.sub(r"\1_\2", token)
    token = _LETTER_DIGIT.sub(r"\1_\2", token)
    return token.lower()


def env_id_to_dataset_name(env_id: str) -> str:
    """Convert a gym env_id to its canonical dataset folder name.

    Examples:
        ``BatteriesCheckerHard-3-VLA-v0`` -> ``batteries_checker_hard_3_vla_v0``
        ``RememberShapeAndColor3x2-Long-VLA-v0`` -> ``remember_shape_and_color_3x2_long_vla_v0``
        ``ShellGameTouch-VLA-v0`` -> ``shell_game_touch_vla_v0``

    If ``env_id`` already looks like a snake_case dataset name it is returned unchanged.
    """
    if is_dataset_name(env_id):
        return env_id
    parts = env_id.split("-")
    return "_".join(_token_to_snake(part) for part in parts if part)


def is_dataset_name(value: str) -> bool:
    """Return True if ``value`` is already a snake_case dataset name."""
    if not value:
        return False
    return all(ch.islower() or ch.isdigit() or ch == "_" for ch in value)
