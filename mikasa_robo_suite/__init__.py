"""Top-level package for MIKASA-Robo."""

from importlib import import_module

__all__ = ["rl", "vla", "memory_envs"]


def __getattr__(name: str):
    # Backward compatibility for legacy imports:
    #   from mikasa_robo_suite import memory_envs
    # The canonical module now lives under vla/ and rl/.
    if name == "memory_envs":
        return import_module("mikasa_robo_suite.vla.memory_envs")
    raise AttributeError(f"module 'mikasa_robo_suite' has no attribute {name!r}")
