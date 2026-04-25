"""Custom Gymnasium environments for path tracking RL."""

from __future__ import annotations

from gymnasium.envs.registration import register

from envs.path_tracking_env import PathTrackingEnv


def register_env() -> None:
    """Register the path-tracking environment with Gymnasium."""
    try:
        register(
            id="PathTracking-v0",
            entry_point="envs.path_tracking_env:PathTrackingEnv",
        )
    except Exception as exc:
        if "Cannot re-register id" in str(exc):
            return
        raise RuntimeError("Failed to register PathTracking-v0 env.") from exc


register_env()

__all__ = ["PathTrackingEnv", "register_env"]
