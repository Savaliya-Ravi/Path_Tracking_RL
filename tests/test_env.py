"""Pytest suite for the custom path-tracking environment."""

from __future__ import annotations

import numpy as np

from envs.path_tracking_env import PathTrackingEnv


def test_env_reset() -> None:
    """Environment reset should return valid finite 14D observation."""
    env = PathTrackingEnv()
    try:
        observation, _ = env.reset(seed=123)
        assert observation.shape == (14,)
        assert np.all(np.isfinite(observation))
    finally:
        env.close()


def test_env_step() -> None:
    """Random action should produce valid observation and float reward."""
    env = PathTrackingEnv()
    try:
        env.reset(seed=123)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        assert observation.shape == (14,)
        assert np.all(np.isfinite(observation))
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_reward_components() -> None:
    """Reward must increase for positive progress with same penalties."""
    env = PathTrackingEnv()
    try:
        toward_reward = env._compose_reward(
            progress_reward=0.20,
            cross_track_error=0.10,
            heading_error=0.10,
            action_smoothness=0.05,
            waypoint_reached=0.0,
            episode_complete=0.0,
            out_of_bounds=0.0,
        )
        away_reward = env._compose_reward(
            progress_reward=-0.20,
            cross_track_error=0.10,
            heading_error=0.10,
            action_smoothness=0.05,
            waypoint_reached=0.0,
            episode_complete=0.0,
            out_of_bounds=0.0,
        )
        assert toward_reward > away_reward
    finally:
        env.close()


def test_observation_bounds() -> None:
    """Observed states should stay inside declared Box space bounds."""
    env = PathTrackingEnv()
    try:
        observation, _ = env.reset(seed=456)
        assert env.observation_space.contains(observation.astype(np.float32))

        for _ in range(20):
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(
                observation.astype(np.float32)
            )
            if terminated or truncated:
                observation, _ = env.reset()
                assert env.observation_space.contains(
                    observation.astype(np.float32)
                )
    finally:
        env.close()
