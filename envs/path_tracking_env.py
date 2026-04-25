"""MuJoCo path-tracking environment for PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class EnvConfig:
    """Configuration values for the path-tracking environment."""

    max_steps: int = 1000
    n_waypoints: int = 50
    waypoint_threshold: float = 0.2
    arena_limit: float = 3.0
    waypoint_spacing: float = 0.3
    substeps: int = 5


class PathTrackingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Differential-drive robot that tracks a smooth 2D waypoint path."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        config: Optional[EnvConfig] = None,
    ) -> None:
        """Initialize MuJoCo model, spaces, and episode state."""
        try:
            if render_mode not in {None, "human", "rgb_array"}:
                raise ValueError(f"Unsupported render_mode: {render_mode}")

            self.config = config or EnvConfig()
            self.render_mode = render_mode
            self._base_seed = seed
            self._episode_index = 0
            self._rng = np.random.default_rng(seed)

            model_path = (
                Path(__file__).resolve().parents[1]
                / "assets"
                / "robot.xml"
            )
            self.model = mujoco.MjModel.from_xml_path(str(model_path))
            self.data = mujoco.MjData(self.model)

            self.path = np.zeros(
                (self.config.n_waypoints, 2),
                dtype=np.float32,
            )
            self.current_waypoint_idx = 1
            self.step_count = 0
            self.prev_action = np.zeros(2, dtype=np.float32)
            self.prev_distance = 0.0

            self._renderer: Optional[mujoco.Renderer] = None
            self._viewer: Optional[Any] = None

            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            )

            obs_high = np.array(
                [
                    10.0,
                    10.0,
                    1.0,
                    1.0,
                    5.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    10.0,
                    np.pi,
                ],
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-obs_high,
                high=obs_high,
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize PathTrackingEnv."
            ) from exc

    def _resample_path(self, raw_points: np.ndarray) -> np.ndarray:
        """Resample dense points to fixed arc-length spacing."""
        try:
            deltas = np.diff(raw_points, axis=0)
            segment_lengths = np.linalg.norm(deltas, axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))

            targets = (
                np.arange(self.config.n_waypoints, dtype=np.float64)
                * self.config.waypoint_spacing
            )

            if cumulative[-1] < targets[-1]:
                scale = (targets[-1] + 1e-6) / max(cumulative[-1], 1e-6)
                raw_points = raw_points * scale
                deltas = np.diff(raw_points, axis=0)
                segment_lengths = np.linalg.norm(deltas, axis=1)
                cumulative = np.concatenate(
                    ([0.0], np.cumsum(segment_lengths))
                )

            x_interp = np.interp(targets, cumulative, raw_points[:, 0])
            y_interp = np.interp(targets, cumulative, raw_points[:, 1])
            return np.column_stack((x_interp, y_interp)).astype(np.float32)
        except Exception as exc:
            raise RuntimeError("Failed to resample waypoint path.") from exc

    def _generate_waypoints(self) -> np.ndarray:
        """Generate a randomized figure-8 path with 50 points."""
        try:
            t = np.linspace(0.0, 2.0 * np.pi, 1600, dtype=np.float64)
            amp_x = 2.4 + self._rng.uniform(-0.15, 0.15)
            amp_y = 1.2 + self._rng.uniform(-0.10, 0.10)
            phase = self._rng.uniform(-0.25, 0.25)
            freq = 1.0 + self._rng.uniform(-0.08, 0.08)

            theta = freq * t + phase
            x = amp_x * np.sin(theta)
            y = amp_y * np.sin(theta) * np.cos(theta)
            raw = np.column_stack((x, y)).astype(np.float32)
            return self._resample_path(raw)
        except Exception as exc:
            raise RuntimeError("Failed to generate waypoints.") from exc

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        try:
            return (angle + np.pi) % (2.0 * np.pi) - np.pi
        except Exception as exc:
            raise RuntimeError("Failed to wrap angle.") from exc

    def _robot_pose(self) -> Tuple[float, float, float]:
        """Return robot x, y, and yaw from MuJoCo state."""
        try:
            x = float(self.data.qpos[0])
            y = float(self.data.qpos[1])
            yaw = float(self.data.qpos[2])
            return x, y, yaw
        except Exception as exc:
            raise RuntimeError("Failed to read robot pose.") from exc

    def _set_robot_pose(self, x: float, y: float, yaw: float) -> None:
        """Set planar robot pose and zero its velocities."""
        try:
            self.data.qpos[:] = 0.0
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            self.data.qpos[0] = x
            self.data.qpos[1] = y
            self.data.qpos[2] = yaw
            mujoco.mj_forward(self.model, self.data)
        except Exception as exc:
            raise RuntimeError("Failed to set robot pose.") from exc

    def _compute_cross_track_error(self, x: float, y: float) -> float:
        """Compute minimum distance to the reference waypoint set."""
        try:
            position = np.array([x, y], dtype=np.float32)
            distances = np.linalg.norm(self.path - position, axis=1)
            return float(np.min(distances))
        except Exception as exc:
            raise RuntimeError("Failed to compute cross-track error.") from exc

    def _compute_heading_error(self, x: float, y: float, yaw: float) -> float:
        """Compute absolute heading error to the next target waypoint."""
        try:
            target = self.path[self.current_waypoint_idx]
            dx = float(target[0] - x)
            dy = float(target[1] - y)
            desired = float(np.arctan2(dy, dx))
            return abs(self._wrap_angle(desired - yaw))
        except Exception as exc:
            raise RuntimeError("Failed to compute heading error.") from exc

    def _next_waypoints_local(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> np.ndarray:
        """Return next 3 waypoints in robot-local frame as 6 values."""
        try:
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rel_points = []
            for offset in range(3):
                idx = min(
                    self.current_waypoint_idx + offset,
                    self.config.n_waypoints - 1,
                )
                dx = float(self.path[idx, 0] - x)
                dy = float(self.path[idx, 1] - y)
                local_x = cos_yaw * dx + sin_yaw * dy
                local_y = -sin_yaw * dx + cos_yaw * dy
                rel_points.extend([local_x, local_y])
            return np.array(rel_points, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError("Failed to compute local waypoints.") from exc

    def _compute_tracking_metrics(self) -> Tuple[float, float, float]:
        """Compute cross-track, heading, and target-distance metrics."""
        try:
            x, y, yaw = self._robot_pose()
            cross_track = self._compute_cross_track_error(x, y)
            heading_error = self._compute_heading_error(x, y, yaw)
            target = self.path[self.current_waypoint_idx]
            target_distance = float(np.linalg.norm(target - np.array([x, y])))
            return cross_track, heading_error, target_distance
        except Exception as exc:
            raise RuntimeError("Failed to compute tracking metrics.") from exc

    def _compose_reward(
        self,
        progress_reward: float,
        cross_track_error: float,
        heading_error: float,
        action_smoothness: float,
        waypoint_reached: float,
        episode_complete: float,
        out_of_bounds: float,
    ) -> float:
        """Compose scalar reward from weighted component terms."""
        try:
            reward = (
                2.0 * progress_reward
                - 1.5 * cross_track_error
                - 0.5 * heading_error
                - 0.1 * action_smoothness
                + 5.0 * waypoint_reached
                + 50.0 * episode_complete
                - 10.0 * out_of_bounds
            )
            return float(reward)
        except Exception as exc:
            raise RuntimeError("Failed to compose reward.") from exc

    def _build_info(
        self,
        reward: float,
        components: Dict[str, float],
        cross_track_error: float,
        heading_error: float,
    ) -> Dict[str, float]:
        """Build info dictionary with state and reward diagnostics."""
        try:
            x, y, _ = self._robot_pose()
            completion = (
                float(self.current_waypoint_idx)
                / float(self.config.n_waypoints - 1)
            )
            info = {
                "reward": float(reward),
                "waypoint_index": float(self.current_waypoint_idx),
                "path_completion": completion,
                "cross_track_error": float(cross_track_error),
                "heading_error": float(heading_error),
                "position_x": float(x),
                "position_y": float(y),
                "progress_reward": float(components["progress_reward"]),
                "action_smoothness": float(components["action_smoothness"]),
                "waypoint_reached": float(components["waypoint_reached"]),
                "episode_complete": float(components["episode_complete"]),
                "out_of_bounds": float(components["out_of_bounds"]),
                "reward_progress_term": float(
                    2.0 * components["progress_reward"]
                ),
                "reward_cross_track_term": float(
                    -1.5 * cross_track_error
                ),
                "reward_heading_term": float(-0.5 * heading_error),
                "reward_smoothness_term": float(
                    -0.1 * components["action_smoothness"]
                ),
                "reward_waypoint_term": float(
                    5.0 * components["waypoint_reached"]
                ),
                "reward_completion_term": float(
                    50.0 * components["episode_complete"]
                ),
                "reward_bounds_term": float(
                    -10.0 * components["out_of_bounds"]
                ),
            }
            return info
        except Exception as exc:
            raise RuntimeError("Failed to build info dict.") from exc

    def _get_observation(
        self,
        cross_track_error: Optional[float] = None,
        heading_error: Optional[float] = None,
    ) -> np.ndarray:
        """Assemble the 14D observation vector."""
        try:
            x, y, yaw = self._robot_pose()
            if cross_track_error is None or heading_error is None:
                cross_track_error, heading_error, _ = (
                    self._compute_tracking_metrics()
                )

            vx_world = float(self.data.qvel[0])
            vy_world = float(self.data.qvel[1])
            angular_velocity = float(self.data.qvel[2])
            linear_velocity = (
                np.cos(yaw) * vx_world + np.sin(yaw) * vy_world
            )

            waypoints_local = self._next_waypoints_local(x, y, yaw)

            observation = np.array(
                [
                    x,
                    y,
                    np.sin(yaw),
                    np.cos(yaw),
                    linear_velocity,
                    angular_velocity,
                    waypoints_local[0],
                    waypoints_local[1],
                    waypoints_local[2],
                    waypoints_local[3],
                    waypoints_local[4],
                    waypoints_local[5],
                    cross_track_error,
                    heading_error,
                ],
                dtype=np.float32,
            )
            return np.nan_to_num(observation, nan=0.0).astype(np.float32)
        except Exception as exc:
            raise RuntimeError("Failed to build observation.") from exc

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Reset simulation and randomize path and initial robot pose."""
        try:
            del options
            super().reset(seed=seed)

            if seed is not None:
                self._base_seed = seed

            if self._base_seed is None:
                seed_sequence = np.random.SeedSequence()
                episode_seed = int(seed_sequence.generate_state(1)[0])
            else:
                episode_seed = int(self._base_seed + self._episode_index)

            self._rng = np.random.default_rng(episode_seed)
            self._episode_index += 1

            self.path = self._generate_waypoints()
            self.current_waypoint_idx = min(1, self.config.n_waypoints - 1)
            self.step_count = 0
            self.prev_action[:] = 0.0

            start = self.path[0]
            heading_vec = self.path[1] - self.path[0]
            nominal_yaw = float(np.arctan2(heading_vec[1], heading_vec[0]))
            x = float(start[0] + self._rng.normal(0.0, 0.02))
            y = float(start[1] + self._rng.normal(0.0, 0.02))
            yaw = float(nominal_yaw + self._rng.normal(0.0, 0.05))
            self._set_robot_pose(x, y, yaw)

            cross_track, heading_error, distance_to_target = (
                self._compute_tracking_metrics()
            )
            self.prev_distance = distance_to_target

            components = {
                "progress_reward": 0.0,
                "action_smoothness": 0.0,
                "waypoint_reached": 0.0,
                "episode_complete": 0.0,
                "out_of_bounds": 0.0,
            }
            observation = self._get_observation(cross_track, heading_error)
            info = self._build_info(
                reward=0.0,
                components=components,
                cross_track_error=cross_track,
                heading_error=heading_error,
            )

            if self.render_mode == "human":
                self.render()
            return observation, info
        except Exception as exc:
            raise RuntimeError("Environment reset failed.") from exc

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        """Advance simulation and compute reward/termination signals."""
        try:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if action.shape != (2,):
                raise ValueError("Action must have shape (2,).")

            clipped_action = np.clip(action, -1.0, 1.0)
            self.data.ctrl[0] = float(clipped_action[0])
            self.data.ctrl[1] = float(clipped_action[1])

            for _ in range(self.config.substeps):
                mujoco.mj_step(self.model, self.data)

            self.step_count += 1
            cross_track, heading_error, distance_to_target = (
                self._compute_tracking_metrics()
            )

            progress_reward = float(self.prev_distance - distance_to_target)
            waypoint_reached = 0.0

            if distance_to_target <= self.config.waypoint_threshold:
                waypoint_reached = 1.0
                if self.current_waypoint_idx < self.config.n_waypoints - 1:
                    self.current_waypoint_idx += 1
                    cross_track, heading_error, distance_to_target = (
                        self._compute_tracking_metrics()
                    )

            episode_complete = float(
                self.current_waypoint_idx == self.config.n_waypoints - 1
                and distance_to_target <= self.config.waypoint_threshold
            )
            out_of_bounds = float(cross_track > self.config.arena_limit)
            action_smoothness = float(
                np.linalg.norm(clipped_action - self.prev_action)
            )

            components = {
                "progress_reward": progress_reward,
                "action_smoothness": action_smoothness,
                "waypoint_reached": waypoint_reached,
                "episode_complete": episode_complete,
                "out_of_bounds": out_of_bounds,
            }

            reward = self._compose_reward(
                progress_reward=progress_reward,
                cross_track_error=cross_track,
                heading_error=heading_error,
                action_smoothness=action_smoothness,
                waypoint_reached=waypoint_reached,
                episode_complete=episode_complete,
                out_of_bounds=out_of_bounds,
            )

            terminated = bool(out_of_bounds > 0.0 or episode_complete > 0.0)
            truncated = bool(self.step_count >= self.config.max_steps)

            observation = self._get_observation(cross_track, heading_error)
            info = self._build_info(
                reward=reward,
                components=components,
                cross_track_error=cross_track,
                heading_error=heading_error,
            )

            self.prev_action = clipped_action.astype(np.float32)
            self.prev_distance = distance_to_target

            if self.render_mode == "human":
                self.render()

            return observation, float(reward), terminated, truncated, info
        except Exception as exc:
            raise RuntimeError("Environment step failed.") from exc

    def render(self) -> Optional[np.ndarray]:
        """Render in human mode or return RGB array frame."""
        try:
            if self.render_mode == "rgb_array":
                if self._renderer is None:
                    self._renderer = mujoco.Renderer(
                        self.model,
                        width=640,
                        height=480,
                    )
                self._renderer.update_scene(self.data, camera="track_cam")
                return self._renderer.render().copy()

            if self.render_mode == "human":
                if self._viewer is None:
                    import mujoco.viewer

                    self._viewer = mujoco.viewer.launch_passive(
                        self.model,
                        self.data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                self._viewer.sync()
            return None
        except Exception as exc:
            raise RuntimeError("Environment render failed.") from exc

    def close(self) -> None:
        """Close renderer and viewer handles."""
        try:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None

            if self._viewer is not None and hasattr(self._viewer, "close"):
                self._viewer.close()
                self._viewer = None
        except Exception as exc:
            raise RuntimeError("Environment close failed.") from exc
