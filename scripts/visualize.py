"""Create publication-style plots for path-tracking RL results."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tensorboard.backend.event_processing import event_accumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.path_tracking_env import PathTrackingEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for plotting."""
    try:
        parser = argparse.ArgumentParser(description="Visualize PPO results")
        parser.add_argument(
            "--episodes",
            type=int,
            default=3,
            help="Number of rollout episodes to plot",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default="models/ppo_path_tracking_final.zip",
            help="Model path relative to project root",
        )
        parser.add_argument(
            "--vecnorm-path",
            type=str,
            default="models/vec_normalize.pkl",
            help="VecNormalize path relative to project root",
        )
        parser.add_argument(
            "--log-dir",
            type=str,
            default="logs",
            help="TensorBoard logs directory",
        )
        return parser.parse_args()
    except Exception as exc:
        raise RuntimeError("Failed to parse visualization args.") from exc


def set_global_seeds(seed: int) -> None:
    """Set random seeds for deterministic rollouts."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
    except Exception as exc:
        raise RuntimeError("Failed to set visualization seeds.") from exc


def unwrap_env(env: Any) -> PathTrackingEnv:
    """Unwrap wrappers and return the base PathTrackingEnv instance."""
    try:
        current = env
        for _ in range(16):
            if isinstance(current, PathTrackingEnv):
                return current
            if hasattr(current, "env"):
                current = current.env
                continue
            break
        raise TypeError("Could not unwrap PathTrackingEnv.")
    except Exception as exc:
        raise RuntimeError("Failed to unwrap env.") from exc


def load_policy_and_env(
    project_root: Path,
    model_path: Path,
    vecnorm_path: Path,
    seed: int,
) -> Tuple[PPO, VecNormalize]:
    """Load trained PPO model and VecNormalize wrapper."""
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vecnorm_path.exists():
            raise FileNotFoundError(f"VecNormalize not found: {vecnorm_path}")

        base_env = DummyVecEnv(
            [
                lambda: Monitor(
                    PathTrackingEnv(render_mode=None, seed=seed)
                )
            ]
        )
        vec_env = VecNormalize.load(str(vecnorm_path), base_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(str(model_path), env=vec_env)
        return model, vec_env
    except Exception as exc:
        raise RuntimeError("Failed to load policy and env.") from exc


def collect_rollout_data(
    model: PPO,
    vec_env: VecNormalize,
    episodes: int,
) -> List[Dict[str, np.ndarray]]:
    """Collect rollout traces for paths, errors, and reward terms."""
    try:
        data: List[Dict[str, np.ndarray]] = []
        for _ in range(episodes):
            obs = vec_env.reset()
            raw_env = unwrap_env(vec_env.venv.envs[0])
            reference_path = raw_env.path.copy()

            done = False
            actual_path: List[List[float]] = []
            cross_track_error: List[float] = []
            reward_progress: List[float] = []
            reward_cross_track: List[float] = []
            reward_heading: List[float] = []
            reward_smoothness: List[float] = []
            reward_waypoint: List[float] = []
            reward_completion: List[float] = []
            reward_bounds: List[float] = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec_env.step(action)
                info = infos[0]

                actual_path.append(
                    [
                        float(info.get("position_x", np.nan)),
                        float(info.get("position_y", np.nan)),
                    ]
                )
                cross_track_error.append(
                    float(info.get("cross_track_error", np.nan))
                )
                reward_progress.append(
                    float(info.get("reward_progress_term", 0.0))
                )
                reward_cross_track.append(
                    float(abs(info.get("reward_cross_track_term", 0.0)))
                )
                reward_heading.append(
                    float(abs(info.get("reward_heading_term", 0.0)))
                )
                reward_smoothness.append(
                    float(abs(info.get("reward_smoothness_term", 0.0)))
                )
                reward_waypoint.append(
                    float(info.get("reward_waypoint_term", 0.0))
                )
                reward_completion.append(
                    float(info.get("reward_completion_term", 0.0))
                )
                reward_bounds.append(
                    float(abs(info.get("reward_bounds_term", 0.0)))
                )
                done = bool(dones[0])

            data.append(
                {
                    "reference_path": reference_path,
                    "actual_path": np.array(actual_path, dtype=np.float32),
                    "cross_track_error": np.array(
                        cross_track_error,
                        dtype=np.float32,
                    ),
                    "reward_progress": np.array(reward_progress),
                    "reward_cross_track": np.array(reward_cross_track),
                    "reward_heading": np.array(reward_heading),
                    "reward_smoothness": np.array(reward_smoothness),
                    "reward_waypoint": np.array(reward_waypoint),
                    "reward_completion": np.array(reward_completion),
                    "reward_bounds": np.array(reward_bounds),
                }
            )
        return data
    except Exception as exc:
        raise RuntimeError("Failed to collect rollout data.") from exc


def average_series(series_list: List[np.ndarray]) -> np.ndarray:
    """Pad series with NaN and compute timestep-wise mean."""
    try:
        max_len = max(len(series) for series in series_list)
        stacked = np.full(
            (len(series_list), max_len),
            np.nan,
            dtype=np.float64,
        )
        for idx, series in enumerate(series_list):
            stacked[idx, : len(series)] = series
        return np.nanmean(stacked, axis=0)
    except Exception as exc:
        raise RuntimeError(
            "Failed to average variable-length series."
        ) from exc


def load_training_reward_curve(log_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load reward curve from TensorBoard event files."""
    try:
        event_files = sorted(
            log_dir.rglob("events.out.tfevents.*"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not event_files:
            raise FileNotFoundError("No TensorBoard event files found.")

        candidate_tags = ["rollout/ep_rew_mean", "eval/mean_reward"]
        for event_file in event_files:
            accumulator = event_accumulator.EventAccumulator(str(event_file))
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
            for tag in candidate_tags:
                if tag in scalar_tags:
                    scalars = accumulator.Scalars(tag)
                    if scalars:
                        steps = np.array([item.step for item in scalars])
                        values = np.array([item.value for item in scalars])
                        return steps, values
        raise RuntimeError("No reward scalar tag found in TensorBoard logs.")
    except Exception as exc:
        raise RuntimeError("Failed to load reward curve.") from exc


def plot_paths(data: List[Dict[str, np.ndarray]], output_path: Path) -> None:
    """Plot actual and reference x-y trajectories for three episodes."""
    try:
        fig, axis = plt.subplots(figsize=(8, 6))
        for idx, episode in enumerate(data, start=1):
            reference = episode["reference_path"]
            actual = episode["actual_path"]
            axis.plot(
                reference[:, 0],
                reference[:, 1],
                "--",
                linewidth=1.2,
                alpha=0.6,
                label=f"Reference ep{idx}",
            )
            axis.plot(
                actual[:, 0],
                actual[:, 1],
                linewidth=1.8,
                label=f"Actual ep{idx}",
            )

        axis.set_title("Path Tracking: Actual vs Reference")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.axis("equal")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError("Failed to plot path overlay.") from exc


def plot_cross_track(
    data: List[Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot cross-track error over timestep for each episode."""
    try:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        for idx, episode in enumerate(data, start=1):
            axis.plot(
                episode["cross_track_error"],
                linewidth=1.6,
                label=f"Episode {idx}",
            )
        axis.set_title("Cross-Track Error Over Time")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Cross-track error [m]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError("Failed to plot cross-track error.") from exc


def plot_training_reward(
    steps: np.ndarray,
    rewards: np.ndarray,
    output_path: Path,
) -> None:
    """Plot training reward curve from TensorBoard scalars."""
    try:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.plot(steps, rewards, color="#1f77b4", linewidth=2.0)
        axis.set_title("Training Reward Curve")
        axis.set_xlabel("Environment steps")
        axis.set_ylabel("Mean episodic reward")
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError("Failed to plot training reward.") from exc


def plot_reward_breakdown(
    data: List[Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    """Plot stacked reward-component magnitude over time."""
    try:
        mean_progress = average_series([ep["reward_progress"] for ep in data])
        mean_cross_track = average_series(
            [ep["reward_cross_track"] for ep in data]
        )
        mean_heading = average_series([ep["reward_heading"] for ep in data])
        mean_smoothness = average_series(
            [ep["reward_smoothness"] for ep in data]
        )
        mean_waypoint = average_series([ep["reward_waypoint"] for ep in data])
        mean_completion = average_series(
            [ep["reward_completion"] for ep in data]
        )
        mean_bounds = average_series([ep["reward_bounds"] for ep in data])

        x_axis = np.arange(len(mean_progress))
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.stackplot(
            x_axis,
            np.maximum(mean_progress, 0.0),
            mean_cross_track,
            mean_heading,
            mean_smoothness,
            mean_waypoint,
            mean_completion,
            mean_bounds,
            labels=[
                "Progress term",
                "Cross-track penalty",
                "Heading penalty",
                "Smoothness penalty",
                "Waypoint bonus",
                "Completion bonus",
                "Bounds penalty",
            ],
            alpha=0.9,
        )
        axis.set_title("Reward Components Breakdown")
        axis.set_xlabel("Timestep")
        axis.set_ylabel("Absolute contribution")
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    except Exception as exc:
        raise RuntimeError("Failed to plot reward breakdown.") from exc


def run_visualization(args: argparse.Namespace) -> None:
    """Generate all required result visualizations."""
    vec_env = None
    try:
        set_global_seeds(args.seed)
        plt.style.use("seaborn-v0_8")

        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / args.model_path
        vecnorm_path = project_root / args.vecnorm_path
        log_dir = project_root / args.log_dir
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        model, vec_env = load_policy_and_env(
            project_root=project_root,
            model_path=model_path,
            vecnorm_path=vecnorm_path,
            seed=args.seed,
        )
        rollout_data = collect_rollout_data(
            model=model,
            vec_env=vec_env,
            episodes=args.episodes,
        )
        reward_steps, reward_values = load_training_reward_curve(log_dir)

        path_plot = results_dir / "path_overlay.png"
        cte_plot = results_dir / "cross_track_error.png"
        reward_plot = results_dir / "training_reward_curve.png"
        breakdown_plot = results_dir / "reward_components_breakdown.png"

        plot_paths(rollout_data, path_plot)
        plot_cross_track(rollout_data, cte_plot)
        plot_training_reward(reward_steps, reward_values, reward_plot)
        plot_reward_breakdown(rollout_data, breakdown_plot)

        print("Saved visualization files:")
        print(f"- {path_plot}")
        print(f"- {cte_plot}")
        print(f"- {reward_plot}")
        print(f"- {breakdown_plot}")
    except Exception as exc:
        raise RuntimeError("Visualization pipeline failed.") from exc
    finally:
        if vec_env is not None:
            vec_env.close()


def main() -> None:
    """Entry point for visualization script."""
    try:
        args = parse_args()
        run_visualization(args)
    except Exception as exc:
        raise RuntimeError("visualize.py failed.") from exc


if __name__ == "__main__":
    main()
