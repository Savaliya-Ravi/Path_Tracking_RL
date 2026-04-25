"""Evaluate a trained PPO path-tracking agent."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.path_tracking_env import PathTrackingEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""
    try:
        parser = argparse.ArgumentParser(
            description="Evaluate PPO path tracker"
        )
        parser.add_argument(
            "--episodes",
            type=int,
            default=10,
            help="Number of evaluation episodes",
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
        return parser.parse_args()
    except Exception as exc:
        raise RuntimeError("Failed to parse evaluation args.") from exc


def set_global_seeds(seed: int) -> None:
    """Set seeds for deterministic evaluation rollout."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
    except Exception as exc:
        raise RuntimeError("Failed to set evaluation seeds.") from exc


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate trained model and return per-episode metrics."""
    vec_env = None
    try:
        project_root = Path(__file__).resolve().parents[1]
        model_path = project_root / args.model_path
        vecnorm_path = project_root / args.vecnorm_path
        results_dir = project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not vecnorm_path.exists():
            raise FileNotFoundError(
                f"VecNormalize stats not found: {vecnorm_path}"
            )

        set_global_seeds(args.seed)

        base_env = DummyVecEnv(
            [
                lambda: Monitor(
                    PathTrackingEnv(render_mode=None, seed=args.seed)
                )
            ]
        )
        vec_env = VecNormalize.load(str(vecnorm_path), base_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(str(model_path), env=vec_env)

        rows: List[Dict[str, float]] = []
        for episode in range(1, args.episodes + 1):
            obs = vec_env.reset()
            done = False
            total_reward = 0.0
            cross_track_errors: List[float] = []
            heading_errors: List[float] = []
            completion = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = vec_env.step(action)
                total_reward += float(rewards[0])

                info = infos[0]
                if "cross_track_error" in info:
                    cross_track_errors.append(float(info["cross_track_error"]))
                if "heading_error" in info:
                    heading_errors.append(float(info["heading_error"]))
                if "path_completion" in info:
                    completion = max(
                        completion,
                        float(info["path_completion"]),
                    )

                done = bool(dones[0])

            rows.append(
                {
                    "episode": float(episode),
                    "total_reward": total_reward,
                    "path_completion_pct": 100.0 * completion,
                    "mean_cross_track_error": float(
                        np.mean(cross_track_errors)
                    ),
                    "mean_heading_error": float(np.mean(heading_errors)),
                }
            )

        metrics_df = pd.DataFrame(rows)
        output_path = results_dir / "eval_metrics.csv"
        metrics_df.to_csv(output_path, index=False)

        print("\nEvaluation metrics")
        print("------------------")
        print(metrics_df.to_string(index=False, float_format="{:.4f}".format))

        summary = metrics_df[
            [
                "total_reward",
                "path_completion_pct",
                "mean_cross_track_error",
                "mean_heading_error",
            ]
        ].mean()
        print("\nAggregate means")
        print("---------------")
        print(summary.to_string(float_format="{:.4f}".format))
        print(f"\nMetrics CSV saved to: {output_path}")

        mean_cte = float(metrics_df["mean_cross_track_error"].mean())
        if mean_cte < 0.3:
            print("Cross-track target met: mean error < 0.3 m")
        else:
            print("Cross-track target not met: mean error >= 0.3 m")
        return metrics_df
    except Exception as exc:
        raise RuntimeError("Evaluation pipeline failed.") from exc
    finally:
        if vec_env is not None:
            vec_env.close()


def main() -> None:
    """Entry point for evaluation script."""
    try:
        args = parse_args()
        evaluate(args)
    except Exception as exc:
        raise RuntimeError("evaluate.py failed.") from exc


if __name__ == "__main__":
    main()
