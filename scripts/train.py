"""Train a PPO agent for MuJoCo path tracking."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.path_tracking_env import PathTrackingEnv  # noqa: E402


class TensorboardCallback(BaseCallback):
    """Log custom environment metrics to TensorBoard."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize callback state."""
        try:
            super().__init__(verbose=verbose)
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize TensorboardCallback."
            ) from exc

    def _on_step(self) -> bool:
        """Record custom scalar metrics each rollout step."""
        try:
            infos = self.locals.get("infos", [])
            if not infos:
                return True

            cross_track = [
                float(info["cross_track_error"])
                for info in infos
                if "cross_track_error" in info
            ]
            heading_error = [
                float(info["heading_error"])
                for info in infos
                if "heading_error" in info
            ]
            completion = [
                float(info["path_completion"])
                for info in infos
                if "path_completion" in info
            ]

            if cross_track:
                self.logger.record(
                    "custom/mean_cross_track_error",
                    float(np.mean(cross_track)),
                )
            if heading_error:
                self.logger.record(
                    "custom/mean_heading_error",
                    float(np.mean(heading_error)),
                )
            if completion:
                self.logger.record(
                    "custom/mean_path_completion",
                    float(np.mean(completion)),
                )
            return True
        except Exception as exc:
            raise RuntimeError("Tensorboard callback logging failed.") from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""
    try:
        parser = argparse.ArgumentParser(description="Train PPO path tracker")
        parser.add_argument(
            "--timesteps",
            type=int,
            default=500_000,
            help="Total training timesteps",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Global random seed",
        )
        parser.add_argument(
            "--run-name",
            type=str,
            default="ppo_path_tracking",
            help="TensorBoard run name",
        )
        return parser.parse_args()
    except Exception as exc:
        raise RuntimeError("Failed to parse training arguments.") from exc


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, dict):
            raise ValueError("Config must deserialize to a dictionary.")
        return config
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load config from {config_path}."
        ) from exc


def set_global_seeds(seed: int) -> None:
    """Set seeds for reproducible experiments."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        set_random_seed(seed)
    except Exception as exc:
        raise RuntimeError("Failed to set global seeds.") from exc


def make_env(rank: int, base_seed: int) -> Callable[[], Monitor]:
    """Create a monitored environment factory for vectorized training."""

    def _init() -> Monitor:
        try:
            env_seed = base_seed + rank
            env = PathTrackingEnv(render_mode=None, seed=env_seed)
            env.reset(seed=env_seed)
            return Monitor(env)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize env rank {rank}."
            ) from exc

    return _init


def train(args: argparse.Namespace) -> None:
    """Run PPO training with evaluation and normalization wrappers."""
    train_env = None
    eval_env = None
    try:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "config.yaml")
        set_global_seeds(args.seed)

        models_dir = project_root / "models"
        logs_dir = project_root / "logs"
        models_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / args.run_name).mkdir(parents=True, exist_ok=True)

        n_envs = int(config["training"]["n_envs"])
        train_env = SubprocVecEnv(
            [make_env(rank=i, base_seed=args.seed) for i in range(n_envs)],
            start_method="spawn",
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

        eval_env = SubprocVecEnv(
            [make_env(rank=10_000, base_seed=args.seed)],
            start_method="spawn",
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        eval_env.training = False

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(models_dir / "best_model"),
            log_path=str(logs_dir / args.run_name / "eval"),
            eval_freq=int(config["training"]["eval_freq"]),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        )
        callback = CallbackList([TensorboardCallback(), eval_callback])

        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.005,
            clip_range=0.2,
            gae_lambda=0.95,
            tensorboard_log=str(logs_dir),
            seed=args.seed,
            verbose=1,
        )

        start_time = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=args.run_name,
            progress_bar=True,
        )
        elapsed = time.time() - start_time

        final_model_path = models_dir / "ppo_path_tracking_final"
        vec_norm_path = models_dir / "vec_normalize.pkl"
        model.save(str(final_model_path))
        train_env.save(str(vec_norm_path))

        print("\nTraining summary")
        print("----------------")
        print(f"Run name: {args.run_name}")
        print(f"Total timesteps: {args.timesteps}")
        print(f"Seed: {args.seed}")
        print(f"Elapsed seconds: {elapsed:.2f}")
        print(f"Final model: {final_model_path}")
        print(f"VecNormalize stats: {vec_norm_path}")
        print(f"Best model directory: {models_dir / 'best_model'}")
    except Exception as exc:
        raise RuntimeError("Training pipeline failed.") from exc
    finally:
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()


def main() -> None:
    """Entry point for training script."""
    try:
        args = parse_args()
        train(args)
    except Exception as exc:
        raise RuntimeError("train.py failed.") from exc


if __name__ == "__main__":
    main()
