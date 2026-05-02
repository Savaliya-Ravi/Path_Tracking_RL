import argparse
import sys
from pathlib import Path
import time
import shutil

import numpy as np
import yaml
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.path_tracking_env import PathTrackingEnv, EnvConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="ppo_path_tracking")
    return p.parse_args()


def make_env(rank, seed, config):
    def _init():
        env = PathTrackingEnv(seed=seed + rank, config=config)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    args = parse_args()

    root = PROJECT_ROOT
    config = yaml.safe_load(open(root / "config.yaml"))

    env_cfg = EnvConfig(**config.get("env", {}))
    train_cfg = config.get("training", {})
    ppo_cfg = config.get("ppo", {})

    set_random_seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    models_dir = root / "models"
    logs_dir = root / "logs"
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # -------- ENV -------- #

    n_envs = int(train_cfg.get("n_envs", 4))

    train_env = SubprocVecEnv(
        [make_env(i, args.seed, env_cfg) for i in range(n_envs)],
        start_method="spawn",
    )

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    eval_env = SubprocVecEnv(
        [make_env(999, args.seed, env_cfg)],
        start_method="spawn",
    )

    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.training = False

    # -------- CALLBACK -------- #

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best_model"),
        eval_freq=int(train_cfg.get("eval_freq", 10000)),
        n_eval_episodes=5,
        deterministic=True,
    )

    # -------- MODEL -------- #

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 10)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.005)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        gae_lambda=0.95,
        tensorboard_log=str(logs_dir),
        seed=args.seed,
        verbose=1,
    )

    # -------- TRAIN -------- #

    start = time.time()

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    elapsed = time.time() - start

    # -------- SAVE -------- #

    final_model = models_dir / args.run_name
    vec_path = models_dir / f"{args.run_name}_vec_normalize.pkl"

    model.save(str(final_model))
    train_env.save(str(vec_path))

    # rename best model
    best_src = models_dir / "best_model" / "best_model.zip"
    best_dst = models_dir / "best_model" / f"{args.run_name}.zip"

    if best_src.exists():
        if best_dst.exists():
            best_dst.unlink()
        shutil.move(best_src, best_dst)

    print("\nTraining done")
    print("-------------")
    print(f"Timesteps: {args.timesteps}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Model: {final_model}")
    print(f"VecNorm: {vec_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

# """Train a PPO agent for MuJoCo path tracking."""

# from __future__ import annotations

# import argparse
# import random
# import sys
# import time
# from pathlib import Path
# from typing import Any, Callable, Dict

# import shutil
# import numpy as np
# import torch as th
# import yaml
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import (
#     BaseCallback,
#     CallbackList,
#     EvalCallback,
# )
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.utils import set_random_seed
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from envs.path_tracking_env import EnvConfig, PathTrackingEnv  # noqa: E402


# class TensorboardCallback(BaseCallback):
#     """Log custom environment metrics to TensorBoard."""

#     def __init__(self, verbose: int = 0) -> None:
#         """Initialize callback state."""
#         try:
#             super().__init__(verbose=verbose)
#         except Exception as exc:
#             raise RuntimeError(
#                 "Failed to initialize TensorboardCallback."
#             ) from exc

#     def _on_step(self) -> bool:
#         """Record custom scalar metrics each rollout step."""
#         try:
#             infos = self.locals.get("infos", [])
#             if not infos:
#                 return True

#             cross_track = [
#                 float(info["cross_track_error"])
#                 for info in infos
#                 if "cross_track_error" in info
#             ]
#             heading_error = [
#                 float(info["heading_error"])
#                 for info in infos
#                 if "heading_error" in info
#             ]
#             completion = [
#                 float(info["path_completion"])
#                 for info in infos
#                 if "path_completion" in info
#             ]

#             if cross_track:
#                 self.logger.record(
#                     "custom/mean_cross_track_error",
#                     float(np.mean(cross_track)),
#                 )
#             if heading_error:
#                 self.logger.record(
#                     "custom/mean_heading_error",
#                     float(np.mean(heading_error)),
#                 )
#             if completion:
#                 self.logger.record(
#                     "custom/mean_path_completion",
#                     float(np.mean(completion)),
#                 )
#             return True
#         except Exception as exc:
#             raise RuntimeError("Tensorboard callback logging failed.") from exc


# def parse_args() -> argparse.Namespace:
#     """Parse CLI arguments for training."""
#     try:
#         parser = argparse.ArgumentParser(description="Train PPO path tracker")
#         parser.add_argument(
#             "--timesteps",
#             type=int,
#             default=500_000,
#             help="Total training timesteps",
#         )
#         parser.add_argument(
#             "--seed",
#             type=int,
#             default=42,
#             help="Global random seed",
#         )
#         parser.add_argument(
#             "--run-name",
#             type=str,
#             default="ppo_path_tracking",
#             help="TensorBoard run name",
#         )
#         return parser.parse_args()
#     except Exception as exc:
#         raise RuntimeError("Failed to parse training arguments.") from exc


# def load_config(config_path: Path) -> Dict[str, Any]:
#     """Load YAML configuration file."""
#     try:
#         with config_path.open("r", encoding="utf-8") as handle:
#             config = yaml.safe_load(handle)
#         if not isinstance(config, dict):
#             raise ValueError("Config must deserialize to a dictionary.")
#         return config
#     except Exception as exc:
#         raise RuntimeError(
#             f"Failed to load config from {config_path}."
#         ) from exc


# def build_env_config(config: Dict[str, Any]) -> EnvConfig:
#     """Build EnvConfig from YAML configuration."""
#     try:
#         env_cfg = config.get("env", {})
#         return EnvConfig(
#             max_steps=int(env_cfg.get("max_steps", EnvConfig.max_steps)),
#             n_waypoints=int(env_cfg.get("n_waypoints", EnvConfig.n_waypoints)),
#             waypoint_threshold=float(
#                 env_cfg.get("waypoint_threshold", EnvConfig.waypoint_threshold)
#             ),
#             arena_limit=float(env_cfg.get("arena_limit", EnvConfig.arena_limit)),
#             waypoint_spacing=float(
#                 env_cfg.get("waypoint_spacing", EnvConfig.waypoint_spacing)
#             ),
#             substeps=int(env_cfg.get("substeps", EnvConfig.substeps)),
#             path_type=str(env_cfg.get("path_type", EnvConfig.path_type)),
#         )
#     except Exception as exc:
#         raise RuntimeError("Failed to build EnvConfig.") from exc


# def set_global_seeds(seed: int) -> None:
#     """Set seeds for reproducible experiments."""
#     try:
#         random.seed(seed)
#         np.random.seed(seed)
#         th.manual_seed(seed)
#         set_random_seed(seed)
#     except Exception as exc:
#         raise RuntimeError("Failed to set global seeds.") from exc


# def make_env(
#     rank: int,
#     base_seed: int,
#     env_config: EnvConfig,
# ) -> Callable[[], Monitor]:
#     """Create a monitored environment factory for vectorized training."""

#     def _init() -> Monitor:
#         try:
#             env_seed = base_seed + rank
#             env = PathTrackingEnv(
#                 render_mode=None,
#                 seed=env_seed,
#                 config=env_config,
#             )
#             env.reset(seed=env_seed)
#             return Monitor(env)
#         except Exception as exc:
#             raise RuntimeError(
#                 f"Failed to initialize env rank {rank}."
#             ) from exc

#     return _init


# def train(args: argparse.Namespace) -> None:
#     """Run PPO training with evaluation and normalization wrappers."""
#     train_env = None
#     eval_env = None
#     try:
#         project_root = Path(__file__).resolve().parents[1]
#         config = load_config(project_root / "config.yaml")
#         set_global_seeds(args.seed)
#         env_config = build_env_config(config)

#         models_dir = project_root / "models"
#         logs_dir = project_root / "logs"
#         models_dir.mkdir(parents=True, exist_ok=True)
#         logs_dir.mkdir(parents=True, exist_ok=True)
#         (logs_dir / args.run_name).mkdir(parents=True, exist_ok=True)

#         n_envs = int(config["training"]["n_envs"])
#         train_env = SubprocVecEnv(
#             [
#                 make_env(rank=i, base_seed=args.seed, env_config=env_config)
#                 for i in range(n_envs)
#             ],
#             start_method="spawn",
#         )
#         train_env = VecNormalize(
#             train_env,
#             norm_obs=True,
#             norm_reward=True,
#             clip_obs=10.0,
#         )

#         eval_env = SubprocVecEnv(
#             [make_env(rank=10_000, base_seed=args.seed, env_config=env_config)],
#             start_method="spawn",
#         )
#         eval_env = VecNormalize(
#             eval_env,
#             norm_obs=True,
#             norm_reward=False,
#             clip_obs=10.0,
#         )
#         eval_env.training = False

#         eval_callback = EvalCallback(
#             eval_env,
#             best_model_save_path=str(models_dir / "best_model"),
#             log_path=str(logs_dir / args.run_name / "eval"),
#             eval_freq=int(config["training"]["eval_freq"]),
#             n_eval_episodes=5,
#             deterministic=True,
#             render=False,
#         )
#         callback = CallbackList([TensorboardCallback(), eval_callback])

#         # Read PPO hyperparameters from config with sensible defaults
#         ppo_cfg = config.get("ppo", {})
#         ppo_n_steps = int(ppo_cfg.get("n_steps", 2048))
#         ppo_batch_size = int(ppo_cfg.get("batch_size", 64))
#         ppo_n_epochs = int(ppo_cfg.get("n_epochs", 10))
#         ppo_gamma = float(ppo_cfg.get("gamma", 0.99))
#         ppo_lr = float(ppo_cfg.get("learning_rate", 3e-4))
#         ppo_ent = float(ppo_cfg.get("ent_coef", 0.005))
#         ppo_clip = float(ppo_cfg.get("clip_range", 0.2))

#         model = PPO(
#             policy="MlpPolicy",
#             env=train_env,
#             n_steps=ppo_n_steps,
#             batch_size=ppo_batch_size,
#             n_epochs=ppo_n_epochs,
#             gamma=ppo_gamma,
#             learning_rate=ppo_lr,
#             ent_coef=ppo_ent,
#             clip_range=ppo_clip,
#             gae_lambda=0.95,
#             tensorboard_log=str(logs_dir),
#             seed=args.seed,
#             verbose=1,
#         )

#         start_time = time.time()
#         model.learn(
#             total_timesteps=args.timesteps,
#             callback=callback,
#             tb_log_name=args.run_name,
#             progress_bar=True,
#         )
#         elapsed = time.time() - start_time

#         final_model_path = models_dir / args.run_name
#         vec_norm_path = models_dir / f"{args.run_name}_vec_normalize.pkl"

#         model.save(str(final_model_path))
#         train_env.save(str(vec_norm_path))

#         best_model_dir = models_dir / "best_model"
#         best_model_src = best_model_dir / "best_model.zip"
#         best_model_dst = best_model_dir / f"{args.run_name}.zip"

#         if best_model_src.exists():
#             if best_model_dst.exists():
#                 best_model_dst.unlink()
#             shutil.move(best_model_src, best_model_dst)

#         print("\nTraining summary")
#         print("----------------")
#         print(f"Run name: {args.run_name}")
#         print(f"Total timesteps: {args.timesteps}")
#         print(f"Seed: {args.seed}")
#         print(f"Elapsed seconds: {elapsed:.2f}")
#         print(f"Final model: {final_model_path}")
#         print(f"VecNormalize stats: {vec_norm_path}")
#         print(f"Best model directory: {models_dir / 'best_model'}")
#     except Exception as exc:
#         raise RuntimeError("Training pipeline failed.") from exc
#     finally:
#         if train_env is not None:
#             train_env.close()
#         if eval_env is not None:
#             eval_env.close()


# def main() -> None:
#     """Entry point for training script."""
#     try:
#         args = parse_args()
#         train(args)
#     except Exception as exc:
#         raise RuntimeError("train.py failed.") from exc


# if __name__ == "__main__":
#     main()
