import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.path_tracking_env2 import PathTrackingEnv, EnvConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--vecnorm-path", type=str, required=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--record-gif", type=str, default="")
    p.add_argument("--gif-fps", type=int, default=20)
    return p.parse_args()


def save_gif(frames, path, fps):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
    imgs[0].save(
        str(path),
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def main():
    args = parse_args()

    root = PROJECT_ROOT
    model_path = root / args.model_path
    vec_path = root / args.vecnorm_path
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    render_mode = None
    if args.record_gif:
        render_mode = "rgb_array"
    elif args.render:
        render_mode = "human"

    config = yaml.safe_load(open(PROJECT_ROOT / "config.yaml"))
    env_cfg = config.get("env", {})
    env_config = EnvConfig(**env_cfg)

    env = DummyVecEnv([
        lambda: Monitor(PathTrackingEnv(render_mode=render_mode, seed=args.seed, config=env_config))
    ])
    
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)

    base_env = env.venv.envs[0].env  # unwrap

    gif_frames = []
    gif_path = results_dir / args.record_gif if args.record_gif else None

    rows = []

    for ep in range(args.episodes):
        obs = env.reset()
        done = False

        total_reward = 0.0
        steps = 0

        cte = []
        he = []
        completion = 0.0

        if gif_path and ep == 0:
            frame = base_env.render()
            if frame is not None:
                gif_frames.append(frame.copy())

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            r = float(rewards[0])
            total_reward += r
            steps += 1

            info = infos[0]

            if "cross_track_error" in info:
                cte.append(info["cross_track_error"])
            if "heading_error" in info:
                he.append(info["heading_error"])
            if "path_completion" in info:
                completion = max(completion, info["path_completion"])

            if gif_path and ep == 0:
                frame = base_env.render()
                if frame is not None:
                    gif_frames.append(frame.copy())

            done = bool(dones[0])

        rows.append({
            "episode": ep + 1,
            "step_count": steps,
            "total_reward": total_reward,
            "path_completion_pct": 100.0 * completion,
            "mean_cross_track_error": np.mean(cte) if cte else 0.0,
            "mean_heading_error": np.mean(he) if he else 0.0,
        })

    df = pd.DataFrame(rows)
    out_csv = results_dir / "eval_metrics.csv"
    df.to_csv(out_csv, index=False)

    if gif_path:
        save_gif(gif_frames, gif_path, args.gif_fps)

    print("\nResults:")
    print(df)

    print("\nMean:")
    print(df.mean(numeric_only=True))

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()



# """Evaluate a trained PPO path-tracking agent."""

# from __future__ import annotations

# import argparse
# import os
# import random
# import sys
# from pathlib import Path
# from typing import Dict, List

# import numpy as np
# import pandas as pd
# import torch as th
# import yaml
# from PIL import Image
# from stable_baselines3 import PPO
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# from envs.path_tracking_env2 import EnvConfig, PathTrackingEnv  # noqa: E402


# def parse_args() -> argparse.Namespace:
#     """Parse CLI arguments for evaluation."""
#     try:
#         parser = argparse.ArgumentParser(
#             description="Evaluate PPO path tracker"
#         )
#         parser.add_argument(
#             "--episodes",
#             type=int,
#             default=10,
#             help="Number of evaluation episodes",
#         )
#         parser.add_argument(
#             "--seed",
#             type=int,
#             default=42,
#             help="Random seed",
#         )
#         parser.add_argument(
#             "--model-path",
#             type=str,
#             default="models/ppo_path_tracking_final.zip",
#             help="Model path relative to project root",
#         )
#         parser.add_argument(
#             "--vecnorm-path",
#             type=str,
#             default="models/vec_normalize.pkl",
#             help="VecNormalize path relative to project root",
#         )
#         parser.add_argument(
#             "--render",
#             action="store_true",
#             help="Render rollout in MuJoCo viewer (human mode)",
#         )
#         parser.add_argument(
#             "--record-gif",
#             type=str,
#             default="",
#             help="Optional GIF filename relative to results/",
#         )
#         parser.add_argument(
#             "--gif-fps",
#             type=int,
#             default=20,
#             help="Frame rate for recorded GIF",
#         )
#         return parser.parse_args()
#     except Exception as exc:
#         raise RuntimeError("Failed to parse evaluation args.") from exc


# def set_global_seeds(seed: int) -> None:
#     """Set seeds for deterministic evaluation rollout."""
#     try:
#         random.seed(seed)
#         np.random.seed(seed)
#         th.manual_seed(seed)
#     except Exception as exc:
#         raise RuntimeError("Failed to set evaluation seeds.") from exc


# def load_config(config_path: Path) -> Dict[str, object]:
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


# def build_env_config(config: Dict[str, object]) -> EnvConfig:
#     """Build EnvConfig from YAML configuration."""
#     try:
#         env_cfg = config.get("env", {}) if isinstance(config, dict) else {}
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


# def _unwrap_env(maybe_wrapped_env: object) -> PathTrackingEnv:
#     """Unwrap wrappers and return the base PathTrackingEnv instance."""
#     try:
#         current = maybe_wrapped_env
#         for _ in range(8):
#             if isinstance(current, PathTrackingEnv):
#                 return current
#             if hasattr(current, "env"):
#                 current = getattr(current, "env")
#                 continue
#             break
#         raise TypeError("Could not unwrap PathTrackingEnv from wrapper.")
#     except Exception as exc:
#         raise RuntimeError("Failed to unwrap base environment.") from exc


# def _save_gif(frames: List[np.ndarray], output_path: Path, fps: int) -> None:
#     """Save a list of RGB frames to a GIF file."""
#     try:
#         if not frames:
#             raise ValueError("No frames collected for GIF export.")

#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
#         frame_duration_ms = int(1000 / max(1, fps))
#         images[0].save(
#             str(output_path),
#             save_all=True,
#             append_images=images[1:],
#             duration=frame_duration_ms,
#             loop=0,
#         )
#     except Exception as exc:
#         raise RuntimeError(f"Failed to save GIF at {output_path}.") from exc


# def evaluate(args: argparse.Namespace) -> pd.DataFrame:
#     """Evaluate trained model and return per-episode metrics."""
#     vec_env = None
#     try:
#         project_root = Path(__file__).resolve().parents[1]
#         config = load_config(project_root / "config.yaml")
#         env_config = build_env_config(config)
#         model_path = project_root / args.model_path
#         vecnorm_path = project_root / args.vecnorm_path
#         results_dir = project_root / "results"
#         results_dir.mkdir(parents=True, exist_ok=True)

#         if not model_path.exists():
#             raise FileNotFoundError(f"Model not found: {model_path}")
#         if not vecnorm_path.exists():
#             raise FileNotFoundError(
#                 f"VecNormalize stats not found: {vecnorm_path}"
#             )

#         set_global_seeds(args.seed)

#         has_display = bool(
#             os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
#         )
#         if args.render and (not args.record_gif) and (not has_display):
#             args.record_gif = "robot_motion.gif"
#             print(
#                 "No GUI display detected. Falling back to offscreen GIF "
#                 "recording at results/robot_motion.gif"
#             )

#         render_mode = None
#         if args.record_gif:
#             render_mode = "rgb_array"
#         elif args.render:
#             render_mode = "human"

#         base_env = DummyVecEnv(
#             [
#                 lambda: Monitor(
#                     PathTrackingEnv(
#                         render_mode=render_mode,
#                         seed=args.seed,
#                         config=env_config,
#                     )
#                 )
#             ]
#         )
#         vec_env = VecNormalize.load(str(vecnorm_path), base_env)
#         vec_env.training = False
#         vec_env.norm_reward = False

#         model = PPO.load(str(model_path), env=vec_env)
#         path_env = _unwrap_env(vec_env.venv.envs[0])

#         gif_frames: List[np.ndarray] = []
#         gif_path: Path | None = None
#         if args.record_gif:
#             record_path = Path(args.record_gif)
#             if record_path.is_absolute():
#                 gif_path = record_path
#             else:
#                 gif_path = results_dir / record_path

#         rows: List[Dict[str, float]] = []
#         for episode in range(1, args.episodes + 1):
#             obs = vec_env.reset()
#             done = False
#             total_reward = 0.0
#             cross_track_errors: List[float] = []
#             heading_errors: List[float] = []
#             completion = 0.0

#             if args.record_gif and episode == 1:
#                 first_frame = path_env.render()
#                 if first_frame is not None:
#                     gif_frames.append(first_frame.copy())
            
#             step_count = 0
#             while not done:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, rewards, dones, infos = vec_env.step(action)
#                 total_reward += float(rewards[0])
#                 step_count += 1
#                 if args.record_gif and episode == 1:
#                     frame = path_env.render()
#                     if frame is not None:
#                         gif_frames.append(frame.copy())

#                 info = infos[0]
#                 if "cross_track_error" in info:
#                     cross_track_errors.append(float(info["cross_track_error"]))
#                 if "heading_error" in info:
#                     heading_errors.append(float(info["heading_error"]))
#                 if "path_completion" in info:
#                     completion = max(
#                         completion,
#                         float(info["path_completion"]),
#                     )

#                 done = bool(dones[0])

#             rows.append(
#                 {
#                     "episode": float(episode),
#                     "step_count": float(step_count),
#                     "total_reward": total_reward,
#                     "path_completion_pct": 100.0 * completion,
#                     "mean_cross_track_error": float(
#                         np.mean(cross_track_errors)
#                     ),
#                     "mean_heading_error": float(np.mean(heading_errors)),
#                 }
#             )

#         metrics_df = pd.DataFrame(rows)
#         output_path = results_dir / "eval_metrics.csv"
#         metrics_df.to_csv(output_path, index=False)

#         if args.record_gif and gif_path is not None:
#             _save_gif(gif_frames, gif_path, args.gif_fps)
#             print(f"Saved MuJoCo rollout GIF to: {gif_path}")

#         print("\nEvaluation metrics")
#         print("------------------")
#         print(metrics_df.to_string(index=False, float_format="{:.4f}".format))

#         summary = metrics_df[
#             [
#                 "total_reward",
#                 "path_completion_pct",
#                 "mean_cross_track_error",
#                 "mean_heading_error",
#             ]
#         ].mean()
#         print("\nAggregate means")
#         print("---------------")
#         print(summary.to_string(float_format="{:.4f}".format))
#         print(f"\nMetrics CSV saved to: {output_path}")

#         mean_cte = float(metrics_df["mean_cross_track_error"].mean())
#         if mean_cte < 0.3:
#             print("Cross-track target met: mean error < 0.3 m")
#         else:
#             print("Cross-track target not met: mean error >= 0.3 m")
#         return metrics_df
#     except Exception as exc:
#         raise RuntimeError("Evaluation pipeline failed.") from exc
#     finally:
#         if vec_env is not None:
#             vec_env.close()


# def main() -> None:
#     """Entry point for evaluation script."""
#     try:
#         args = parse_args()
#         evaluate(args)
#     except Exception as exc:
#         raise RuntimeError("evaluate.py failed.") from exc


# if __name__ == "__main__":
#     main()
