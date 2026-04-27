# MuJoCo Path Tracking RL with PPO

A portfolio-grade reinforcement learning project where a PPO agent learns differential-drive waypoint tracking in a custom MuJoCo environment.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.1.6-green)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3.2-orange)
![ROS2-ready](https://img.shields.io/badge/ROS2-ready-success)

## Architecture

```text
          +-------------------------+
          |  Figure-8 Waypoint Path |
          +-----------+-------------+
                      |
                      v
+---------------------+----------------------+
|    PathTrackingEnv (Gymnasium + MuJoCo)    |
|  obs(14): pose, heading, velocity, path    |
|  act(2): left/right wheel velocity commands|
+---------------------+----------------------+
                      |
                      v
          +-----------+-------------+
          | VecNormalize (obs/reward)|
          +-----------+-------------+
                      |
                      v
              +-------+-------+
              |   PPO Agent   |
              +-------+-------+
                      |
                      v
           +----------+-----------+
           | MuJoCo Differential  |
           | Drive Robot Dynamics |
           +----------+-----------+
                      |
                      v
         +------------+-------------+
         | Reward Components + Info |
         +--------------------------+
```

## Repository Layout

```text
path_tracking_rl/
├── envs/
│   ├── __init__.py
│   └── path_tracking_env.py
├── models/
├── logs/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── assets/
│   └── robot.xml
├── results/
├── tests/
│   └── test_env.py
├── requirements.txt
├── README.md
└── config.yaml
```

## Installation

1. Create or use a Python 3.9-3.11 environment.
2. Install project dependencies:

```bash
cd path_tracking_rl
conda run -p ../.conda311 pip install -r requirements.txt
```

3. Validate environment and tests:

```bash
conda run -p ../.conda311 python -m pytest tests -v
```

## Train

```bash
cd path_tracking_rl
conda run -p ../.conda311 python scripts/train.py \
  --timesteps 500000 \
  --seed 42 \
  --run-name ppo_path_tracking
```

TensorBoard:

```bash
tensorboard --logdir logs
```

## Evaluate

```bash
cd path_tracking_rl
conda run -p ../.conda311 python scripts/evaluate.py --episodes 10
```

Live MuJoCo viewer:

```bash
cd path_tracking_rl
conda run -p ../.conda311 python scripts/evaluate.py --episodes 1 --render
```

Record robot motion as GIF:

```bash
cd path_tracking_rl
conda run -p ../.conda311 python scripts/evaluate.py \
  --episodes 1 \
  --record-gif robot_motion.gif
```

Metrics are saved to:

- results/eval_metrics.csv
- results/robot_motion.gif

## Visualize

```bash
cd path_tracking_rl
conda run -p ../.conda311 python scripts/visualize.py --episodes 3
```

Generated artifacts (300 DPI PNG):

- results/path_overlay.png
- results/cross_track_error.png
- results/training_reward_curve.png
- results/reward_components_breakdown.png

## Results

Placeholders for project showcase assets:

- GIF: results/path_tracking_demo.gif
- Plot: results/path_overlay.png
- Plot: results/cross_track_error.png
- Plot: results/training_reward_curve.png
- Plot: results/reward_components_breakdown.png

## Future Work

- Deploy policy to ROS 2 with ros2_sb3_bridge for real-time robot control.
- Add domain randomization for sim-to-real robustness.
- Integrate LiDAR or vision observations for obstacle-aware tracking.

## License

MIT
