"""MuJoCo path-tracking environment for PPO training (clean version)."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# ---------------- CONFIG ---------------- #

@dataclass(frozen=True)
class EnvConfig:
    max_steps: int = 2500
    n_waypoints: int = 25
    waypoint_threshold: float = 0.15
    arena_limit: float = 3.0
    waypoint_spacing: float = 0.3
    substeps: int = 5
    path_type: str = "figure8"


# ---------------- ENV ---------------- #

class PathTrackingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, seed=None, config=None):
        if render_mode not in {None, "human", "rgb_array"}:
            raise ValueError("Invalid render_mode")

        self.config = config or EnvConfig()
        self.render_mode = render_mode

        self._rng = np.random.default_rng(seed)
        self._base_seed = seed
        self._episode_index = 0

        # MuJoCo
        model_path = Path(__file__).resolve().parents[1] / "assets" / "robot.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        # State
        self.path = np.zeros((self.config.n_waypoints, 2), dtype=np.float32)
        self.prev_pos = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.current_waypoint_idx = 1
        self.max_waypoint_reached = 0
        self.step_count = 0
        self.prev_distance = 0.0

        self._linear_vel = 0.0
        self._angular_vel = 0.0

        self._renderer = None
        self._viewer = None

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        obs_high = np.array(
            [10,10,1,1,5,10,10,10,10,10,10,10,10,np.pi],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)


    # ---------------- PATH ---------------- #

    def _generate_waypoints(self):
        t = np.linspace(0, 2*np.pi, 1600)
        amp_x = 2.4 + self._rng.uniform(-0.15, 0.15)
        amp_y = 1.2 + self._rng.uniform(-0.1, 0.1)
        phase = self._rng.uniform(-0.25, 0.25)
        freq = 1.0 + self._rng.uniform(-0.08, 0.08)

        if self.config.path_type == "sine":
            x = amp_x * (t/(2*np.pi)*2 - 1)
            y = amp_y * np.sin(freq*t + phase)
        elif self.config.path_type == "figure8":
            theta = freq*t + phase
            x = amp_x * np.sin(theta)
            y = amp_y * np.sin(theta)*np.cos(theta)
        else:
            raise ValueError("Invalid path_type")

        raw = np.column_stack((x, y))
        return self._resample_path(raw)


    def _resample_path(self, raw):
        deltas = np.diff(raw, axis=0)
        seg_len = np.linalg.norm(deltas, axis=1)
        cum = np.concatenate(([0], np.cumsum(seg_len)))

        targets = np.arange(self.config.n_waypoints) * self.config.waypoint_spacing

        if cum[-1] < targets[-1]:
            scale = (targets[-1]+1e-6) / max(cum[-1], 1e-6)
            raw *= scale
            return self._resample_path(raw)

        x = np.interp(targets, cum, raw[:,0])
        y = np.interp(targets, cum, raw[:,1])
        return np.column_stack((x, y)).astype(np.float32)


    # ---------------- ROBOT ---------------- #

    def _robot_pose(self):
        return float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])


    def _set_robot_pose(self, x, y, yaw):
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.qpos[0], self.data.qpos[1], self.data.qpos[2] = x, y, yaw
        mujoco.mj_forward(self.model, self.data)


    def _apply_action(self, action):
        wheel_r, base, max_speed = 0.035, 0.18, 12.0
        dt = self.model.opt.timestep * self.config.substeps

        left = action[0]*max_speed*wheel_r
        right = action[1]*max_speed*wheel_r

        self._linear_vel = (left + right)/2
        self._angular_vel = (right - left)/base

        x, y, yaw = self._robot_pose()

        self.data.qpos[0] = x + self._linear_vel*np.cos(yaw)*dt
        self.data.qpos[1] = y + self._linear_vel*np.sin(yaw)*dt
        self.data.qpos[2] = yaw + self._angular_vel*dt

        mujoco.mj_forward(self.model, self.data)


    # ---------------- METRICS ---------------- #

    def _metrics(self):
        x, y, yaw = self._robot_pose()

        dists = np.linalg.norm(self.path - [x,y], axis=1)
        cross = float(np.min(dists))

        target = self.path[self.current_waypoint_idx]
        dx, dy = target[0]-x, target[1]-y
        desired = np.arctan2(dy, dx)
        heading = abs((desired - yaw + np.pi)%(2*np.pi) - np.pi)

        target_dist = np.linalg.norm(target - [x,y])

        return cross, heading, target_dist


    # ---------------- OBS ---------------- #

    def _observation(self, cross=None, heading=None):
        x, y, yaw = self._robot_pose()
        if cross is None:
            cross, heading, _ = self._metrics()

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        rel = []
        for i in range(3):
            idx = min(self.current_waypoint_idx+i, self.config.n_waypoints-1)
            dx, dy = self.path[idx] - [x,y]
            rel += [cos_y*dx + sin_y*dy, -sin_y*dx + cos_y*dy]

        obs = np.array([
            x,y,np.sin(yaw),np.cos(yaw),
            self._linear_vel,self._angular_vel,
            *rel,cross,heading
        ], dtype=np.float32)

        return np.nan_to_num(obs)


    # ---------------- RESET ---------------- #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        episode_seed = (
            seed if seed is not None
            else (self._base_seed or 0) + self._episode_index
        )
        self._rng = np.random.default_rng(episode_seed)
        self._episode_index += 1

        self.path = self._generate_waypoints()
        self.current_waypoint_idx = 1
        self.max_waypoint_reached = 0
        self.step_count = 0
        self.prev_action[:] = 0

        start = self.path[0]
        heading = np.arctan2(*(self.path[1]-start)[::-1])

        x = start[0] + self._rng.normal(0,0.02)
        y = start[1] + self._rng.normal(0,0.02)
        yaw = heading + self._rng.normal(0,0.05)

        self._set_robot_pose(x,y,yaw)
        self.prev_pos = np.array([x,y], dtype=np.float32)

        cross, heading, dist = self._metrics()
        self.prev_distance = dist

        obs = self._observation(cross, heading)

        return obs, {}


    # ---------------- STEP ---------------- #

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1, 1)

        self._apply_action(action)
        self.step_count += 1

        cross, heading, dist = self._metrics()

        # progress
        x,y,_ = self._robot_pose()
        curr = np.array([x,y])
        move = curr - self.prev_pos

        prev_i = max(0, self.current_waypoint_idx-1)
        seg = self.path[self.current_waypoint_idx] - self.path[prev_i]
        seg_dir = seg/np.linalg.norm(seg) if np.linalg.norm(seg)>1e-6 else np.zeros(2)

        progress = max(0.0, np.dot(move, seg_dir))

        # waypoint update
        while (
            self.current_waypoint_idx < self.config.n_waypoints-1
            and dist <= self.config.waypoint_threshold
        ):
            self.current_waypoint_idx += 1
            dist = np.linalg.norm(self.path[self.current_waypoint_idx]-curr)

        waypoint_gain = max(0, self.current_waypoint_idx - self.max_waypoint_reached)
        self.max_waypoint_reached = max(self.max_waypoint_reached, self.current_waypoint_idx)

        done = self.current_waypoint_idx == self.config.n_waypoints-1 and dist <= self.config.waypoint_threshold
        out = cross > self.config.arena_limit

        reward = (
            10*progress
            + self._linear_vel
            - 0.5*cross
            - 0.01*heading
            + 15*waypoint_gain
            + 50*done
            - 10*out
            - 0.01
        )

        self.prev_action = action
        self.prev_pos = curr

        obs = self._observation(cross, heading)

        info = {
            "cross_track_error": float(cross),
            "heading_error": float(heading),
            "path_completion": float(self.current_waypoint_idx / (self.config.n_waypoints - 1)),
            "position_x": float(x),
            "position_y": float(y),
        }

        return obs, float(reward), bool(done or out), self.step_count >= self.config.max_steps, info


    # ---------------- RENDER ---------------- #

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 640, 480)
            self._renderer.update_scene(self.data, camera="free_cam")
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()


    def close(self):
        if self._renderer:
            self._renderer.close()
        if self._viewer and hasattr(self._viewer, "close"):
            self._viewer.close()