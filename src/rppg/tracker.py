"""Simple frequency tracker (alpha-beta / Kalman-like) for BPM.

Tracks heart rate as frequency with optional acceleration (drift) using a
constant-velocity model. Measurement noise can be adapted from signal quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackConfig:
    q_freq: float = 0.05   # process noise for frequency (Hz^2)
    q_drift: float = 0.01  # process noise for drift (Hz^2)
    r_meas: float = 0.1    # nominal measurement noise (Hz^2)
    f_min: float = 0.7     # Hz (42 BPM)
    f_max: float = 4.0     # Hz (240 BPM)


@dataclass
class TrackState:
    f: float  # Hz
    fd: float  # Hz/s


class FreqTracker:
    def __init__(self, cfg: TrackConfig | None = None) -> None:
        self.cfg = cfg or TrackConfig()
        self.x = None  # state vector [f, fd]
        self.P = None  # covariance 2x2

    def reset(self, f_init: float, fd_init: float = 0.0) -> None:
        self.x = np.array([f_init, fd_init], dtype=np.float32)
        self.P = np.eye(2, dtype=np.float32)

    def predict(self, dt: float) -> None:
        if self.x is None or self.P is None:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
        Q = np.array(
            [[self.cfg.q_freq * dt, 0.0], [0.0, self.cfg.q_drift * dt]], dtype=np.float32
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        # Clamp frequency to plausible range
        self.x[0] = float(np.clip(self.x[0], self.cfg.f_min, self.cfg.f_max))

    def update(self, f_meas: float, quality: float = 1.0) -> None:
        if self.x is None or self.P is None:
            # initialize at first measurement
            self.reset(f_meas)
            return
        H = np.array([[1.0, 0.0]], dtype=np.float32)
        # Adapt measurement noise by (1/quality)
        q = float(np.clip(quality, 1e-3, 1.0))
        R = np.array([[self.cfg.r_meas / q]], dtype=np.float32)
        y = np.array([f_meas], dtype=np.float32) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).reshape(-1)
        eye2 = np.eye(2, dtype=np.float32)
        self.P = (eye2 - K @ H) @ self.P
        # Clamp again
        self.x[0] = float(np.clip(self.x[0], self.cfg.f_min, self.cfg.f_max))

    def value_bpm(self) -> float | None:
        if self.x is None:
            return None
        return float(self.x[0] * 60.0)
