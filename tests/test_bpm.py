from __future__ import annotations

import numpy as np

from rppg.bpm import estimate_bpm


def test_estimate_bpm_on_sine() -> None:
    fs = 30.0
    duration = 20.0
    f = 1.2  # Hz -> 72 BPM
    t = np.arange(0, duration, 1 / fs)
    x = np.sin(2 * np.pi * f * t).astype(np.float32)
    bpm, f_peak = estimate_bpm(x, fs=fs, fmin=0.7, fmax=4.0)
    assert 70.0 <= bpm <= 74.0
    assert abs(f_peak - f) < 0.1
