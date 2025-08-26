"""Hilbert-based instantaneous frequency BPM estimator.

Computes instantaneous phase via analytic signal and differentiates to obtain
instantaneous frequency, then maps to BPM. A light smoothing is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass
class IfResult:
    bpm: float | None
    if_series_bpm: np.ndarray  # instantaneous BPM series over the window


def estimate_bpm_if(
    x: np.ndarray,
    fs: float,
    smooth_len: int = 5,
) -> IfResult:
    """Estimate BPM from instantaneous frequency of x.

    Args:
        x: time-domain rPPG signal (1D array)
        fs: sampling rate (Hz)
        smooth_len: moving-average length for IF smoothing (samples)
    """
    if x.size < 8 or fs <= 0:
        return IfResult(None, np.zeros(0, dtype=np.float32))
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    a = hilbert(x)
    phase = np.unwrap(np.angle(a)).astype(np.float32)
    # Numerical derivative (central differences where possible)
    dphi = np.empty_like(phase)
    dphi[1:-1] = 0.5 * (phase[2:] - phase[:-2])
    dphi[0] = phase[1] - phase[0]
    dphi[-1] = phase[-1] - phase[-2]
    inst_freq = (fs / (2.0 * np.pi)) * dphi
    inst_bpm = 60.0 * inst_freq
    # Smooth with moving average to reduce jitter
    L = max(1, int(smooth_len))
    if L > 1:
        k = np.ones(L, dtype=np.float32) / float(L)
        inst_bpm = np.convolve(inst_bpm, k, mode="same")
    bpm_val = float(np.median(inst_bpm[-L:])) if inst_bpm.size > 0 else None
    return IfResult(bpm_val, inst_bpm.astype(np.float32))

