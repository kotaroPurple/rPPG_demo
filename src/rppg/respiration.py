"""Respiration rate (RR) estimation from rPPG signal.

Provides simple RR estimators based on the amplitude envelope of the
band-limited rPPG signal. Designed to run alongside HR estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert


@dataclass
class RrResult:
    brpm: float | None  # breaths per minute
    env: np.ndarray  # envelope waveform (same length as input)


def estimate_rr_envelope(
    s: np.ndarray,
    fs: float,
    rr_min_hz: float = 0.1,
    rr_max_hz: float = 0.5,
) -> RrResult:
    """Estimate respiration rate by analyzing the amplitude envelope.

    Args:
        s: rPPG signal within HR band (1D float array)
        fs: sampling rate (Hz)
        rr_min_hz/rr_max_hz: respiration band (Hz)

    Returns:
        RrResult with BrPM and the envelope waveform.
    """
    if s.size < 8 or fs <= 0:
        return RrResult(None, np.zeros(0, dtype=np.float32))
    x = np.asarray(s, dtype=np.float32)
    x = x - float(np.mean(x))
    # Amplitude envelope via analytic signal
    env = np.abs(hilbert(x)).astype(np.float32)
    # Normalize envelope to unit variance for stability
    std = float(np.std(env))
    if std > 0:
        env_n = (env - float(np.mean(env))) / std
    else:
        env_n = env
    # Spectrum and peak within RR band
    X = np.fft.rfft(env_n * np.hanning(env_n.size).astype(np.float32))
    freqs = np.fft.rfftfreq(env_n.size, d=1.0 / fs)
    mag = np.abs(X)
    band = (freqs >= rr_min_hz) & (freqs <= rr_max_hz)
    if not np.any(band):
        return RrResult(None, env)
    idx = int(np.argmax(mag * band))
    f_rr = float(freqs[idx])
    brpm = 60.0 * f_rr if f_rr > 0 else None
    return RrResult(brpm, env)

