"""BPM estimation utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_bpm(
    signal: np.ndarray,
    fs: float,
    fmin: float = 0.7,
    fmax: float = 4.0,
) -> Tuple[float, float]:
    """Estimate BPM by peak in band-limited magnitude spectrum.

    Returns (bpm, peak_freq_hz). If invalid, returns (0.0, 0.0).
    """
    x = np.asarray(signal, dtype=np.float32)
    n = x.size
    if n < 8 or fs <= 0:
        return 0.0, 0.0
    # Hann window to reduce leakage
    w = np.hanning(n).astype(np.float32)
    X = np.fft.rfft((x - x.mean()) * w)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (freqs >= max(0.0, fmin)) & (freqs <= fmax)
    if not np.any(band):
        return 0.0, 0.0
    mag = np.abs(X)
    idx = np.argmax(mag * band)
    f_peak = float(freqs[idx])
    bpm = float(60.0 * f_peak) if f_peak > 0 else 0.0
    return bpm, f_peak
