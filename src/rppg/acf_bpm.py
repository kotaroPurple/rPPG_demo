"""Autocorrelation-based BPM estimation for short windows.

The ACF method is more robust on short windows than FFT peak-picking and
handles non-stationary changes better when used with high overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AcfResult:
    bpm: float | None
    peak_lag: float | None  # seconds
    score: float  # 0..1 relative peak strength


def _parabolic_interp(y: np.ndarray, i: int) -> tuple[float, float]:
    """Quadratic interpolation around index i. Returns (x_peak, y_peak)."""
    i0 = max(0, i - 1)
    i2 = min(y.size - 1, i + 1)
    y0, y1, y2 = float(y[i0]), float(y[i]), float(y[i2])
    denom = (y0 - 2 * y1 + y2)
    if denom == 0.0:
        return float(i), y1
    x = i + 0.5 * (y0 - y2) / denom
    a = 0.5 * denom
    b = 0.5 * (y2 - y0)
    c = y1
    y_peak = a * (x - i) ** 2 + b * (x - i) + c
    return float(x), float(y_peak)


def estimate_bpm_acf(
    x: np.ndarray,
    fs: float,
    bpm_min: float = 42.0,
    bpm_max: float = 240.0,
) -> AcfResult:
    """Estimate BPM by finding the first dominant ACF peak in HR range.

    Args:
        x: time-domain rPPG signal (1D array).
        fs: sampling rate (Hz).
        bpm_min/bpm_max: search bounds (BPM).

    Returns:
        AcfResult with BPM, peak lag (s), and a simple 0..1 score.
    """
    if x.size < 8 or fs <= 0:
        return AcfResult(None, None, 0.0)
    x = np.asarray(x, dtype=np.float32)
    x = x - float(np.mean(x))
    # Normalize to unit variance to stabilize ACF amplitude
    std = float(np.std(x))
    if std > 0:
        x = x / std
    # FFT-based autocorrelation (Wiener–Khinchin)
    n = int(2 ** int(np.ceil(np.log2(2 * x.size - 1))))
    X = np.fft.rfft(x, n=n)
    acf = np.fft.irfft(np.abs(X) ** 2, n=n)[: x.size]
    # Consider only positive lags (exclude lag=0 region by a guard)
    lag_min = int(np.floor(fs * 60.0 / bpm_max))
    lag_max = int(np.ceil(fs * 60.0 / bpm_min))
    lag_min = max(1, lag_min)
    lag_max = min(acf.size - 1, max(lag_min + 2, lag_max))
    roi = acf[lag_min : lag_max + 1]
    if roi.size <= 3:
        return AcfResult(None, None, 0.0)
    k = int(np.argmax(roi)) + lag_min
    # Parabolic interpolation for sub-sample lag
    xk, yk = _parabolic_interp(acf, k)
    lag_sec = xk / fs
    if lag_sec <= 0:
        return AcfResult(None, None, 0.0)
    bpm = 60.0 / lag_sec
    # Relative peak score vs local median baseline in ROI
    med = float(np.median(roi))
    if med <= 1e-6:
        score = 1.0
    else:
        score = float(np.clip((yk - med) / (abs(yk) + med), 0.0, 1.0))
    return AcfResult(float(bpm), float(lag_sec), score)

