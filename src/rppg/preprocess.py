"""Signal preprocessing for rPPG."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def moving_average_normalize(x: np.ndarray, win: int) -> np.ndarray:
    """Normalize a 1D signal by its moving average (ratio minus 1).

    Args:
        x: 1D array.
        win: window length in samples (>=1).
    """
    x = np.asarray(x, dtype=np.float32)
    win = max(int(win), 1)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    # Edge-aware moving average using edge padding to keep scale near boundaries
    pad_left = win // 2
    pad_right = win - 1 - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    mean = np.convolve(xp, kernel, mode="valid")
    mean[mean == 0] = 1.0
    return x / mean - 1.0


def bandpass(
    x: np.ndarray,
    fs: float,
    fmin: float = 0.7,
    fmax: float = 4.0,
    order: int = 3,
) -> np.ndarray:
    """Causal Butterworth band-pass filter（lfilter）。

    Args:
        x: 1D array.
        fs: sampling rate [Hz].
        fmin: low cut [Hz].
        fmax: high cut [Hz].
        order: IIR order.
    """
    x = np.asarray(x, dtype=np.float32)
    nyq = 0.5 * fs
    low = max(1e-6, fmin / nyq)
    high = min(0.999, fmax / nyq)
    if not (0 < low < high < 1):
        return x.copy()
    b, a = butter(order, [low, high], btype="band")
    # 組み込み安定性優先で常に lfilter を使用（ゼロ位相は要求しない）
    return lfilter(b, a, x)
