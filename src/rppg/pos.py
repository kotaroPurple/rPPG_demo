"""POS color projection."""

from __future__ import annotations

import numpy as np


def pos_signal(Rn: np.ndarray, Gn: np.ndarray, Bn: np.ndarray) -> np.ndarray:
    """Compute POS composite signal for a window of normalized RGB.

    Args:
        Rn, Gn, Bn: 1D arrays of equal length (normalized & filtered).
    """
    Rn = np.asarray(Rn, dtype=np.float32)
    Gn = np.asarray(Gn, dtype=np.float32)
    Bn = np.asarray(Bn, dtype=np.float32)
    X = Gn - Bn
    Y = -2 * Rn + Gn + Bn
    sx = np.std(X) or 1.0
    sy = np.std(Y) or 1.0
    alpha = sx / sy
    return X + alpha * Y

