"""ROI extraction and mean RGB utilities.

This module will integrate MediaPipe for robust face landmarks; for now, it
contains lightweight helpers not to block early testing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def mean_rgb(
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute mean RGB over an optional boolean mask.

    Args:
        frame_bgr: HxWx3 uint8 or float array in BGR order.
        mask: optional HxW boolean array; True selects pixels to include.

    Returns:
        (R, G, B) means as floats.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be HxWx3 array")
    bgr = frame_bgr.astype(np.float32)
    if mask is not None:
        if mask.shape != frame_bgr.shape[:2]:
            raise ValueError("mask must match frame spatial shape")
        m = mask.astype(bool)
        if not np.any(m):
            return 0.0, 0.0, 0.0
        sel = bgr[m]
    else:
        sel = bgr.reshape(-1, 3)
    # Convert BGR to RGB means
    b_mean, g_mean, r_mean = sel.mean(axis=0)
    return float(r_mean), float(g_mean), float(b_mean)
