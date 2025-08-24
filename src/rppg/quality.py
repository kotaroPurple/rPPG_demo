"""Quality metrics for rPPG signals."""

from __future__ import annotations

import numpy as np


def snr_db(
    power_spectrum: np.ndarray,
    peak_index: int,
    guard_bins: int = 1,
    band_bins: int = 5,
) -> float:
    """Simple SNR estimate around a known peak index in the spectrum.

    Args:
        power_spectrum: magnitude or power spectrum (1D array).
        peak_index: index of the detected peak within the array.
        guard_bins: bins excluded around the peak when computing noise.
        band_bins: number of bins on either side used as the signal band.
    """
    p = np.asarray(power_spectrum, dtype=np.float32)
    n = p.size
    i0 = max(0, peak_index - band_bins)
    i1 = min(n, peak_index + band_bins + 1)
    sig = np.sum(p[i0:i1])
    noise = np.sum(p[: max(0, i0 - guard_bins)]) + np.sum(p[min(n, i1 + guard_bins) :])
    if noise <= 0:
        return 0.0
    return 10.0 * float(np.log10(sig / noise))
