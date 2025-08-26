"""Quality metrics for rPPG signals.

Includes a simple SNR and a normalized peak confidence in the spectrum.
"""

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


def peak_confidence(
    power_spectrum: np.ndarray,
    peak_index: int,
    neighborhood: int = 2,
) -> float:
    """Return a 0..1 confidence based on peak prominence.

    Confidence is computed as (peak - median(neighborhood)) / (peak + median), clipped to [0, 1].

    Args:
        power_spectrum: magnitude or power spectrum (1D array).
        peak_index: index of the detected peak within the array.
        neighborhood: half-width around the peak to compute a local median.
    """
    p = np.asarray(power_spectrum, dtype=np.float32)
    if p.size == 0:
        return 0.0
    i0 = max(0, peak_index - neighborhood)
    i1 = min(p.size, peak_index + neighborhood + 1)
    local = p[i0:i1]
    peak = float(p[peak_index]) if 0 <= peak_index < p.size else 0.0
    med = float(np.median(local)) if local.size > 0 else 0.0
    num = max(0.0, peak - med)
    den = max(1e-9, peak + med)
    c = num / den
    return float(np.clip(c, 0.0, 1.0))
