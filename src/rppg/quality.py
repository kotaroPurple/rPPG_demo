"""Quality metrics for rPPG signals.

Includes a simple SNR and a normalized peak confidence in the spectrum.
"""

from __future__ import annotations

import numpy as np


def snr_db(
    power_spectrum: np.ndarray,
    peak_index: int,
    guard_bins: int = 1,
    band_bins: int = 1,
) -> float:
    """Estimate SNR at a known peak using local noise floor.

    Signal is taken as the mean magnitude within the peak band (±band_bins),
    while noise is estimated as the median magnitude of bins outside a guard
    region (±guard_bins beyond the band). This is robust to outliers and
    independent of spectrum length.
    """
    p = np.asarray(power_spectrum, dtype=np.float32)
    n = int(p.size)
    if n == 0:
        return 0.0
    # Clip band around peak
    i0 = max(0, int(peak_index) - int(band_bins))
    i1 = min(n, int(peak_index) + int(band_bins) + 1)
    sig = float(np.mean(p[i0:i1])) if i1 > i0 else float(p[int(peak_index)])
    # Build noise mask excluding a guard region around the band
    g0 = max(0, i0 - int(guard_bins))
    g1 = min(n, i1 + int(guard_bins))
    if g0 <= 0 and g1 >= n:
        # No room for noise estimation
        return 0.0
    noise_bins = np.concatenate([p[:g0], p[g1:]]) if g1 < n else p[:g0]
    if noise_bins.size == 0:
        return 0.0
    noise = float(np.median(noise_bins))
    if noise <= 0.0:
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
