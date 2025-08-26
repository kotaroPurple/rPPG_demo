from __future__ import annotations

import numpy as np

from rppg.quality import peak_confidence, snr_db


def test_snr_db_peak_higher_than_noise() -> None:
    # Construct a spectrum with a clear peak at index 10
    p = np.ones(64, dtype=np.float32)
    p[10] = 50.0
    s = snr_db(p, peak_index=10, guard_bins=1, band_bins=2)
    assert s > 10.0  # >10 dB indicates a strong peak


def test_peak_confidence_behaviour() -> None:
    p = np.ones(64, dtype=np.float32)
    # Low confidence when flat
    c0 = peak_confidence(p, peak_index=5)
    assert 0.0 <= c0 <= 0.1
    # High confidence when clear peak exists
    p[5] = 100.0
    c1 = peak_confidence(p, peak_index=5)
    assert 0.5 <= c1 <= 1.0

