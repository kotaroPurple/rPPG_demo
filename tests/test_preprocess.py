from __future__ import annotations

import numpy as np

from rppg.preprocess import bandpass, moving_average_normalize


def test_moving_average_normalize_constant_signal() -> None:
    x = np.ones(100, dtype=np.float32)
    y = moving_average_normalize(x, win=10)
    # ratio minus 1 on constant is approximately 0
    assert np.allclose(y, 0.0, atol=1e-6)


def test_bandpass_preserves_inband_and_attenuates_outband() -> None:
    fs = 30.0
    t = np.arange(0, 10.0, 1 / fs)
    # In-band 1.2 Hz and out-of-band 0.2 Hz components
    x = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.2 * t)
    y = bandpass(x, fs=fs, fmin=0.7, fmax=4.0)
    # Correlate with in-band component should be strong
    corr = np.corrcoef(y, np.sin(2 * np.pi * 1.2 * t))[0, 1]
    assert corr > 0.7
