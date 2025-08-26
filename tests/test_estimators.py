from __future__ import annotations

import numpy as np

from rppg.acf_bpm import estimate_bpm_acf
from rppg.bpm import estimate_bpm
from rppg.hilbert_if import estimate_bpm_if


def test_fft_and_acf_bpm_on_sine_with_noise() -> None:
    fs = 30.0
    dur = 12.0
    f = 1.5  # 90 BPM
    t = np.arange(0, dur, 1 / fs)
    x = np.sin(2 * np.pi * f * t).astype(np.float32)
    x += 0.2 * np.random.RandomState(0).randn(t.size).astype(np.float32)

    bpm_fft, fpk = estimate_bpm(x, fs=fs, fmin=0.7, fmax=4.0)
    ar = estimate_bpm_acf(x, fs=fs, bpm_min=42.0, bpm_max=240.0)

    assert 85.0 <= bpm_fft <= 95.0
    assert abs(fpk - f) < 0.2
    assert ar.bpm is not None and 85.0 <= ar.bpm <= 95.0


def test_hilbert_if_tracks_frequency() -> None:
    fs = 30.0
    dur = 10.0
    f = 1.2  # 72 BPM
    t = np.arange(0, dur, 1 / fs)
    x = np.sin(2 * np.pi * f * t).astype(np.float32)
    ir = estimate_bpm_if(x, fs=fs, smooth_len=5)
    assert ir.bpm is not None and 70.0 <= ir.bpm <= 74.0

