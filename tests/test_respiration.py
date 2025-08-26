from __future__ import annotations

import numpy as np

from rppg.respiration import estimate_rr_envelope


def test_respiration_from_am_amplitude_modulated_signal() -> None:
    # Simulate s(t) = (1 + a*sin(2π f_rr t)) * sin(2π f_hr t)
    fs = 30.0
    dur = 20.0
    f_hr = 1.4  # ~84 BPM
    f_rr = 0.25  # 15 BrPM
    t = np.arange(0, dur, 1 / fs)
    env = 1.0 + 0.5 * np.sin(2 * np.pi * f_rr * t)
    s = (env * np.sin(2 * np.pi * f_hr * t)).astype(np.float32)

    rr = estimate_rr_envelope(s, fs=fs, rr_min_hz=0.1, rr_max_hz=0.5)
    assert rr.brpm is not None
    assert 13.0 <= rr.brpm <= 17.0

