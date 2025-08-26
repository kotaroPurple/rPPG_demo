from __future__ import annotations

import numpy as np

from rppg.tracker import FreqTracker, TrackConfig


def test_freq_tracker_converges_to_measurement() -> None:
    # True frequency ~1.2 Hz (72 BPM)
    true_f = 1.2
    cfg = TrackConfig(q_freq=0.01, q_drift=0.005, r_meas=0.05, f_min=0.7, f_max=4.0)
    trk = FreqTracker(cfg)
    trk.reset(f_init=1.0)

    rng = np.random.RandomState(1)
    t = 0.0
    for _ in range(100):
        dt = 0.1
        t += dt
        trk.predict(dt)
        meas = true_f + 0.03 * rng.randn()
        trk.update(float(meas), quality=1.0)

    bpm = trk.value_bpm()
    assert bpm is not None
    assert 69.0 <= bpm <= 75.0

