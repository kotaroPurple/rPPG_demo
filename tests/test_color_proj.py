from __future__ import annotations

import numpy as np

from rppg.chrom import chrom_signal
from rppg.pos import pos_signal


def test_color_projections_shapes() -> None:
    n = 128
    R = np.random.RandomState(0).randn(n).astype(np.float32)
    G = np.random.RandomState(1).randn(n).astype(np.float32)
    B = np.random.RandomState(2).randn(n).astype(np.float32)

    s_chrom = chrom_signal(R, G, B)
    s_pos = pos_signal(R, G, B)

    assert s_chrom.shape == (n,)
    assert s_pos.shape == (n,)
    # Non-trivial outputs
    assert np.isfinite(s_chrom).all()
    assert np.isfinite(s_pos).all()
