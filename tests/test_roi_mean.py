from __future__ import annotations

import numpy as np

from rppg.roi import mean_rgb


def test_mean_rgb_with_and_without_mask() -> None:
    # Create a simple 2x2 RGB image
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    # No mask -> mean over all pixels
    r, g, b = mean_rgb(rgb, mask=None)
    assert np.isclose(r, (255 + 0 + 0 + 255) / 4.0)
    assert np.isclose(g, (0 + 255 + 0 + 255) / 4.0)
    assert np.isclose(b, (0 + 0 + 255 + 255) / 4.0)

    # Mask only the top row
    mask = np.array([[1, 1], [0, 0]], dtype=bool)
    r2, g2, b2 = mean_rgb(rgb, mask=mask)
    assert np.isclose(r2, (255 + 0) / 2.0)
    assert np.isclose(g2, (0 + 255) / 2.0)
    assert np.isclose(b2, (0 + 0) / 2.0)

