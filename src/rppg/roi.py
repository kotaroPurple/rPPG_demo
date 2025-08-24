"""ROI extraction and mean RGB utilities.

Provides two ROI detectors:
- FaceCascadeROI: OpenCV Haar-cascade based (lightweight, no TFLite)
- FaceBoxROI: MediaPipe Face Detection based（CPU/TFLite）。

Both return simple cheek/forehead rectangular masks from a detected face box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def mean_rgb(
    frame_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute mean RGB over an optional boolean mask.

    Args:
        frame_bgr: HxWx3 uint8 or float array in BGR order.
        mask: optional HxW boolean array; True selects pixels to include.

    Returns:
        (R, G, B) means as floats.
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be HxWx3 array")
    bgr = frame_bgr.astype(np.float32)
    if mask is not None:
        if mask.shape != frame_bgr.shape[:2]:
            raise ValueError("mask must match frame spatial shape")
        m = mask.astype(bool)
        if not np.any(m):
            return 0.0, 0.0, 0.0
        sel = bgr[m]
    else:
        sel = bgr.reshape(-1, 3)
    # Convert BGR to RGB means
    b_mean, g_mean, r_mean = sel.mean(axis=0)
    return float(r_mean), float(g_mean), float(b_mean)


@dataclass
class FaceRoiConfig:
    downscale: int = 2  # speed-up for detection
    min_confidence: float = 0.5


class FaceBoxROI:
    """Face bounding-box based ROI builder using MediaPipe Face Detection.

    Defines three rectangular ROIs (left/right cheek, forehead) inside the
    detected face box and returns a combined boolean mask.
    """

    def __init__(self, cfg: Optional[FaceRoiConfig] = None) -> None:
        self.cfg = cfg or FaceRoiConfig()
        self._fd = None

    def _ensure_model(self) -> None:
        if self._fd is None:
            try:
                import mediapipe as mp  # type: ignore

                self._fd = mp.solutions.face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.cfg.min_confidence,
                )
            except Exception as exc:  # pragma: no cover - optional path
                raise RuntimeError(
                    f"Failed to initialize MediaPipe FaceDetection: {exc}"
                ) from exc

    def mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Compute a boolean mask of cheek+forehead regions.

        Args:
            frame_rgb: HxWx3 RGB uint8 array.

        Returns:
            HxW boolean mask.
        """
        self._ensure_model()
        h, w, _ = frame_rgb.shape
        # Downscale for speed
        if self.cfg.downscale > 1:
            ds = self.cfg.downscale
            small = frame_rgb[::ds, ::ds]
        else:
            ds = 1
            small = frame_rgb

        # MediaPipe expects RGB
        result = self._fd.process(small)  # type: ignore[union-attr]
        mask = np.zeros((h, w), dtype=bool)
        if not result.detections:
            return mask

        det = result.detections[0]
        location = det.location_data.relative_bounding_box
        # Convert relative bbox to full-res pixels
        x = int(location.xmin * (w / ds))
        y = int(location.ymin * (h / ds))
        bw = int(location.width * (w / ds))
        bh = int(location.height * (h / ds))
        # Clamp
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        bw = max(1, min(w - x, bw))
        bh = max(1, min(h - y, bh))

        # Define sub-rectangles within face box
        # Cheeks: lower half, left/right thirds; Forehead: upper quarter, middle third
        x1 = x
        x2 = x + bw
        y1 = y
        y2 = y + bh
        w_third = bw // 3
        h_quarter = bh // 4

        # Left cheek
        lc = (slice(y1 + bh // 2, y2), slice(x1, x1 + w_third))
        # Right cheek
        rc = (slice(y1 + bh // 2, y2), slice(x2 - w_third, x2))
        # Forehead (middle third)
        fh = (slice(y1, y1 + h_quarter), slice(x1 + w_third, x2 - w_third))

        mask[lc] = True
        mask[rc] = True
        mask[fh] = True
        return mask


class FaceCascadeROI:
    """OpenCV Haar-cascade based face ROI (no ML runtime beyond OpenCV)."""

    def __init__(self, downscale: int = 2) -> None:
        import cv2

        self.downscale = max(1, int(downscale))
        # Use default frontal face cascade bundled with OpenCV
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._clf = cv2.CascadeClassifier(cascade_path)

    def mask(self, frame_rgb: np.ndarray) -> np.ndarray:
        import cv2

        h, w, _ = frame_rgb.shape
        ds = self.downscale
        small = frame_rgb[::ds, ::ds]
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        faces = self._clf.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE
        )
        mask = np.zeros((h, w), dtype=bool)
        if len(faces) == 0:
            return mask
        # Pick the largest face
        x, y, bw, bh = max(faces, key=lambda r: r[2] * r[3])
        # Scale back to full-res
        x *= ds
        y *= ds
        bw *= ds
        bh *= ds
        x = int(x)
        y = int(y)
        bw = int(bw)
        bh = int(bh)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        bw = max(1, min(w - x, bw))
        bh = max(1, min(h - y, bh))

        x1, x2 = x, x + bw
        y1, y2 = y, y + bh
        w_third = bw // 3
        h_quarter = bh // 4
        lc = (slice(y1 + bh // 2, y2), slice(x1, x1 + w_third))
        rc = (slice(y1 + bh // 2, y2), slice(x2 - w_third, x2))
        fh = (slice(y1, y1 + h_quarter), slice(x1 + w_third, x2 - w_third))
        mask[lc] = True
        mask[rc] = True
        mask[fh] = True
        return mask
