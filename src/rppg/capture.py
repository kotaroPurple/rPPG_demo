"""Camera capture utilities (OpenCV-based)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Tuple


@dataclass
class CaptureConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


class Capture:
    """Thin wrapper around OpenCV VideoCapture.

    Imports cv2 lazily to avoid import-time side effects in non-camera contexts.
    """

    def __init__(self, cfg: Optional[CaptureConfig] = None) -> None:
        self.cfg = cfg or CaptureConfig()
        self._cap = None

    def open(self) -> None:
        import cv2  # local import

        self._cap = cv2.VideoCapture(self.cfg.device_index)
        if not self._cap.isOpened():  # type: ignore[union-attr]
            raise RuntimeError("Failed to open camera")
        # Set properties (best-effort)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)  # type: ignore[union-attr]
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)  # type: ignore[union-attr]
        self._cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)  # type: ignore[union-attr]

    def read(self) -> Tuple[float, "object"]:
        """Read a frame and return (timestamp, frame[BGR])."""
        if self._cap is None:
            raise RuntimeError("Capture is not opened")
        ts = perf_counter()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("Camera read failed")
        return ts, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()  # type: ignore[union-attr]
            self._cap = None

