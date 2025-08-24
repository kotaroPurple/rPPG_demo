"""DearPyGUI entry point with minimal camera preview and BPM display.

Run with: `uv run task run`
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional


def main() -> None:
    """Launch a minimal DearPyGUI window with camera preview and BPM display."""
    # Import locally to avoid hard dependency at import time
    import cv2
    import dearpygui.dearpygui as dpg
    import numpy as np

    from .bpm import estimate_bpm
    from .chrom import chrom_signal
    from .pos import pos_signal
    from .preprocess import bandpass, moving_average_normalize
    from .roi import FaceBoxROI, mean_rgb

    # Config
    width, height, fps_target = 640, 480, 30
    win_sec = 2.0
    fmin, fmax = 0.7, 4.0
    algo = "POS"  # or CHROM

    # State
    running = True
    frame_lock = threading.Lock()
    latest_frame_rgb: Optional[np.ndarray] = None
    # Buffers for mean RGB and timestamps
    R_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    G_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    B_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    T_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    bpm_value = 0.0

    # Capture thread
    roi_detector = FaceBoxROI()

    def capture_loop() -> None:
        nonlocal latest_frame_rgb
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps_target)
        if not cap.isOpened():
            print("[ERROR] Failed to open camera")
            return
        try:
            while running:
                ts = time.perf_counter()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                # BGR -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Build ROI mask once face detected; fallback to full frame
                try:
                    mask = roi_detector.mask(rgb)
                except Exception:
                    mask = None
                r, g, b = mean_rgb(rgb, mask=mask)
                with frame_lock:
                    latest_frame_rgb = rgb
                    R_buf.append(r)
                    G_buf.append(g)
                    B_buf.append(b)
                    T_buf.append(ts)
        finally:
            cap.release()

    # Processing thread
    def processing_loop() -> None:
        nonlocal bpm_value
        while running:
            # Need enough samples for a window
            if len(T_buf) < 8:
                time.sleep(0.05)
                continue
            # Estimate sampling rate from timestamps
            t = np.array(T_buf, dtype=np.float64)
            fs = 1.0 / np.median(np.diff(t[-min(len(t), 50) :]))
            L = max(8, int(win_sec * fs))
            if len(R_buf) < L:
                time.sleep(0.05)
                continue
            R = np.array(list(R_buf)[-L:], dtype=np.float32)
            G = np.array(list(G_buf)[-L:], dtype=np.float32)
            B = np.array(list(B_buf)[-L:], dtype=np.float32)
            # Normalize by moving average and band-pass filter
            Rn = bandpass(moving_average_normalize(R, max(1, int(0.5 * fs))), fs, fmin, fmax)
            Gn = bandpass(moving_average_normalize(G, max(1, int(0.5 * fs))), fs, fmin, fmax)
            Bn = bandpass(moving_average_normalize(B, max(1, int(0.5 * fs))), fs, fmin, fmax)
            if algo == "POS":
                s = pos_signal(Rn, Gn, Bn)
            else:
                s = chrom_signal(Rn, Gn, Bn)
            bpm, _ = estimate_bpm(s, fs=fs, fmin=fmin, fmax=fmax)
            bpm_value = bpm
            time.sleep(0.1)

    # UI setup
    dpg.create_context()
    dpg.create_viewport(title="rPPG Demo", width=1024, height=720)

    # Texture for preview (RGBA float)
    tex_tag = "preview_tex"
    primary_tag = "primary_window"
    with dpg.texture_registry():
        empty = np.zeros((height, width, 4), dtype=np.float32).ravel()
        dpg.add_dynamic_texture(width, height, empty, tag=tex_tag)

    def on_close() -> None:
        nonlocal running
        running = False
        # Give threads time to exit
        time.sleep(0.2)
        dpg.stop_dearpygui()

    with dpg.window(tag=primary_tag, label="rPPG Demo", width=1000, height=680):
        dpg.add_text("Camera Preview")
        dpg.add_image(tex_tag)
        dpg.add_spacer(height=8)
        bpm_text = dpg.add_text("BPM: --")
        # Controls
        def on_algo(sender, app_data, user_data):
            nonlocal algo
            algo = app_data

        def on_win(sender, app_data, user_data):
            nonlocal win_sec
            win_sec = float(app_data)

        def on_band_min(sender, app_data, user_data):
            nonlocal fmin
            fmin = float(app_data)

        def on_band_max(sender, app_data, user_data):
            nonlocal fmax
            fmax = float(app_data)

        dpg.add_combo(("POS", "CHROM"), default_value=algo, label="Algorithm",
                      callback=on_algo)
        dpg.add_slider_float(label="Window (s)", default_value=win_sec,
                             min_value=1.0, max_value=5.0, callback=on_win)
        dpg.add_slider_float(label="Band min (Hz)", default_value=fmin,
                             min_value=0.2, max_value=2.0, callback=on_band_min)
        dpg.add_slider_float(label="Band max (Hz)", default_value=fmax,
                             min_value=2.5, max_value=5.0, callback=on_band_max)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(primary_tag, True)
    dpg.set_exit_callback(on_close)

    # Start threads
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=processing_loop, daemon=True)
    t_cap.start()
    t_proc.start()

    # UI update loop (uses frame callback scheduling)
    def ui_update_callback() -> None:
        # Update preview
        with frame_lock:
            frame = latest_frame_rgb.copy() if latest_frame_rgb is not None else None
        if frame is not None:
            # Ensure size matches texture; resize if camera differs
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            rgba = np.concatenate(
                [frame.astype(np.float32) / 255.0, np.ones((height, width, 1), dtype=np.float32)],
                axis=2,
            ).ravel()
            dpg.set_value(tex_tag, rgba)
        # Update BPM
        dpg.set_value(bpm_text, f"BPM: {bpm_value:.1f}")

    # Schedule periodic UI updates (~10 Hz) using frame callbacks
    def schedule_ui_updates(interval_frames: int = 6) -> None:
        def _tick() -> None:
            ui_update_callback()
            if running:
                dpg.set_frame_callback(dpg.get_frame_count() + interval_frames, _tick)

        dpg.set_frame_callback(dpg.get_frame_count() + interval_frames, _tick)

    schedule_ui_updates()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()
