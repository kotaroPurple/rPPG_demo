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
    from .pos import pos_signal
    from .preprocess import bandpass, moving_average_normalize
    from .roi import mean_rgb

    # Config
    width, height, fps_target = 640, 480, 30
    win_sec = 2.0

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
                r, g, b = mean_rgb(rgb)
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
            Rn = bandpass(moving_average_normalize(R, max(1, int(0.5 * fs))), fs)
            Gn = bandpass(moving_average_normalize(G, max(1, int(0.5 * fs))), fs)
            Bn = bandpass(moving_average_normalize(B, max(1, int(0.5 * fs))), fs)
            s = pos_signal(Rn, Gn, Bn)
            bpm, _ = estimate_bpm(s, fs=fs)
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

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window(primary_tag, True)
    dpg.set_exit_callback(on_close)

    # Start threads
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_proc = threading.Thread(target=processing_loop, daemon=True)
    t_cap.start()
    t_proc.start()

    # UI update loop (uses DearPyGUI callbacks/timers)
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

    # Add a timer to refresh UI at ~10 Hz
    dpg.add_timer(callback=ui_update_callback, delay=0.1, user_data=None)

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()
