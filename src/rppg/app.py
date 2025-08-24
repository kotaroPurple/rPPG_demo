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
    from .capture import Capture, CaptureConfig
    from .chrom import chrom_signal
    from .pos import pos_signal
    from .preprocess import bandpass, moving_average_normalize
    from .quality import snr_db
    from .recorder import Recorder, RecorderConfig
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
    snr_value = 0.0
    recording = False
    rec: Optional[Recorder] = None
    rec_started_wall: Optional[float] = None

    # Capture thread
    roi_detector = FaceBoxROI()

    def capture_loop() -> None:
        nonlocal latest_frame_rgb
        cap_wrap = Capture(CaptureConfig(0, width, height, fps_target))
        try:
            cap_wrap.open()
        except Exception:
            print("[ERROR] Failed to open camera")
            return
        try:
            while running:
                ts, frame_bgr = cap_wrap.read()
                # BGR -> RGB
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Build ROI mask once face detected; fallback to full frame
                try:
                    mask = roi_detector.mask(rgb)
                    # If detector returns empty mask, fallback to full frame
                    if mask is not None and not mask.any():
                        mask = None
                except Exception:
                    mask = None
                r, g, b = mean_rgb(rgb, mask=mask)
                with frame_lock:
                    latest_frame_rgb = rgb
                    R_buf.append(r)
                    G_buf.append(g)
                    B_buf.append(b)
                    T_buf.append(ts)
                # Optional recording per frame
                if recording and rec is not None:
                    try:
                        rec.write_row((ts, r, g, b, bpm_value))
                    except Exception:
                        pass
        finally:
            cap_wrap.release()

    # Processing thread
    def processing_loop() -> None:
        nonlocal bpm_value, snr_value
        while running:
            # Need enough samples for a window
            if len(T_buf) < 8:
                time.sleep(0.05)
                continue
            # Estimate sampling rate from timestamps
            t = np.array(T_buf, dtype=np.float64)
            fs = 1.0 / np.median(np.diff(t[-min(len(t), 50) :]))
            # Effective upper band limited by sampling rate
            fmax_eff = min(fmax, 0.45 * fs)
            if fmax_eff <= fmin:
                fmax_eff = fmin + 0.1
            L = max(8, int(win_sec * fs))
            if len(R_buf) < L:
                time.sleep(0.05)
                continue
            R = np.array(list(R_buf)[-L:], dtype=np.float32)
            G = np.array(list(G_buf)[-L:], dtype=np.float32)
            B = np.array(list(B_buf)[-L:], dtype=np.float32)
            # Normalize by moving average and band-pass filter
            Rn = bandpass(moving_average_normalize(R, max(1, int(0.5 * fs))), fs, fmin, fmax_eff)
            Gn = bandpass(moving_average_normalize(G, max(1, int(0.5 * fs))), fs, fmin, fmax_eff)
            Bn = bandpass(moving_average_normalize(B, max(1, int(0.5 * fs))), fs, fmin, fmax_eff)
            if algo == "POS":
                s = pos_signal(Rn, Gn, Bn)
            else:
                s = chrom_signal(Rn, Gn, Bn)
            # Spectrum and SNR
            x = (s - s.mean()) * np.hanning(s.size).astype(np.float32)
            X = np.fft.rfft(x)
            freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
            band = (freqs >= fmin) & (freqs <= fmax_eff)
            mag = np.abs(X)
            idx = int(np.argmax(mag * band))
            snr_value = snr_db(mag, idx)  # used in UI update
            bpm, _ = estimate_bpm(s, fs=fs, fmin=fmin, fmax=fmax_eff)
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
        # Ensure recorder closes with metadata
        try:
            if recording and rec is not None:
                meta = {
                    "algo": algo,
                    "window_sec": win_sec,
                    "band_hz": [fmin, fmax],
                    "camera": {"device": 0, "resolution": [width, height]},
                    "started": rec_started_wall,
                    "ended": time.time(),
                }
                rec.write_meta(meta)
                rec.close()
        except Exception:
            pass
        # Give threads time to exit
        time.sleep(0.2)
        dpg.stop_dearpygui()

    with dpg.window(tag=primary_tag, label="rPPG Demo", width=1000, height=680):
        dpg.add_text("Camera Preview")
        dpg.add_image(tex_tag)
        dpg.add_spacer(height=8)
        bpm_text = dpg.add_text("BPM: --")
        snr_text = dpg.add_text("SNR: -- dB")
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
        # Recording controls
        def on_record(sender, app_data, user_data):
            nonlocal recording, rec
            if app_data and not recording:
                # Start recording
                ts_name = time.strftime("%Y%m%d-%H%M%S")
                out_dir = RecorderConfig(out_dir=Path("runs") / ts_name)
                rec = Recorder(out_dir)
                rec.open(["ts", "R", "G", "B", "BPM"])
                rec_started_wall = time.time()
                recording = True
            elif (not app_data) and recording:
                # Stop recording
                if rec is not None:
                    try:
                        meta = {
                            "algo": algo,
                            "window_sec": win_sec,
                            "band_hz": [fmin, fmax],
                            "camera": {"device": 0, "resolution": [width, height]},
                            "started": rec_started_wall,
                            "ended": time.time(),
                        }
                        rec.write_meta(meta)
                    except Exception:
                        pass
                    rec.close()
                    rec = None
                recording = False

        from pathlib import Path

        dpg.add_checkbox(label="Record (CSV)", default_value=False, callback=on_record)

        # Spectrum plot (magnitude vs Hz)
        plot_tag = "spectrum_plot"
        series_tag = "spectrum_series"
        with dpg.plot(label="Spectrum", height=200, width=-1, tag=plot_tag):
            dpg.add_plot_axis(dpg.mvXAxis, label="Hz")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Mag")
            dpg.add_line_series([0.0, 1.0], [0.0, 0.0], parent=y_axis, tag=series_tag)

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
        # Update BPM/SNR
        dpg.set_value(bpm_text, f"BPM: {bpm_value:.1f}")
        dpg.set_value(snr_text, f"SNR: {snr_value:.1f} dB")
        # Update spectrum series if available
        # Recompute minimal spectrum from latest buffer for display purpose
        if len(T_buf) >= 8:
            t = np.array(T_buf, dtype=np.float64)
            fs = 1.0 / np.median(np.diff(t[-min(len(t), 50) :]))
            L = max(8, int(win_sec * fs))
            if len(R_buf) >= L:
                R = np.array(list(R_buf)[-L:], dtype=np.float32)
                G = np.array(list(G_buf)[-L:], dtype=np.float32)
                B = np.array(list(B_buf)[-L:], dtype=np.float32)
                Rn = bandpass(moving_average_normalize(R, max(1, int(0.5 * fs))), fs, fmin, fmax)
                Gn = bandpass(moving_average_normalize(G, max(1, int(0.5 * fs))), fs, fmin, fmax)
                Bn = bandpass(moving_average_normalize(B, max(1, int(0.5 * fs))), fs, fmin, fmax)
                s = pos_signal(Rn, Gn, Bn) if algo == "POS" else chrom_signal(Rn, Gn, Bn)
                x = (s - s.mean()) * np.hanning(s.size).astype(np.float32)
                X = np.fft.rfft(x)
                freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
                dpg.set_value(series_tag, [freqs.tolist(), np.abs(X).tolist()])

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
