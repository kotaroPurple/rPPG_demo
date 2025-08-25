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
    from pathlib import Path

    import cv2
    import dearpygui.dearpygui as dpg
    import numpy as np

    # Initialize logging and fault handler
    logs_dir = Path("logs")
    try:
        logs_dir.mkdir(exist_ok=True)
    except Exception:
        pass
    try:
        import faulthandler
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(logs_dir / "app.log", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        fh = (logs_dir / "faulthandler.log").open("w")
        faulthandler.enable(fh)
    except Exception:
        pass

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
    # Default camera index
    import sys as _sys
    selected_device = 0

    # State
    running = True
    connected = False
    frame_lock = threading.Lock()
    latest_frame_rgb: Optional[np.ndarray] = None
    # Buffers for mean RGB and timestamps
    R_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    G_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    B_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    T_buf: Deque[float] = deque(maxlen=int(fps_target * 10))
    bpm_value = 0.0
    snr_value = 0.0
    # Shared spectrum data (processing thread produces, UI consumes)
    spec_lock = threading.Lock()
    spec_freqs_ds: Optional[list[float]] = None
    spec_mag_ds: Optional[list[float]] = None
    recording = False
    rec: Optional[Recorder] = None
    rec_started_wall: Optional[float] = None

    # ROI status (for UI)
    roi_mode_used = "Full"
    roi_face_found = False

    # Capture thread
    # ROI detectors (lazy imported in their modules)
    roi_mediapipe = FaceBoxROI()
    from .roi import FaceCascadeROI
    roi_cascade = FaceCascadeROI()
    use_face_roi_cv = False  # OpenCV Haar-based ROI
    use_face_roi_mp = False  # MediaPipe ROI

    def capture_loop() -> None:
        nonlocal latest_frame_rgb, use_face_roi_cv, use_face_roi_mp, roi_mode_used, roi_face_found
        cap_wrap: Optional[Capture] = None
        current_dev: Optional[int] = None
        try:
            while running:
                if not connected:
                    # Ensure closed when not connected
                    try:
                        if cap_wrap is not None:
                            cap_wrap.release()
                    except Exception:
                        pass
                    cap_wrap = None
                    current_dev = None
                    time.sleep(0.1)
                    continue
                # Reopen capture if device changed or not opened
                if current_dev != selected_device or cap_wrap is None:
                    # Close previous
                    try:
                        if cap_wrap is not None:
                            cap_wrap.release()
                    except Exception:
                        pass
                    # Open new device
                    try:
                        cfg = CaptureConfig(selected_device, width, height, fps_target)
                        cap_wrap = Capture(cfg)
                        cap_wrap.open()
                        current_dev = selected_device
                    except Exception:
                        # Could not open device, wait and retry
                        time.sleep(0.5)
                        continue
                # Read frame
                try:
                    ts, frame_bgr = cap_wrap.read()
                except Exception:
                    # Read failure; force reopen next loop
                    current_dev = None
                    time.sleep(0.05)
                    continue
                # BGR -> RGB
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Build ROI mask once face detected; fallback to full frame
                mask = None
                # Prefer lightweight OpenCV ROI if enabled
                if use_face_roi_cv:
                    try:
                        mask = roi_cascade.mask(rgb)
                        if mask is not None and not mask.any():
                            mask = None
                    except Exception:
                        use_face_roi_cv = False
                elif use_face_roi_mp:
                    try:
                        mask = roi_mediapipe.mask(rgb)
                        if mask is not None and not mask.any():
                            mask = None
                    except Exception:
                        use_face_roi_mp = False
                        # log silently; mediapipe may fail on some setups
                        try:
                            import logging as _logging

                            _logging.exception("MediaPipe ROI failed")
                        except Exception:
                            pass
                # Update ROI status for UI
                if use_face_roi_cv:
                    roi_mode_used = "OpenCV"
                elif use_face_roi_mp:
                    roi_mode_used = "MediaPipe"
                else:
                    roi_mode_used = "Full"
                roi_face_found = bool(mask is not None)

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
            try:
                if cap_wrap is not None:
                    cap_wrap.release()
            except Exception:
                pass

    # Processing thread
    def processing_loop() -> None:
        nonlocal bpm_value, snr_value, spec_freqs_ds, spec_mag_ds
        while running:
            if not connected:
                time.sleep(0.1)
                continue
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
            # Write lightweight metrics snapshot (for external service/debug)
            try:
                import json as _json
                from pathlib import Path as _Path

                md = {
                    "t": float(time.time()),
                    "bpm": float(bpm_value),
                    "snr": float(snr_value),
                    "fs": float(fs),
                }
                p = _Path("logs")
                try:
                    p.mkdir(exist_ok=True)
                except Exception:
                    pass
                (_Path("logs") / "current_metrics.json").write_text(_json.dumps(md))
            except Exception:
                pass
            # Prepare downsampled spectrum for UI (avoid heavy work in UI thread)
            max_points = 256
            if mag.size > 0 and freqs.size == mag.size:
                mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
                if freqs.size > max_points:
                    idx_ds = np.linspace(0, freqs.size - 1, max_points).astype(int)
                    f_ds = freqs[idx_ds].tolist()
                    m_ds = mag[idx_ds].tolist()
                else:
                    f_ds = freqs.tolist()
                    m_ds = mag.tolist()
                with spec_lock:
                    spec_freqs_ds = f_ds
                    spec_mag_ds = m_ds
            time.sleep(0.1)

    # UI setup
    dpg.create_context()
    dpg.create_viewport(title="rPPG Demo", width=1280, height=900)

    # Texture for preview (RGBA float)
    tex_tag = "preview_tex"
    primary_tag = "primary_window"
    # Use smaller internal texture to reduce update load
    tex_w, tex_h = 320, 240
    with dpg.texture_registry():
        tex_buf = np.zeros((tex_h, tex_w, 4), dtype=np.float32)
        dpg.add_dynamic_texture(tex_w, tex_h, tex_buf.ravel(), tag=tex_tag)

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

    with dpg.window(tag=primary_tag, label="rPPG Demo", width=1260, height=860):
        with dpg.group(horizontal=True):
            # Left panel: Preview + Spectrum
            with dpg.child_window(width=900, height=820):
                dpg.add_text("Camera Preview")
                # Display scaled-up size regardless of internal texture size
                dpg.add_image(tex_tag, width=900, height=675)
                dpg.add_spacer(height=8)
                bpm_text = dpg.add_text("BPM: --")
                snr_text = dpg.add_text("SNR: -- dB")
                status_text = dpg.add_text("Status: idle")
                # Spectrum plot (magnitude vs Hz)
                plot_tag = "spectrum_plot"
                series_line_tag = "spectrum_line"
                series_bar_tag = "spectrum_bar"
                with dpg.plot(label="Spectrum", height=220, width=-1, tag=plot_tag):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Hz")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Mag")
                    dpg.add_line_series([0.0, 1.0], [0.0, 0.0], parent=y_axis, tag=series_line_tag)
                    dpg.add_bar_series([0.0, 1.0], [0.0, 0.0], parent=y_axis, tag=series_bar_tag)
                    dpg.configure_item(series_line_tag, show=False)
                    dpg.configure_item(series_bar_tag, show=False)
            # Right panel: Controls
            with dpg.child_window(width=340, height=820):
                dpg.add_text("Controls")
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
        def on_roi_cv(sender, app_data, user_data):
            nonlocal use_face_roi_cv
            use_face_roi_cv = bool(app_data)
        dpg.add_checkbox(label="Use Face ROI (OpenCV)", default_value=False, callback=on_roi_cv)

        def on_roi_mp(sender, app_data, user_data):
            nonlocal use_face_roi_mp
            use_face_roi_mp = bool(app_data)
        dpg.add_checkbox(label="Use Face ROI (MediaPipe)", default_value=False, callback=on_roi_mp)

        # Preview and spectrum toggles
        preview_enabled = True
        spectrum_enabled = True

        def on_preview(sender, app_data, user_data):
            nonlocal preview_enabled
            preview_enabled = bool(app_data)

        def on_spectrum(sender, app_data, user_data):
            nonlocal spectrum_enabled
            spectrum_enabled = bool(app_data)

        dpg.add_checkbox(label="Preview", default_value=True, callback=on_preview)
        dpg.add_checkbox(label="Spectrum", default_value=False, callback=on_spectrum)

        # ROI status indicator
        roi_status_text = dpg.add_text("ROI: Full | Face: --")

        # Camera selection
        # Avoid probing devices to prevent triggering Continuity Camera side-effects.
        camera_options = ["0", "1"] if _sys.platform == "darwin" else [str(i) for i in range(3)]

        def on_camera(sender, app_data, user_data):
            nonlocal selected_device
            try:
                selected_device = int(app_data)
            except Exception:
                pass

        dpg.add_combo(camera_options, default_value=str(selected_device), label="Camera",
                      callback=on_camera)

        # Connect/Disconnect
        def on_connect(sender, app_data, user_data):
            nonlocal connected
            connected = bool(app_data)

        dpg.add_checkbox(label="Connect", default_value=False, callback=on_connect)
        # Recording controls
        record_base_dir = Path("runs")
        dir_label = dpg.add_text(f"Output Dir: {record_base_dir}")

        def on_choose_dir(sender, app_data):
            nonlocal record_base_dir
            try:
                p = Path(app_data.get("file_path_name", ""))
                if p.exists():
                    record_base_dir = p
                    dpg.set_value(dir_label, f"Output Dir: {record_base_dir}")
            except Exception:
                pass

        with dpg.file_dialog(
            directory_selector=True,
            show=False,
            callback=on_choose_dir,
            tag="dir_dialog",
        ):
            dpg.add_file_extension("")

        dpg.add_button(label="Choose Output Dir", callback=lambda: dpg.show_item("dir_dialog"))

        def on_record(sender, app_data, user_data):
            nonlocal recording, rec
            if app_data and not recording:
                # Start recording
                ts_name = time.strftime("%Y%m%d-%H%M%S")
                out_dir = RecorderConfig(out_dir=record_base_dir / ts_name)
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
        if frame is not None and preview_enabled:
            # Ensure size matches texture; resize if camera differs
            # Throttle preview updates (~30 FPS)
            if not hasattr(ui_update_callback, "_prev_counter"):
                ui_update_callback._prev_counter = 0  # type: ignore[attr-defined]
            ui_update_callback._prev_counter += 1  # type: ignore[attr-defined]
            if frame.shape[0] != tex_h or frame.shape[1] != tex_w:
                frame_small = cv2.resize(frame, (tex_w, tex_h))
            else:
                frame_small = frame
            try:
                if ui_update_callback._prev_counter % 2 == 0:  # type: ignore[attr-defined]
                    # Write into persistent buffer to avoid realloc
                    tex_buf[:, :, :3] = frame_small.astype(np.float32) / 255.0
                    tex_buf[:, :, 3] = 1.0
                    dpg.set_value(tex_tag, tex_buf.ravel())
            except Exception:
                pass
        # Update BPM/SNR
        dpg.set_value(bpm_text, f"BPM: {bpm_value:.1f}")
        dpg.set_value(snr_text, f"SNR: {snr_value:.1f} dB")
        # Update ROI status line
        status_roi = f"ROI: {roi_mode_used} | Face: {'Y' if roi_face_found else 'N'}"
        try:
            dpg.set_value(roi_status_text, status_roi)
        except Exception:
            pass
        # Compute a rough fs estimate for status line
        if len(T_buf) > 1:
            t_np = np.array(T_buf)
            dt = np.diff(t_np[-min(len(t_np), 50) :])
            fs_est = float(1.0 / np.median(dt)) if dt.size > 0 else 0.0
        else:
            fs_est = 0.0
        conn = "on" if connected else "off"
        dpg.set_value(status_text, f"Status: conn={conn} dev={selected_device} fs~{fs_est:.1f}Hz")
        # Update spectrum series if available (throttled and downsampled)
        # Recompute spectrum from latest buffer for display purpose
        if spectrum_enabled and len(T_buf) >= 8:
            # Throttle updates to reduce GL load
            if not hasattr(ui_update_callback, "_spec_counter"):
                ui_update_callback._spec_counter = 0  # type: ignore[attr-defined]
            ui_update_callback._spec_counter += 1  # type: ignore[attr-defined]
            spectrum_interval_frames = 60  # ~1 Hz if frame callback ~60 FPS
            if ui_update_callback._spec_counter % spectrum_interval_frames == 0:  # type: ignore[attr-defined]
                t = np.array(T_buf, dtype=np.float64)
                fs = 1.0 / np.median(np.diff(t[-min(len(t), 50) :]))
                L = max(8, int(win_sec * fs))
                if len(R_buf) >= L:
                    R = np.array(list(R_buf)[-L:], dtype=np.float32)
                    G = np.array(list(G_buf)[-L:], dtype=np.float32)
                    B = np.array(list(B_buf)[-L:], dtype=np.float32)
                    Rn = bandpass(
                        moving_average_normalize(R, max(1, int(0.5 * fs))), fs, fmin, fmax
                    )
                    Gn = bandpass(
                        moving_average_normalize(G, max(1, int(0.5 * fs))), fs, fmin, fmax
                    )
                    Bn = bandpass(
                        moving_average_normalize(B, max(1, int(0.5 * fs))), fs, fmin, fmax
                    )
                    s = pos_signal(Rn, Gn, Bn) if algo == "POS" else chrom_signal(Rn, Gn, Bn)
                    x = (s - s.mean()) * np.hanning(s.size).astype(np.float32)
                    X = np.fft.rfft(x)
                    freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
                    mag = np.abs(X)
                    # Sanitize and downsample to max_points
                    max_points = 256
                    if mag.size > 0 and freqs.size == mag.size:
                        # Remove non-finite
                        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
                        if freqs.size > max_points:
                            idx = np.linspace(0, freqs.size - 1, max_points).astype(int)
                            freqs_ds = freqs[idx]
                            mag_ds = mag[idx]
                        else:
                            freqs_ds = freqs
                            mag_ds = mag
                        try:
                            # Bars only（安定性優先）
                            dpg.configure_item(series_line_tag, show=False)
                            dpg.configure_item(series_bar_tag, show=True)
                            # 64本にさらに間引き
                            if freqs_ds.size > 64:
                                idx2 = np.linspace(0, freqs_ds.size - 1, 64).astype(int)
                                fx = freqs_ds[idx2]
                                my = mag_ds[idx2]
                            else:
                                fx = freqs_ds
                                my = mag_ds
                            dpg.set_value(series_bar_tag, [fx.tolist(), my.tolist()])
                        except Exception:
                            try:
                                dpg.configure_item(series_bar_tag, show=False)
                                dpg.configure_item(series_line_tag, show=False)
                            except Exception:
                                pass

    # Schedule periodic UI updates (~10 Hz) using frame callbacks
    def schedule_ui_updates(interval_frames: int = 6) -> None:
        def _tick() -> None:
            ui_update_callback()
            dpg.set_frame_callback(dpg.get_frame_count() + interval_frames, _tick)

        dpg.set_frame_callback(dpg.get_frame_count() + interval_frames, _tick)

    schedule_ui_updates()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":  # pragma: no cover - manual entry
    main()
