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

    from .acf_bpm import estimate_bpm_acf
    from .bpm import estimate_bpm
    from .capture import Capture, CaptureConfig
    from .chrom import chrom_signal
    from .hilbert_if import estimate_bpm_if
    from .pos import pos_signal
    from .preprocess import bandpass, moving_average_normalize
    from .quality import peak_confidence, snr_db
    from .recorder import Recorder, RecorderConfig
    from .roi import FaceBoxROI, mean_rgb
    from .tracker import FreqTracker, TrackConfig

    # Config
    width, height, fps_target = 1280, 720, 30
    win_sec = 3.0
    fmin, fmax = 0.7, 2.0  # default upper bound 120 BPM for stability
    algo = "POS"  # or CHROM
    estimator = "FFT"  # FFT | ACF | Hilbert-IF | Tracker(FFT|ACF|IF)
    timeline_source = "Estimator"  # Estimator | Tracker
    # Estimator parameters
    if_smooth_sec = 0.10  # seconds for IF moving-average
    # Quality mapping parameters
    quality_mode = "SNR"  # SNR | Conf | SNRxConf
    quality_floor = 0.05
    quality_snr_scale = 15.0
    # Default camera index
    import sys as _sys
    selected_device = 0
    # Resolution selection (name -> (w,h))
    res_options = {
        "640x480": (640, 480),
        "960x540": (960, 540),
        "1280x720": (1280, 720),
        "1920x1080": (1920, 1080),
    }
    selected_res_name = "1280x720"

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
    conf_value = 0.0
    # Waveform buffer (latest processed window) and BPM timeline
    wave_lock = threading.Lock()
    wave_t_ds: Optional[list[float]] = None
    wave_y_ds: Optional[list[float]] = None
    bpm_hist_lock = threading.Lock()
    est_bpm_hist_t: Deque[float] = deque(maxlen=240)
    est_bpm_hist_y: Deque[float] = deque(maxlen=240)
    trk_bpm_hist_t: Deque[float] = deque(maxlen=240)
    trk_bpm_hist_y: Deque[float] = deque(maxlen=240)
    # Shared spectrum data (processing thread produces, UI consumes)
    spec_lock = threading.Lock()
    spec_freqs_ds: Optional[list[float]] = None
    spec_mag_ds: Optional[list[float]] = None
    recording = False
    rec: Optional[Recorder] = None
    rec_started_wall: Optional[float] = None
    # Tracker (optional)
    tracker = FreqTracker(TrackConfig())
    last_t_for_tracker: Optional[float] = None

    # ROI status (for UI)
    roi_mode_used = "Full"
    roi_face_found = False
    roi_bbox: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h) in full-res

    # Capture thread
    # ROI detectors (lazy imported in their modules)
    roi_mediapipe = FaceBoxROI()
    from .roi import FaceCascadeROI
    roi_cascade = FaceCascadeROI()
    use_face_roi_cv = False  # OpenCV Haar-based ROI
    use_face_roi_mp = False  # MediaPipe ROI

    def capture_loop() -> None:
        nonlocal latest_frame_rgb, width, height
        nonlocal use_face_roi_cv, use_face_roi_mp
        nonlocal roi_mode_used, roi_face_found, roi_bbox
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
                # Reopen capture if device or resolution changed or not opened
                desired_res = res_options.get(selected_res_name, (width, height))
                if (
                    (current_dev != selected_device)
                    or (cap_wrap is None)
                    or (desired_res != (width, height))
                ):
                    # Close previous
                    try:
                        if cap_wrap is not None:
                            cap_wrap.release()
                    except Exception:
                        pass
                    # Open new device
                    try:
                        # Update capture resolution to desired
                        width, height = desired_res
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
                # Compute bbox for overlay if mask is available
                if mask is not None and mask.any():
                    ys, xs = np.where(mask)
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    roi_bbox = (x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1))
                else:
                    roi_bbox = None

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
        nonlocal bpm_value, snr_value, conf_value
        nonlocal spec_freqs_ds, spec_mag_ds, wave_t_ds, wave_y_ds
        nonlocal last_t_for_tracker
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
            conf_value = peak_confidence(mag, idx)
            est_bpm_current: Optional[float] = None
            trk_bpm_current: Optional[float] = None
            # BPM estimation by selected estimator
            if estimator == "FFT":
                bpm, _ = estimate_bpm(s, fs=fs, fmin=fmin, fmax=fmax_eff)
                bpm_value = bpm
                est_bpm_current = float(bpm)
                last_t_for_tracker = float(time.time())
            elif estimator == "ACF":
                ar = estimate_bpm_acf(s, fs=fs, bpm_min=60.0 * fmin, bpm_max=60.0 * fmax_eff)
                if ar.bpm is not None:
                    bpm_value = float(ar.bpm)
                    est_bpm_current = float(ar.bpm)
                last_t_for_tracker = float(time.time())
            elif estimator == "Hilbert-IF":
                ir = estimate_bpm_if(s, fs=fs, smooth_len=max(3, int(if_smooth_sec * fs)))
                if ir.bpm is not None:
                    bpm_value = float(ir.bpm)
                    est_bpm_current = float(ir.bpm)
                last_t_for_tracker = float(time.time())
            else:
                # Tracker modes use sub-estimator as measurement
                now = float(time.time())
                if last_t_for_tracker is None:
                    last_t_for_tracker = now
                dt = max(1e-3, now - last_t_for_tracker)
                last_t_for_tracker = now
                tracker.predict(dt)
                meas_bpm: Optional[float] = None
                if estimator == "Tracker(FFT)":
                    m, _ = estimate_bpm(s, fs=fs, fmin=fmin, fmax=fmax_eff)
                    meas_bpm = m
                    est_bpm_current = float(m)
                elif estimator == "Tracker(ACF)":
                    ar = estimate_bpm_acf(s, fs=fs, bpm_min=60.0 * fmin, bpm_max=60.0 * fmax_eff)
                    meas_bpm = ar.bpm
                    if ar.bpm is not None:
                        est_bpm_current = float(ar.bpm)
                elif estimator == "Tracker(IF)":
                    ir = estimate_bpm_if(s, fs=fs, smooth_len=max(3, int(if_smooth_sec * fs)))
                    meas_bpm = ir.bpm
                    if ir.bpm is not None:
                        est_bpm_current = float(ir.bpm)
                # Map SNR/Conf to quality [0..1]
                if quality_mode == "SNR":
                    qual = float(np.clip(snr_value / quality_snr_scale, quality_floor, 1.0))
                elif quality_mode == "Conf":
                    qual = float(np.clip(conf_value, quality_floor, 1.0))
                else:
                    qual = float(
                        np.clip((snr_value / quality_snr_scale) * conf_value, quality_floor, 1.0)
                    )
                if meas_bpm is not None:
                    tracker.update(float(meas_bpm) / 60.0, quality=qual)
                tb = tracker.value_bpm()
                if tb is not None:
                    bpm_value = tb
                    trk_bpm_current = tb
            # Update BPM histories for timeline
            with bpm_hist_lock:
                nowt = float(time.time())
                if est_bpm_current is not None:
                    est_bpm_hist_t.append(nowt)
                    est_bpm_hist_y.append(est_bpm_current)
                if trk_bpm_current is not None:
                    trk_bpm_hist_t.append(nowt)
                    trk_bpm_hist_y.append(trk_bpm_current)
            # Prepare waveform downsample for UI (time in seconds, centered at 0)
            if s.size > 0:
                t_rel = (np.arange(s.size, dtype=np.float32) - (s.size - 1)) / float(fs)
                # Downsample to at most 256 points for UI
                max_points = 256
                if s.size > max_points:
                    idx_ds = np.linspace(0, s.size - 1, max_points).astype(int)
                    t_ds = t_rel[idx_ds].tolist()
                    y_ds = s[idx_ds].tolist()
                else:
                    t_ds = t_rel.tolist()
                    y_ds = s.tolist()
                with wave_lock:
                    wave_t_ds = t_ds
                    wave_y_ds = y_ds
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
    dpg.create_viewport(title="rPPG Demo", width=1280, height=950)

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
            with dpg.child_window(width=900, height=860):
                dpg.add_text("Camera Preview")
                # Display scaled-up size regardless of internal texture size
                # Fit without scrolling: slightly smaller than full width, 16:9
                dpg.add_image(tex_tag, width=900, height=540)
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    bpm_text = dpg.add_text("BPM: --")
                    dpg.add_spacer(width=12)
                    snr_text = dpg.add_text("SNR: -- dB")
                    dpg.add_spacer(width=12)
                    conf_text = dpg.add_text("Conf: --")
                    dpg.add_spacer(width=12)
                    status_text = dpg.add_text("Status: idle")
                # Plots area without scrolling: place two plots side-by-side
                with dpg.group(horizontal=True):
                    # rPPG waveform (pre-BPM signal)
                    wave_plot_tag = "wave_plot"
                    wave_series_tag = "wave_series"
                    wave_y_axis_tag = "wave_y_axis"
                    wave_x_axis_tag = "wave_x_axis"
                    with dpg.plot(
                        label="rPPG Waveform (sec)", height=170, width=430, tag=wave_plot_tag
                    ):
                        dpg.add_plot_axis(dpg.mvXAxis, label="t (s)", tag=wave_x_axis_tag)
                        y_axis_w = dpg.add_plot_axis(dpg.mvYAxis, label="amp", tag=wave_y_axis_tag)
                        dpg.add_line_series(
                            [0.0, 1.0], [0.0, 0.0], parent=y_axis_w, tag=wave_series_tag
                        )
                        try:
                            dpg.set_axis_limits(wave_y_axis_tag, -0.005, 0.005)
                            dpg.set_axis_limits(wave_x_axis_tag, -win_sec, 0.0)
                        except Exception:
                            pass
                    # BPM timeline plot
                    bpm_plot_tag = "bpm_plot"
                    bpm_series_tag = "bpm_series"
                    bpm_y_axis_tag = "bpm_y_axis"
                    bpm_x_axis_tag = "bpm_x_axis"
                    with dpg.plot(label="BPM Timeline", height=170, width=430, tag=bpm_plot_tag):
                        dpg.add_plot_axis(dpg.mvXAxis, label="time", tag=bpm_x_axis_tag)
                        y_axis_bpm = dpg.add_plot_axis(dpg.mvYAxis, label="BPM", tag=bpm_y_axis_tag)
                        dpg.add_line_series(
                            [0.0, 1.0], [0.0, 0.0], parent=y_axis_bpm, tag=bpm_series_tag
                        )
                        try:
                            dpg.set_axis_limits(bpm_y_axis_tag, 40.0, 120.0)
                            dpg.set_axis_limits(bpm_x_axis_tag, 0.0, 90.0)
                        except Exception:
                            pass
                # Spectrum plot (magnitude vs Hz) — optional
                plot_tag = "spectrum_plot"
                series_line_tag = "spectrum_line"
                series_bar_tag = "spectrum_bar"
                with dpg.plot(label="Spectrum", height=180, width=-1, tag=plot_tag):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Hz")
                    y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Mag")
                    dpg.add_line_series([0.0, 1.0], [0.0, 0.0], parent=y_axis, tag=series_line_tag)
                    dpg.add_bar_series([0.0, 1.0], [0.0, 0.0], parent=y_axis, tag=series_bar_tag)
                    dpg.configure_item(series_line_tag, show=False)
                    dpg.configure_item(series_bar_tag, show=False)
            # Right panel: Controls
            controls_panel_tag = "controls_panel"
            with dpg.child_window(width=340, height=860, tag=controls_panel_tag):
                dpg.add_text("Controls")
        # Controls
        def on_algo(sender, app_data, user_data):
            nonlocal algo
            algo = app_data

        def on_win(sender, app_data, user_data):
            nonlocal win_sec
            win_sec = float(app_data)
            # Update waveform X-axis range to match fixed window
            try:
                dpg.set_axis_limits("wave_x_axis", -win_sec, 0.0)
            except Exception:
                pass

        def on_band_min(sender, app_data, user_data):
            nonlocal fmin
            fmin = float(app_data)

        def on_band_max(sender, app_data, user_data):
            nonlocal fmax
            fmax = float(app_data)

        dpg.add_combo(("POS", "CHROM"), default_value=algo, label="Algorithm",
                      callback=on_algo, parent=controls_panel_tag)
        dpg.add_slider_float(label="Window (s)", default_value=win_sec,
                             min_value=1.0, max_value=5.0, callback=on_win,
                             parent=controls_panel_tag)
        dpg.add_slider_float(label="Band min (Hz)", default_value=fmin,
                             min_value=0.2, max_value=2.0, callback=on_band_min,
                             parent=controls_panel_tag)
        dpg.add_slider_float(label="Band max (Hz)", default_value=fmax,
                             min_value=1.5, max_value=5.0, callback=on_band_max,
                             parent=controls_panel_tag)
        # Estimator selection
        def on_estimator(sender, app_data, user_data):
            nonlocal estimator
            estimator = str(app_data)
        dpg.add_combo(
            ("FFT", "ACF", "Hilbert-IF", "Tracker(FFT)", "Tracker(ACF)", "Tracker(IF)"),
            default_value=estimator,
            label="Estimator",
            callback=on_estimator,
            parent=controls_panel_tag,
        )
        # Timeline source selection
        def on_timeline_src(sender, app_data, user_data):
            nonlocal timeline_source
            timeline_source = str(app_data)
        dpg.add_combo(
            ("Estimator", "Tracker"),
            default_value=timeline_source,
            label="Timeline Source",
            callback=on_timeline_src,
            parent=controls_panel_tag,
        )
        # Estimator parameters
        def on_if_smooth(sender, app_data, user_data):
            nonlocal if_smooth_sec
            if_smooth_sec = float(app_data)
        dpg.add_slider_float(
            label="IF smooth (s)",
            default_value=if_smooth_sec,
            min_value=0.02,
            max_value=0.50,
            format="%.02f",
            callback=on_if_smooth,
            parent=controls_panel_tag,
        )
        # Quality source selection
        def on_quality_mode(sender, app_data, user_data):
            nonlocal quality_mode
            quality_mode = str(app_data)
        dpg.add_combo(
            ("SNR", "Conf", "SNRxConf"),
            default_value=quality_mode,
            label="Quality Source",
            callback=on_quality_mode,
            parent=controls_panel_tag,
        )
        # Tracker parameters (Q/R)
        def on_tracker_qf(sender, app_data, user_data):
            tracker.cfg.q_freq = float(app_data)
        dpg.add_slider_float(
            label="Tracker q_freq",
            default_value=float(tracker.cfg.q_freq),
            min_value=0.001,
            max_value=0.200,
            format="%.3f",
            callback=on_tracker_qf,
            parent=controls_panel_tag,
        )
        def on_tracker_qd(sender, app_data, user_data):
            tracker.cfg.q_drift = float(app_data)
        dpg.add_slider_float(
            label="Tracker q_drift",
            default_value=float(tracker.cfg.q_drift),
            min_value=0.001,
            max_value=0.200,
            format="%.3f",
            callback=on_tracker_qd,
            parent=controls_panel_tag,
        )
        def on_tracker_r(sender, app_data, user_data):
            tracker.cfg.r_meas = float(app_data)
        dpg.add_slider_float(
            label="Tracker r_meas",
            default_value=float(tracker.cfg.r_meas),
            min_value=0.01,
            max_value=1.00,
            format="%.2f",
            callback=on_tracker_r,
            parent=controls_panel_tag,
        )
        def on_quality_floor(sender, app_data, user_data):
            nonlocal quality_floor
            quality_floor = float(app_data)
        dpg.add_slider_float(
            label="Quality floor",
            default_value=float(quality_floor),
            min_value=0.00,
            max_value=0.50,
            format="%.2f",
            callback=on_quality_floor,
            parent=controls_panel_tag,
        )
        def on_quality_scale(sender, app_data, user_data):
            nonlocal quality_snr_scale
            quality_snr_scale = float(app_data)
        dpg.add_slider_float(
            label="Quality SNR scale",
            default_value=float(quality_snr_scale),
            min_value=5.0,
            max_value=30.0,
            format="%.1f",
            callback=on_quality_scale,
            parent=controls_panel_tag,
        )
        # Estimator selection
        def on_estimator(sender, app_data, user_data):
            nonlocal estimator
            estimator = str(app_data)
        dpg.add_combo(
            ("FFT", "ACF", "Hilbert-IF", "Tracker(FFT)", "Tracker(ACF)", "Tracker(IF)"),
            default_value=estimator,
            label="Estimator",
            callback=on_estimator,
            parent=controls_panel_tag,
        )
        # Resolution selection
        def on_res(sender, app_data, user_data):
            nonlocal selected_res_name
            selected_res_name = app_data
        try:
            dpg.add_combo(
                list(res_options.keys()),
                default_value=selected_res_name,
                label="Resolution",
                callback=on_res,
                parent=controls_panel_tag,
            )
        except Exception:
            pass
        def on_roi_cv(sender, app_data, user_data):
            nonlocal use_face_roi_cv
            use_face_roi_cv = bool(app_data)
        dpg.add_checkbox(label="Use Face ROI (OpenCV)", default_value=False, callback=on_roi_cv,
                         parent=controls_panel_tag)

        def on_roi_mp(sender, app_data, user_data):
            nonlocal use_face_roi_mp
            use_face_roi_mp = bool(app_data)
        dpg.add_checkbox(label="Use Face ROI (MediaPipe)", default_value=False, callback=on_roi_mp,
                         parent=controls_panel_tag)

        # Preview and spectrum toggles
        preview_enabled = True
        spectrum_enabled = True

        def on_preview(sender, app_data, user_data):
            nonlocal preview_enabled
            preview_enabled = bool(app_data)

        def on_spectrum(sender, app_data, user_data):
            nonlocal spectrum_enabled
            spectrum_enabled = bool(app_data)

        dpg.add_checkbox(label="Preview", default_value=True, callback=on_preview,
                         parent=controls_panel_tag)
        dpg.add_checkbox(label="Spectrum", default_value=False, callback=on_spectrum,
                         parent=controls_panel_tag)

        # ROI status indicator
        roi_status_text = dpg.add_text("ROI: Full | Face: --", parent=controls_panel_tag)

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
                      callback=on_camera, parent=controls_panel_tag)

        # Connect/Disconnect
        def on_connect(sender, app_data, user_data):
            nonlocal connected
            connected = bool(app_data)

        dpg.add_checkbox(label="Connect", default_value=False, callback=on_connect,
                         parent=controls_panel_tag)
        # Recording controls
        record_base_dir = Path("runs")
        dir_label = dpg.add_text(f"Output Dir: {record_base_dir}", parent=controls_panel_tag)

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

        dpg.add_button(label="Choose Output Dir",
                       callback=lambda: dpg.show_item("dir_dialog"),
                       parent=controls_panel_tag)

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

        dpg.add_checkbox(label="Record (CSV)", default_value=False, callback=on_record,
                         parent=controls_panel_tag)

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
            # Draw face bbox overlay if available (green rectangle)
            if roi_bbox is not None:
                x, y, w, h = roi_bbox
                sx = int(x * tex_w / width)
                sy = int(y * tex_h / height)
                ex = int((x + w) * tex_w / width)
                ey = int((y + h) * tex_h / height)
                try:
                    cv2.rectangle(frame_small, (sx, sy), (ex, ey), (0, 255, 0), 2)
                except Exception:
                    pass
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
        dpg.set_value(conf_text, f"Conf: {conf_value:.2f}")
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
        # Update rPPG waveform plot (lightweight)
        try:
            with wave_lock:
                if wave_t_ds and wave_y_ds:
                    dpg.set_value(wave_series_tag, [wave_t_ds, wave_y_ds])
        except Exception:
            pass
        # Update BPM timeline (throttle to ~2 Hz)
        if not hasattr(ui_update_callback, "_bpm_counter"):
            ui_update_callback._bpm_counter = 0  # type: ignore[attr-defined]
        ui_update_callback._bpm_counter += 1  # type: ignore[attr-defined]
        if ui_update_callback._bpm_counter % 5 == 0:  # type: ignore[attr-defined]
            try:
                with bpm_hist_lock:
                    # Choose source for timeline
                    if 'timeline_source' not in ui_update_callback.__dict__:
                        pass
                    if 'Tracker' == 'Tracker' and timeline_source == 'Tracker':
                        t_src = trk_bpm_hist_t
                        y_src = trk_bpm_hist_y
                    else:
                        t_src = est_bpm_hist_t
                        y_src = est_bpm_hist_y
                    if len(t_src) >= 2:
                        # Show last 90 seconds; convert to relative seconds
                        t0 = t_src[-1] - 90.0
                        xs = [ti - t0 for ti in t_src if ti >= t0]
                        ys = [yv for ti, yv in zip(t_src, y_src, strict=False) if ti >= t0]
                        if len(xs) >= 2:
                            dpg.set_value(bpm_series_tag, [xs, ys])
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
