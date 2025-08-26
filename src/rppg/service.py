"""FastAPI service exposing rPPG metrics for a Web UI.

Keeps desktop app behavior unchanged. Reuses the same processing functions
used by the desktop UI (preprocess, CHROM/POS, FFT/ACF/IF, tracker, respiration).

Two ingestion paths are planned; this version implements a lightweight
"browser meanRGB ingestion": the browser can POST averaged RGB samples to
`/ingest`, and the service runs the same core pipeline to estimate BPM/RR.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .acf_bpm import estimate_bpm_acf
from .bpm import estimate_bpm
from .chrom import chrom_signal
from .hilbert_if import estimate_bpm_if
from .pos import pos_signal
from .preprocess import bandpass, moving_average_normalize
from .quality import peak_confidence, snr_db
from .respiration import estimate_rr_envelope
from .tracker import FreqTracker, TrackConfig


@dataclass
class Params:
    algo: str = "POS"
    est: str = "FFT"  # FFT | ACF | Hilbert-IF | Tracker(FFT|ACF|IF)
    win_sec: float = 8.0
    fmin: float = 0.7
    fmax: float = 2.0
    if_smooth_sec: float = 0.10
    tracker: TrackConfig = TrackConfig()
    quality_mode: str = "SNR"  # SNR | Conf | SNRxConf
    quality_floor: float = 0.05
    quality_snr_scale: float = 15.0


@dataclass
class State:
    params: Params
    R: Deque[float]
    G: Deque[float]
    B: Deque[float]
    T: Deque[float]
    tracker: FreqTracker
    last_t_for_tracker: Optional[float]
    metrics: dict


class ControlModel(BaseModel):
    algo: Optional[str] = Field(None, pattern=r"^(POS|CHROM)$")
    est: Optional[str] = Field(
        None, pattern=r"^(FFT|ACF|Hilbert-IF|Tracker\(FFT\)|Tracker\(ACF\)|Tracker\(IF\))$"
    )
    win_sec: Optional[float] = Field(None, ge=1.0, le=15.0)
    fmin: Optional[float] = Field(None, ge=0.2, le=2.0)
    fmax: Optional[float] = Field(None, ge=1.5, le=5.0)
    if_smooth_sec: Optional[float] = Field(None, ge=0.02, le=0.5)
    tracker: Optional[TrackConfig] = None
    quality_mode: Optional[str] = Field(None, pattern=r"^(SNR|Conf|SNRxConf)$")
    quality_floor: Optional[float] = Field(None, ge=0.0, le=0.5)
    quality_snr_scale: Optional[float] = Field(None, ge=5.0, le=30.0)


class IngestModel(BaseModel):
    t0: float
    dt: float
    mean_rgb: list[list[float]]


def make_app() -> FastAPI:
    app = FastAPI(title="rPPG Service", version="0.2.0")

    state = State(
        params=Params(),
        R=deque(maxlen=15 * 60 * 2),
        G=deque(maxlen=15 * 60 * 2),
        B=deque(maxlen=15 * 60 * 2),
        T=deque(maxlen=15 * 60 * 2),
        tracker=FreqTracker(TrackConfig()),
        last_t_for_tracker=None,
        metrics={"status": "init"},
    )

    loop_task: Optional[asyncio.Task] = None
    lock = asyncio.Lock()
    ws_clients: set[WebSocket] = set()

    @app.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - integration
        nonlocal loop_task
        loop_task = asyncio.create_task(process_loop())

    @app.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - integration
        nonlocal loop_task
        if loop_task:
            loop_task.cancel()
            try:
                await loop_task
            except Exception:
                pass

    async def process_loop() -> None:
        while True:
            try:
                await asyncio.sleep(0.2)
                async with lock:
                    await compute_once()
                # notify WS clients
                if ws_clients:
                    msg = json.dumps(state.metrics)
                    dead: list[WebSocket] = []
                    for w in ws_clients:
                        try:
                            await w.send_text(msg)
                        except Exception:
                            dead.append(w)
                    for w in dead:
                        ws_clients.discard(w)
            except asyncio.CancelledError:
                break
            except Exception:
                # keep loop running
                await asyncio.sleep(0.5)

    async def compute_once() -> None:
        if len(state.T) < 8:
            return
        t_np = np.array(state.T, dtype=np.float64)
        # estimate fs from recent diffs
        fs = float(1.0 / np.median(np.diff(t_np[-min(len(t_np), 50) :])))
        p = state.params
        fmax_eff = min(p.fmax, 0.45 * fs)
        if fmax_eff <= p.fmin:
            fmax_eff = p.fmin + 0.1
        L = max(8, int(p.win_sec * fs))
        if len(state.R) < L:
            return
        R = np.array(list(state.R)[-L:], dtype=np.float32)
        G = np.array(list(state.G)[-L:], dtype=np.float32)
        B = np.array(list(state.B)[-L:], dtype=np.float32)
        Rn = bandpass(moving_average_normalize(R, max(1, int(0.5 * fs))), fs, p.fmin, fmax_eff)
        Gn = bandpass(moving_average_normalize(G, max(1, int(0.5 * fs))), fs, p.fmin, fmax_eff)
        Bn = bandpass(moving_average_normalize(B, max(1, int(0.5 * fs))), fs, p.fmin, fmax_eff)
        s = pos_signal(Rn, Gn, Bn) if p.algo == "POS" else chrom_signal(Rn, Gn, Bn)
        # spectrum for quality
        x = (s - s.mean()) * np.hanning(s.size).astype(np.float32)
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(s.size, d=1.0 / fs)
        mag = np.abs(X)
        band = (freqs >= p.fmin) & (freqs <= fmax_eff)
        idx = int(np.argmax(mag * band))
        snr = snr_db(mag, idx)
        conf = peak_confidence(mag, idx)
        # estimator selection
        bpm_val: Optional[float] = None
        if p.est == "FFT":
            bpm, _ = estimate_bpm(s, fs=fs, fmin=p.fmin, fmax=fmax_eff)
            bpm_val = float(bpm)
            state.last_t_for_tracker = float(t_np[-1])
        elif p.est == "ACF":
            ar = estimate_bpm_acf(s, fs=fs, bpm_min=60.0 * p.fmin, bpm_max=60.0 * fmax_eff)
            bpm_val = float(ar.bpm) if ar.bpm is not None else None
            state.last_t_for_tracker = float(t_np[-1])
        elif p.est == "Hilbert-IF":
            ir = estimate_bpm_if(s, fs=fs, smooth_len=max(3, int(p.if_smooth_sec * fs)))
            bpm_val = float(ir.bpm) if ir.bpm is not None else None
            state.last_t_for_tracker = float(t_np[-1])
        else:
            # Tracker modes
            now = float(t_np[-1])
            if state.last_t_for_tracker is None:
                state.last_t_for_tracker = now
            dt = max(1e-3, now - state.last_t_for_tracker)
            state.last_t_for_tracker = now
            state.tracker.predict(dt)
            meas_bpm: Optional[float] = None
            if p.est == "Tracker(FFT)":
                m, _ = estimate_bpm(s, fs=fs, fmin=p.fmin, fmax=fmax_eff)
                meas_bpm = float(m)
            elif p.est == "Tracker(ACF)":
                ar = estimate_bpm_acf(s, fs=fs, bpm_min=60.0 * p.fmin, bpm_max=60.0 * fmax_eff)
                meas_bpm = float(ar.bpm) if ar.bpm is not None else None
            elif p.est == "Tracker(IF)":
                ir = estimate_bpm_if(s, fs=fs, smooth_len=max(3, int(p.if_smooth_sec * fs)))
                meas_bpm = float(ir.bpm) if ir.bpm is not None else None
            # quality mapping
            if p.quality_mode == "SNR":
                qual = float(np.clip(snr / p.quality_snr_scale, p.quality_floor, 1.0))
            elif p.quality_mode == "Conf":
                qual = float(np.clip(conf, p.quality_floor, 1.0))
            else:
                qual = float(
                    np.clip((snr / p.quality_snr_scale) * conf, p.quality_floor, 1.0)
                )
            if meas_bpm is not None:
                state.tracker.update(meas_bpm / 60.0, quality=qual)
            tb = state.tracker.value_bpm()
            bpm_val = float(tb) if tb is not None else None
        # respiration from envelope
        rr = estimate_rr_envelope(s, fs=fs, rr_min_hz=0.1, rr_max_hz=0.5)
        state.metrics = {
            "t": float(t_np[-1]),
            "bpm": bpm_val,
            "rr": float(rr.brpm) if rr.brpm is not None else None,
            "snr": float(snr),
            "conf": float(conf),
            "fs": float(fs),
            "algo": p.algo,
            "est": p.est,
        }

    @app.get("/health")
    async def health() -> dict[str, str]:  # pragma: no cover - trivial
        return {"status": "ok"}

    @app.get("/metrics")
    async def get_metrics() -> dict:
        async with lock:
            return dict(state.metrics)

    @app.post("/control")
    async def post_control(cfg: ControlModel) -> dict:
        async with lock:
            p = state.params
            data = cfg.model_dump(exclude_none=True)
            for k, v in data.items():
                if k == "tracker" and isinstance(v, TrackConfig):
                    state.tracker.cfg = v
                else:
                    setattr(p, k, v)
            return {"status": "ok", "params": p.__dict__}

    @app.post("/ingest")
    async def post_ingest(payload: IngestModel) -> dict:
        # Append meanRGB samples with timestamps
        t = payload.t0
        dt = payload.dt
        if not payload.mean_rgb:
            return {"status": "empty"}
        async with lock:
            for r, g, b in payload.mean_rgb:
                state.R.append(float(r))
                state.G.append(float(g))
                state.B.append(float(b))
                state.T.append(float(t))
                t += dt
        return {"status": "ok", "count": len(payload.mean_rgb)}

    @app.websocket("/ws")
    async def ws_metrics(ws: WebSocket) -> None:  # pragma: no cover - integration
        await ws.accept()
        ws_clients.add(ws)
        try:
            while True:
                # keep alive; updates are pushed from loop
                await asyncio.sleep(30)
        except WebSocketDisconnect:
            ws_clients.discard(ws)
        except Exception:
            ws_clients.discard(ws)

    return app


app = make_app()


def main() -> None:  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()
