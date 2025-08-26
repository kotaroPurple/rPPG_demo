## rPPG Demo (POS/CHROM) — Desktop & Web-ready

This repository implements remote photoplethysmography (rPPG) based on the TBME 2016 paper by Wang, Stuijk, and de Haan. It provides a DearPyGUI desktop app with live camera preview, rPPG waveform projection (POS/CHROM), BPM estimation, spectrum/SNR, and recording. A future Web UI is planned via a local service.

### Quick Start
- Create venv and sync deps: `uv sync`
- Lint/format: `uv run task lint` / `uv run task fmt`
- Run app: `uv run task run` (or `uv run python -m rppg.app`)
- Web service: `uv run task serve` → http://127.0.0.1:8000/metrics

Notes
- Grant camera permission to the terminal if prompted (macOS: System Settings → Privacy & Security → Camera).
- Lighting: diffuse indoor lighting recommended; avoid strong motion.

### Features
- Camera preview (OpenCV) with ROI (MediaPipe face-detection; cheeks + forehead).
- rPPG pipeline: normalize → band-pass → POS/CHROM → BPM.
- SNR indicator, peak confidence, and optional Spectrum plot (default OFF for stability).
- Recording (CSV + JSON meta) under `runs/<timestamp>/`.

### Controls
- Algorithm: POS / CHROM
- Window (s): waveform X-axis = −win..0 sec
- Band min/max (Hz)
- Record (CSV): toggle to start/stop (meta saved on stop)

### Docs
- `docs/01_requirements.md`: Requirements (uv, DearPyGUI), UX
- `docs/02_rppg_algorithm.md`: Theory and formulas ($/$$, CHROM/POS)
- `docs/03_design.md`: Architecture (Desktop + future Web)
- `docs/00_tasks.md`: Project tasks checklist

### Dev Notes
- Tests (non-GUI): `uv run task test`
- Style: Ruff with line-length 100
- Source layout: `src/`; tasks handled via `taskipy` (`uv run task -l`)

- UI: Plots avoid scrolling. Waveform Y-axis is fixed at ±0.005; BPM timeline Y-axis fixed at 40–120 BPM.
- Filtering: For embedded stability, band-pass uses `lfilter` (causal). Zero-phase is not applied by design.
### Troubleshooting
- macOS Continuity Camera: If iPhone camera attaches automatically and causes instability, disable Continuity Camera in macOS/iPhone Handoff settings, or select the built-in camera (Camera=0/1) before connecting.
- Spectrum plot: On some systems the spectrum drawing can crash GPU drivers. Keep Spectrum OFF (default) or use Bars mode. Updates are throttled and downsampled, but if issues persist, leave Spectrum OFF.
- Logs: Check `logs/app.log` and `logs/faulthandler.log` for diagnostics.
### Web UI (Prototype)
- Start service: `uv run task serve`
- Static HTML (open locally): `web/frontend/index.html`
- The page polls `/metrics` and shows BPM/SNR. Prototype only.
