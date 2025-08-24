## rPPG Demo (POS/CHROM) — Desktop & Web-ready

This repository implements remote photoplethysmography (rPPG) based on the TBME 2016 paper by Wang, Stuijk, and de Haan. It provides a DearPyGUI desktop app with live camera preview, rPPG waveform projection (POS/CHROM), BPM estimation, spectrum/SNR, and recording. A future Web UI is planned via a local service.

### Quick Start
- Create venv and sync deps: `uv sync`
- Lint/format: `uv run task lint` / `uv run task fmt`
- Run app: `uv run task run`

Notes
- Grant camera permission to the terminal if prompted (macOS: System Settings → Privacy & Security → Camera).
- Lighting: diffuse indoor lighting recommended; avoid strong motion.

### Features
- Camera preview (OpenCV) with ROI (MediaPipe face-detection; cheeks + forehead).
- rPPG pipeline: normalize → band-pass → POS/CHROM → BPM.
- Spectrum plot and SNR indicator.
- Recording (CSV + JSON meta) under `runs/<timestamp>/`.

### Controls
- Algorithm: POS / CHROM
- Window (s), Band min/max (Hz)
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
