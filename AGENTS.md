# Repository Guidelines

## Project Structure & Module Organization
- `docs/`: Specifications and algorithm notes (`01_requirements.md`, `02_rppg_algorithm.md`).
- `ref/`: Reference materials (e.g., paper PDFs).
- `src/` (planned): Python packages and app entry points (e.g., `src/rppg/`, `src/app.py`).
- `tests/` (planned): Unit/integration tests.
- `assets/` (optional): Icons, sample configs (avoid committing user videos).

## Build, Test, and Development Commands
- Setup deps: `uv sync` (installs from `pyproject.toml`).
- Add deps: `uv add opencv-python dearpygui mediapipe numpy scipy`.
- Run app (example): `uv run python -m rppg.app` or `uv run python src/app.py`.
- Run tests: `uv run pytest -q`.
- Lint/format (if configured): `uv run ruff check .`, `uv run ruff format .` or `uv run black .`.

## Coding Style & Naming Conventions
- Python 3.12+, PEP 8, type hints required for public APIs.
- Naming: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (consts).
- Keep modules focused; prefer `rppg/` packages for capture, roi, core, ui, quality, recorder.
- Docstrings: Google-style or NumPy-style with argument/return types.

## Testing Guidelines
- Framework: `pytest` with `tests/test_*.py` naming.
- Aim for ≥80% coverage on core signal processing (windowing, filters, CHROM/POS, BPM).
- Include deterministic fixtures; avoid camera I/O in unit tests (mock frames instead).

## Commit & Pull Request Guidelines
- Commits: short imperative summary, include scope when helpful (e.g., `rppg: add POS windowing`).
- PRs: clear description, rationale, screenshots (UI), and steps to verify. Link issues.
- CI/local checks should pass (lint, tests) before review.

## Security & Configuration Tips
- Do not commit captured videos or PII; keep data local. Respect `.gitignore`.
- Camera permissions vary by OS; document any manual steps in PRs touching capture.
- No secrets are required; never hardcode tokens/paths.

## Architecture Overview (Quick)
- Capture (OpenCV) → ROI (MediaPipe) → rPPG Core (CHROM/POS) → BPM/Quality → DearPyGUI UI → Recorder.
