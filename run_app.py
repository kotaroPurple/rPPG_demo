"""Local runner for the rPPG demo with src/ layout.

Usage: uv run python run_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    # Ensure src/ is on sys.path so `import rppg` resolves
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    from rppg.app import main as app_main  # type: ignore

    app_main()


if __name__ == "__main__":
    main()

