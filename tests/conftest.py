from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is importable for tests without editable install
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
