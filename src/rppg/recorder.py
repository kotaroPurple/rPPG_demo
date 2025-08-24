"""Simple CSV/JSON recorder (skeleton)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class RecorderConfig:
    out_dir: Path
    base_name: str = "session"


class Recorder:
    """Append rows to CSV and write a JSON metadata file."""

    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.cfg.out_dir / f"{self.cfg.base_name}.csv"
        self.meta_path = self.cfg.out_dir / f"{self.cfg.base_name}.json"
        self._csv_file: Optional[object] = None
        self._writer: Optional[csv.writer] = None

    def open(self, header: Iterable[str]) -> None:
        self._csv_file = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(list(header))

    def write_row(self, row: Iterable[object]) -> None:
        if self._writer is None:
            raise RuntimeError("Recorder not opened")
        self._writer.writerow(list(row))

    def write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

