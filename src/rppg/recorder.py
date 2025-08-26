"""Simple CSV/JSON recorder with background writing."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
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
        self._queue: "Queue[list[object] | None]" = Queue(maxsize=1024)
        self._worker: Optional[threading.Thread] = None

    def open(self, header: Iterable[str]) -> None:
        self._csv_file = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(list(header))
        # Start background worker
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def write_row(self, row: Iterable[object]) -> None:
        if self._writer is None:
            raise RuntimeError("Recorder not opened")
        try:
            self._queue.put_nowait(list(row))
        except Exception:
            # If queue is full, drop the row to avoid blocking realtime loop
            pass

    def write_meta(self, meta: dict) -> None:
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def close(self) -> None:
        # Signal worker to stop
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._worker is not None:
            try:
                self._worker.join(timeout=0.5)
            except Exception:
                pass
            self._worker = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None

    def _loop(self) -> None:
        assert self._writer is not None and self._csv_file is not None
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except Empty:
                # Periodic flush
                try:
                    self._csv_file.flush()
                except Exception:
                    pass
                continue
            if item is None:
                # Drain remaining items
                while True:
                    try:
                        rest = self._queue.get_nowait()
                    except Empty:
                        break
                    if rest is None:
                        break
                    try:
                        self._writer.writerow(rest)
                    except Exception:
                        pass
                try:
                    self._csv_file.flush()
                except Exception:
                    pass
                break
            try:
                self._writer.writerow(item)
            except Exception:
                pass
