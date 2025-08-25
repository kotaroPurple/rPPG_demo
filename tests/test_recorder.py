from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rppg.recorder import Recorder, RecorderConfig


def test_recorder_writes_csv_and_meta() -> None:
    with TemporaryDirectory() as td:
        out = Path(td)
        rec = Recorder(RecorderConfig(out_dir=out, base_name="test"))
        rec.open(["a", "b"])
        rec.write_row([1, 2])
        rec.write_row([3, 4])
        rec.write_meta({"k": "v"})
        rec.close()

        csv_path = out / "test.csv"
        meta_path = out / "test.json"
        assert csv_path.exists()
        assert meta_path.exists()
        txt = csv_path.read_text().strip().splitlines()
        assert txt[0] == "a,b"
        assert txt[1] == "1,2"
        meta = json.loads(meta_path.read_text())
        assert meta["k"] == "v"

