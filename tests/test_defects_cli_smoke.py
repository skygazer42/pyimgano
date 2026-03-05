from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_defects_cli_smoke(tmp_path: Path) -> None:
    from pyimgano.defects_cli import main as defects_main

    amap = np.zeros((16, 16), dtype=np.float32)
    amap[6:10, 7:11] = 1.0

    amap_path = tmp_path / "map.npy"
    np.save(amap_path, amap)

    out_mask = tmp_path / "mask.png"
    out_jsonl = tmp_path / "regions.jsonl"

    rc = defects_main(
        [
            "--map",
            str(amap_path),
            "--pixel-threshold",
            "0.5",
            "--out-mask",
            str(out_mask),
            "--out-jsonl",
            str(out_jsonl),
        ]
    )
    assert rc == 0
    assert out_mask.exists()
    assert out_jsonl.exists()

    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "regions" in row
