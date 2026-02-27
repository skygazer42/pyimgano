from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_synthesize_cli_smoke(tmp_path: Path) -> None:
    from pyimgano.synthesize_cli import main

    in_dir = tmp_path / "in"
    _write_rgb(in_dir / "a.png", color=(10, 10, 10))
    _write_rgb(in_dir / "b.png", color=(20, 20, 20))
    _write_rgb(in_dir / "c.png", color=(30, 30, 30))

    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"
    code = main(
        [
            "--in-dir",
            str(in_dir),
            "--out-root",
            str(out_root),
            "--category",
            "synthetic",
            "--preset",
            "scratch",
            "--n-train",
            "2",
            "--n-test-normal",
            "1",
            "--n-test-anomaly",
            "2",
            "--seed",
            "0",
            "--manifest",
            str(manifest),
        ]
    )
    assert code == 0
    assert manifest.exists()

    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows

    # Expect some anomaly records with mask_path.
    anomalies = [r for r in rows if r.get("split") == "test" and r.get("label") == 1]
    assert len(anomalies) >= 1
    assert all("mask_path" in r for r in anomalies)

