from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


def _write_mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=255).save(path)


def test_manifest_cli_generate_from_custom_layout(tmp_path: Path) -> None:
    from pyimgano.manifest_cli import main

    root = tmp_path / "custom_root"
    _write_rgb(root / "train" / "normal" / "t0.png")
    _write_rgb(root / "test" / "normal" / "g0.png")
    _write_rgb(root / "test" / "anomaly" / "b0.png")
    _write_mask(root / "ground_truth" / "anomaly" / "b0_mask.png")

    out = tmp_path / "manifest.jsonl"
    code = main(
        [
            "--root",
            str(root),
            "--out",
            str(out),
            "--category",
            "custom",
            "--include-masks",
        ]
    )
    assert code == 0
    assert out.exists()

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    rows = [json.loads(line) for line in lines]

    # Basic schema fields present.
    assert all("image_path" in r and "category" in r and "split" in r for r in rows)
    assert all(r["category"] == "custom" for r in rows)

    # Check split/labels.
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    assert len(train) == 1
    assert len(test) == 2
    assert any(r.get("label") == 0 for r in test)
    assert any(r.get("label") == 1 for r in test)

    # mask_path attached for anomaly record.
    anomaly = [r for r in test if r.get("label") == 1][0]
    assert "mask_path" in anomaly
    assert str(anomaly["mask_path"]).endswith("b0_mask.png")

    # Default is relative paths (should not be absolute).
    assert not str(anomaly["image_path"]).startswith("/")

