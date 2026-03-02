from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _write_rgb(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(int(seed))
    arr = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_defect_bank(bank_dir: Path) -> None:
    pytest.importorskip("cv2")
    import cv2

    bank_dir.mkdir(parents=True, exist_ok=True)
    defect = np.zeros((24, 24, 3), dtype=np.uint8)
    defect[:, :] = (30, 180, 30)
    mask = np.zeros((24, 24), dtype=np.uint8)
    cv2.circle(mask, (12, 12), 9, color=255, thickness=-1)

    cv2.imwrite(str(bank_dir / "spot.png"), defect)
    cv2.imwrite(str(bank_dir / "spot_mask.png"), mask)


def test_synthesize_cli_defect_bank_dir_generates_anomalies_with_meta(tmp_path: Path) -> None:
    from pyimgano.synthesize_cli import main

    in_dir = tmp_path / "in"
    for i in range(6):
        _write_rgb(in_dir / f"{i}.png", seed=100 + i)

    bank_dir = tmp_path / "bank"
    _write_defect_bank(bank_dir)

    out_root = tmp_path / "out"
    manifest = out_root / "manifest.jsonl"

    code = main(
        [
            "--in-dir",
            str(in_dir),
            "--out-root",
            str(out_root),
            "--defect-bank-dir",
            str(bank_dir),
            "--blend",
            "alpha",
            "--alpha",
            "0.9",
            "--n-train",
            "4",
            "--n-test-normal",
            "1",
            "--n-test-anomaly",
            "2",
            "--seed",
            "0",
            "--manifest",
            str(manifest),
            "--absolute-paths",
        ]
    )
    assert code == 0
    assert manifest.exists()

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").strip().splitlines()]
    anomalies = [r for r in rows if int(r.get("label", 0)) == 1]
    assert anomalies, "expected at least one anomaly record"

    for rec in anomalies:
        meta = rec.get("meta") or {}
        assert meta.get("preset_id") == "defect_bank"
        assert "severity" in meta
        assert "num_defects" in meta

