from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=color).save(path)


def _write_mask(path: Path, *, value: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16), int(value), dtype=np.uint8), mode="L").save(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_manifest_cli_mvtec_ad2_converter(tmp_path: Path) -> None:
    from pyimgano.manifest_cli import main as manifest_main

    root = tmp_path / "mvtec_ad2"
    cat = "bottle"

    _write_rgb(root / cat / "train" / "good" / "n0.png", color=(10, 20, 30))
    _write_rgb(root / cat / "test_public" / "good" / "x0.png", color=(20, 30, 40))
    _write_rgb(root / cat / "test_public" / "bad" / "a0.png", color=(200, 10, 10))
    _write_mask(root / cat / "test_public" / "ground_truth" / "bad" / "a0_mask.png", value=255)

    out = tmp_path / "out" / "manifest.jsonl"
    rc = manifest_main(
        [
            "--dataset",
            "mvtec_ad2",
            "--root",
            str(root),
            "--category",
            cat,
            "--out",
            str(out),
            "--include-masks",
        ]
    )
    assert rc == 0
    assert out.exists()

    records = _read_jsonl(out)
    assert records
    assert any(r.get("split") == "test" and int(r.get("label", 0)) == 1 for r in records)
    assert any("mask_path" in r for r in records if int(r.get("label", 0)) == 1)
