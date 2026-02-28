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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mvtec_ad2_manifest_converter_writes_expected_records(tmp_path: Path) -> None:
    from pyimgano.datasets.mvtec_ad2 import convert_mvtec_ad2_to_manifest

    root = tmp_path / "mvtec_ad2"
    cat = "bottle"

    _write_rgb(root / cat / "train" / "good" / "n0.png", color=(10, 20, 30))
    _write_rgb(root / cat / "validation" / "good" / "v0.png", color=(20, 30, 40))
    _write_rgb(root / cat / "test_public" / "good" / "x0.png", color=(30, 40, 50))
    _write_rgb(root / cat / "test_public" / "bad" / "a0.png", color=(200, 10, 10))
    _write_mask(root / cat / "test_public" / "ground_truth" / "bad" / "a0_mask.png", value=255)

    out = tmp_path / "out" / "manifest.jsonl"
    records = convert_mvtec_ad2_to_manifest(
        root=root,
        category=cat,
        out_path=out,
        split="test_public",
        absolute_paths=False,
        include_masks=True,
    )

    assert out.exists()
    parsed = _read_jsonl(out)
    assert parsed == records

    # Basic sanity: we have all key splits.
    splits = {r.get("split") for r in records}
    assert {"train", "val", "test"} <= splits

    # Ensure at least one anomaly carries a mask_path.
    anomaly = [r for r in records if r.get("split") == "test" and int(r.get("label", 0)) == 1]
    assert anomaly
    assert any("mask_path" in r for r in anomaly)

