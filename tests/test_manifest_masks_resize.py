from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pyimgano.datasets.manifest import load_manifest_benchmark_split


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_masks_resize_aligns_with_resize_param(tmp_path: Path) -> None:
    import cv2

    mdir = tmp_path / "m"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = mdir / "manifest.jsonl"

    # Paths only need to exist; no image loading occurs for path-based detectors.
    for name in ["train.png", "good.png", "bad.png"]:
        (mdir / name).touch()

    mask_path = mdir / "bad_mask.png"
    mask_small = np.zeros((3, 4), dtype=np.uint8)
    mask_small[1, 2] = 255
    cv2.imwrite(str(mask_path), mask_small)

    _write_jsonl(
        manifest,
        [
            {"image_path": "train.png", "category": "bottle", "split": "train"},
            {"image_path": "good.png", "category": "bottle", "split": "test", "label": 0},
            {
                "image_path": "bad.png",
                "category": "bottle",
                "split": "test",
                "label": 1,
                "mask_path": "bad_mask.png",
            },
        ],
    )

    split = load_manifest_benchmark_split(
        manifest_path=manifest,
        root_fallback=None,
        category="bottle",
        resize=(6, 8),
        load_masks=True,
    )

    assert split.test_masks is not None
    assert split.test_masks.shape == (len(split.test_paths), 6, 8)
    assert set(np.unique(split.test_masks)).issubset({0, 1})

    # Normal sample mask is all zeros; anomaly sample has a non-empty region.
    assert int(split.test_masks[0].sum()) == 0
    assert int(split.test_masks[1].sum()) > 0
