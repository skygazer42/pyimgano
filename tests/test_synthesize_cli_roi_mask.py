from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=color).save(path)


def _write_roi_mask(path: Path, *, allowed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    v = 255 if allowed else 0
    Image.fromarray(np.full((64, 64), v, dtype=np.uint8), mode="L").save(path)


def test_synthesize_dataset_with_empty_roi_produces_no_anomalies(tmp_path: Path) -> None:
    from pyimgano.datasets.manifest import iter_manifest_records
    from pyimgano.synthesize_cli import synthesize_dataset

    in_dir = tmp_path / "normals"
    _write_rgb(in_dir / "n0.png", color=(10, 20, 30))
    _write_rgb(in_dir / "n1.png", color=(30, 40, 50))

    roi_path = tmp_path / "roi.png"
    _write_roi_mask(roi_path, allowed=False)  # no allowed region

    out_root = tmp_path / "out"
    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category="demo",
        preset="scratch",
        seed=0,
        roi_mask_path=roi_path,
        n_train=2,
        n_test_normal=2,
        n_test_anomaly=2,
    )

    labels = []
    for rec in iter_manifest_records(out_root / "manifest.jsonl"):
        if rec.split == "test":
            labels.append(0 if rec.label is None else int(rec.label))

    assert labels
    assert set(labels).issubset({0})
