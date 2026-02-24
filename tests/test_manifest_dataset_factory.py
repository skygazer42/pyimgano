from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pyimgano.datasets import load_dataset


def _write_rgb(path: Path, *, size: tuple[int, int] = (10, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(50, 60, 70)).save(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_dataset_manifest_factory_paths_and_arrays(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    mdir = tmp_path / "m"
    manifest = mdir / "manifest.jsonl"

    _write_rgb(mdir / "train0.png")
    _write_rgb(mdir / "val0.png")
    _write_rgb(mdir / "test0.png")
    _write_rgb(mdir / "an0.png")

    _write_jsonl(
        manifest,
        [
            {"image_path": "train0.png", "category": "bottle", "split": "train"},
            {"image_path": "val0.png", "category": "bottle", "split": "val"},
            {"image_path": "test0.png", "category": "bottle", "split": "test", "label": 0},
            {"image_path": "an0.png", "category": "bottle", "label": 1},
        ],
    )

    ds = load_dataset(
        "manifest",
        str(root),
        category="bottle",
        manifest_path=str(manifest),
        resize=(8, 8),
        load_masks=False,
        split_policy={"seed": 0, "test_normal_fraction": 0.0},
    )

    train_paths = ds.get_train_paths()
    test_paths, labels, masks = ds.get_test_paths()
    assert train_paths
    assert test_paths
    assert masks is None
    assert set(np.unique(labels)).issubset({0, 1})

    train_arr = ds.get_train_data()
    test_arr, test_labels, _ = ds.get_test_data()
    assert train_arr.ndim == 4
    assert test_arr.ndim == 4
    assert test_labels.shape[0] == test_arr.shape[0]

