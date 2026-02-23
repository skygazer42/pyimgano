from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pyimgano.utils.datasets import CustomDataset


def _write_rgb(path: Path, *, size=(32, 32), value=128) -> None:
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * int(value)
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, *, size=(32, 32)) -> None:
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(path), mask)


def test_custom_dataset_validate_structure_ok(tmp_path: Path) -> None:
    (tmp_path / "train" / "normal").mkdir(parents=True)
    (tmp_path / "test" / "normal").mkdir(parents=True)
    (tmp_path / "test" / "anomaly").mkdir(parents=True)
    (tmp_path / "ground_truth" / "anomaly").mkdir(parents=True)

    _write_rgb(tmp_path / "train" / "normal" / "train_0.png")
    _write_rgb(tmp_path / "test" / "normal" / "ok_0.png")
    _write_rgb(tmp_path / "test" / "anomaly" / "bad_0.png", value=200)
    _write_mask(tmp_path / "ground_truth" / "anomaly" / "bad_0_mask.png")

    ds = CustomDataset(root=str(tmp_path), load_masks=True)
    ds.validate_structure()


def test_custom_dataset_validate_structure_missing_dirs(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    ds = CustomDataset(root=str(tmp_path), load_masks=False)
    try:
        ds.validate_structure()
    except ValueError as exc:
        msg = str(exc).lower()
        assert "invalid custom dataset structure" in msg
        assert "missing directory" in msg
        return
    raise AssertionError("Expected ValueError")


def test_custom_dataset_validate_structure_missing_masks(tmp_path: Path) -> None:
    (tmp_path / "train" / "normal").mkdir(parents=True)
    (tmp_path / "test" / "normal").mkdir(parents=True)
    (tmp_path / "test" / "anomaly").mkdir(parents=True)
    (tmp_path / "ground_truth" / "anomaly").mkdir(parents=True)

    _write_rgb(tmp_path / "train" / "normal" / "train_0.png")
    _write_rgb(tmp_path / "test" / "normal" / "ok_0.png")
    _write_rgb(tmp_path / "test" / "anomaly" / "bad_0.png", value=200)
    # Intentionally omit the mask.

    ds = CustomDataset(root=str(tmp_path), load_masks=True)
    try:
        ds.validate_structure()
    except ValueError as exc:
        msg = str(exc).lower()
        assert "mask" in msg
        return
    raise AssertionError("Expected ValueError")

