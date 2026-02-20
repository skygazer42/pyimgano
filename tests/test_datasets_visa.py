from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pyimgano.utils.datasets import load_dataset


def _write_rgb(path: Path, *, size=(32, 32), value=128) -> None:
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * value
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, *, size=(32, 32)) -> None:
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(path), mask)


def test_load_visa_paths(tmp_path: Path) -> None:
    base = tmp_path / "visa_pytorch" / "dummy"
    (base / "train" / "good").mkdir(parents=True)
    (base / "test" / "good").mkdir(parents=True)
    (base / "test" / "bad").mkdir(parents=True)
    (base / "ground_truth" / "bad").mkdir(parents=True)

    _write_rgb(base / "train" / "good" / "train_0.png")
    _write_rgb(base / "train" / "good" / "train_1.png")
    _write_rgb(base / "test" / "good" / "good_0.png")

    _write_rgb(base / "test" / "bad" / "bad_0.png", value=200)
    _write_mask(base / "ground_truth" / "bad" / "bad_0.png")

    ds = load_dataset("visa", str(tmp_path), category="dummy", resize=(32, 32), load_masks=True)

    train_paths = ds.get_train_paths()
    assert len(train_paths) == 2

    test_paths, labels, masks = ds.get_test_paths()
    assert len(test_paths) == 2
    assert labels.shape == (2,)
    assert set(labels.tolist()).issubset({0, 1})
    assert masks is not None
    assert masks.shape[0] == len(test_paths)

