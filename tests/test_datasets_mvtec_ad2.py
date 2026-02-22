from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pyimgano.utils.datasets import MVTecAD2Dataset


def _write_rgb(path: Path, *, size=(32, 32), value=128) -> None:
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * value
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, *, size=(32, 32)) -> None:
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(path), mask)


def test_mvtec_ad2_path_accessors(tmp_path: Path) -> None:
    root = tmp_path / "mvtec_ad2"
    cat = "capsule"

    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test_public" / "good").mkdir(parents=True)
    (root / cat / "test_public" / "bad").mkdir(parents=True)
    (root / cat / "test_public" / "ground_truth" / "bad").mkdir(parents=True)

    _write_rgb(root / cat / "train" / "good" / "train_0.png")
    _write_rgb(root / cat / "test_public" / "good" / "good_0.png")
    _write_rgb(root / cat / "test_public" / "bad" / "bad_0.png", value=200)
    _write_mask(root / cat / "test_public" / "ground_truth" / "bad" / "bad_0_mask.png")

    ds = MVTecAD2Dataset(
        root=str(root),
        category=cat,
        split="test_public",
        resize=(32, 32),
        load_masks=True,
    )
    train_paths = ds.get_train_paths()
    assert len(train_paths) == 1

    test_paths, labels, masks = ds.get_test_paths()
    assert len(test_paths) == 2
    assert labels.tolist() == [0, 1]
    assert masks is not None
    assert masks.shape == (2, 32, 32)
    assert masks[0].sum() == 0
    assert masks[1].sum() > 0

