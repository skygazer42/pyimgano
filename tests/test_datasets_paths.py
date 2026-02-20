from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pyimgano.utils.datasets import BTADDataset, CustomDataset, MVTecDataset


def _write_rgb(path: Path, *, size=(32, 32), value=128) -> None:
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * value
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, *, size=(32, 32)) -> None:
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(path), mask)


def test_mvtec_path_accessors(tmp_path: Path) -> None:
    root = tmp_path / "mvtec"
    cat = "bottle"

    (root / cat / "train" / "good").mkdir(parents=True)
    (root / cat / "test" / "good").mkdir(parents=True)
    (root / cat / "test" / "crack").mkdir(parents=True)
    (root / cat / "ground_truth" / "crack").mkdir(parents=True)

    _write_rgb(root / cat / "train" / "good" / "train_0.png")
    _write_rgb(root / cat / "test" / "good" / "good_0.png")
    _write_rgb(root / cat / "test" / "crack" / "bad_0.png", value=200)
    _write_mask(root / cat / "ground_truth" / "crack" / "bad_0_mask.png")

    ds = MVTecDataset(root=str(root), category=cat, resize=(32, 32), load_masks=True)
    train_paths = ds.get_train_paths()
    assert len(train_paths) == 1

    test_paths, labels, masks = ds.get_test_paths()
    assert len(test_paths) == 2
    assert labels.shape == (2,)
    assert masks is not None
    assert masks.shape[0] == 2


def test_btad_path_accessors(tmp_path: Path) -> None:
    root = tmp_path / "btad"
    cat = "01"

    (root / cat / "train" / "ok").mkdir(parents=True)
    (root / cat / "test" / "ok").mkdir(parents=True)
    (root / cat / "test" / "ko").mkdir(parents=True)

    _write_rgb(root / cat / "train" / "ok" / "train_0.png")
    _write_rgb(root / cat / "test" / "ok" / "ok_0.png")
    _write_rgb(root / cat / "test" / "ko" / "ko_0.png", value=200)

    ds = BTADDataset(root=str(root), category=cat, resize=(32, 32))
    train_paths = ds.get_train_paths()
    assert len(train_paths) == 1
    test_paths, labels, masks = ds.get_test_paths()
    assert len(test_paths) == 2
    assert masks is None
    assert labels.tolist() == [0, 1]


def test_custom_path_accessors(tmp_path: Path) -> None:
    root = tmp_path / "custom"
    (root / "train" / "normal").mkdir(parents=True)
    (root / "test" / "normal").mkdir(parents=True)
    (root / "test" / "anomaly").mkdir(parents=True)
    (root / "ground_truth" / "anomaly").mkdir(parents=True)

    _write_rgb(root / "train" / "normal" / "train_0.png")
    _write_rgb(root / "test" / "normal" / "ok_0.png")
    _write_rgb(root / "test" / "anomaly" / "bad_0.png", value=200)
    _write_mask(root / "ground_truth" / "anomaly" / "bad_0_mask.png")

    ds = CustomDataset(root=str(root), resize=(32, 32), load_masks=True)
    train_paths = ds.get_train_paths()
    assert len(train_paths) == 1
    test_paths, labels, masks = ds.get_test_paths()
    assert len(test_paths) == 2
    assert set(labels.tolist()).issubset({0, 1})
    assert masks is not None

