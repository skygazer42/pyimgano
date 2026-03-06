from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pyimgano.pipelines.mvtec_visa import load_benchmark_split


def _write_rgb(path: Path, *, size: tuple[int, int] = (32, 32), value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * int(value)
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, *, size: tuple[int, int] = (32, 32)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    cv2.imwrite(str(path), mask)


def test_load_benchmark_split_mvtec_ad2_smoke(tmp_path: Path) -> None:
    root = tmp_path / "mvtec_ad2"
    cat = "capsule"

    _write_rgb(root / cat / "train" / "good" / "train_0.png", value=100)
    _write_rgb(root / cat / "test_public" / "good" / "good_0.png", value=120)
    _write_rgb(root / cat / "test_public" / "bad" / "bad_0.png", value=210)
    _write_mask(root / cat / "test_public" / "ground_truth" / "bad" / "bad_0_mask.png")

    split = load_benchmark_split(
        dataset="mvtec_ad2",
        root=str(root),
        category=cat,
        resize=(32, 32),
        load_masks=True,
    )

    assert len(split.train_paths) == 1
    assert len(split.test_paths) == 2
    assert split.test_labels.tolist() == [0, 1]
    assert split.test_masks is not None
    assert split.test_masks.shape == (2, 32, 32)
    assert split.test_masks[0].sum() == 0
    assert split.test_masks[1].sum() > 0
