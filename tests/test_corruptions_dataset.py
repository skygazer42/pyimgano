from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")


def _write_rgb(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


def test_corruptions_dataset_is_deterministic_per_index(tmp_path: Path) -> None:
    from pyimgano.datasets.corruptions import CorruptionsDataset

    root = tmp_path / "imgs"
    a = root / "a.png"
    b = root / "b.png"
    _write_rgb(a, color=(10, 20, 30))
    _write_rgb(b, color=(40, 50, 60))

    ds = CorruptionsDataset([a, b], corruption="lighting", severity=2, seed=123)
    item0a = ds[0]
    item0b = ds[0]

    assert item0a.image_u8.shape == (32, 32, 3)
    assert item0a.image_u8.dtype == np.uint8
    assert np.array_equal(item0a.image_u8, item0b.image_u8)


def test_corruptions_dataset_smoke_synthesis_preset(tmp_path: Path) -> None:
    from pyimgano.datasets.corruptions import CorruptionsDataset

    root = tmp_path / "imgs"
    a = root / "a.png"
    _write_rgb(a, color=(120, 120, 120))

    ds = CorruptionsDataset([a], corruption="synthesis_preset", severity=3, seed=0)
    item = ds[0]
    assert item.image_u8.shape == (32, 32, 3)
    assert item.mask_u8 is not None
    assert item.mask_u8.shape == (32, 32)
    assert item.mask_u8.dtype == np.uint8
