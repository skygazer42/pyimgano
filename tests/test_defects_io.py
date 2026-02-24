from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from pyimgano.defects.io import save_binary_mask_npy, save_binary_mask_png


def test_save_binary_mask_png_writes_readable_png(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 3:5] = 255

    out = tmp_path / "mask.png"
    save_binary_mask_png(mask, out)

    loaded = np.asarray(Image.open(out), dtype=np.uint8)
    assert loaded.shape == (8, 8)
    assert int(loaded.max()) == 255


def test_save_binary_mask_npy_writes_readable_npy(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 255

    out = tmp_path / "mask.npy"
    save_binary_mask_npy(mask, out)

    loaded = np.load(out)
    assert loaded.dtype == np.uint8
    assert loaded.shape == (8, 8)
    assert int(loaded[1, 1]) == 255

