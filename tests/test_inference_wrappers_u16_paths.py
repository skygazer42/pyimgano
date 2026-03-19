from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_gray_u16_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    arr = np.asarray(array)
    if arr.dtype != np.uint16 or arr.ndim != 2:
        raise ValueError("Expected uint16 HW array for test image writer")

    img = Image.fromarray(arr)
    img.save(str(path))


class _BatchOnlyMaxDetector:
    def fit(self, x, y=None):  # noqa: ANN001 - test stub
        del x, y
        return self

    def decision_function(self, x):  # noqa: ANN001 - test stub
        arr = np.asarray(x)
        if arr.ndim != 4:
            raise TypeError("expected batched ndarray (N,H,W,C)")
        return arr.reshape(arr.shape[0], -1).max(axis=1).astype(np.float32)


def test_tiled_detector_loads_u16_paths_with_u16_max(tmp_path: Path) -> None:
    from pyimgano.inference.tiling import TiledDetector

    gray_u16 = np.zeros((8, 8), dtype=np.uint16)
    gray_u16[2:4, 3:5] = 4095

    p = tmp_path / "u16.png"
    _write_gray_u16_png(p, gray_u16)

    base = _BatchOnlyMaxDetector()
    tiled = TiledDetector(detector=base, tile_size=4, stride=4, u16_max=4095)
    scores = tiled.decision_function([str(p)])

    assert scores.shape == (1,)
    assert np.isclose(float(scores[0]), 255.0)


def test_preprocessing_detector_loads_u16_paths_with_u16_max(tmp_path: Path) -> None:
    from pyimgano.inference.preprocessing import PreprocessingDetector

    gray_u16 = np.zeros((8, 8), dtype=np.uint16)
    gray_u16[1:3, 1:3] = 4095

    p = tmp_path / "u16.png"
    _write_gray_u16_png(p, gray_u16)

    base = _BatchOnlyMaxDetector()
    det = PreprocessingDetector(detector=base, illumination_contrast=None, u16_max=4095)
    scores = det.decision_function([str(p)])

    out = np.asarray(scores, dtype=np.float32).reshape(-1)
    assert out.shape == (1,)
    assert np.isclose(float(out[0]), 255.0)
