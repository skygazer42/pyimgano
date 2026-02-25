from __future__ import annotations

import numpy as np
import pytest

import pyimgano.inference.tiling as tiling


class _BatchOnlyDummyDetector:
    """Dummy detector that only accepts batched ndarray inputs.

    This exercises the list-vs-batch fallback logic in `TiledDetector`.
    """

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        arr = np.asarray(X)
        if arr.ndim != 4:
            raise TypeError("expected (N,H,W,C) batched ndarray")
        # score = max pixel value (across HWC)
        return arr.reshape(arr.shape[0], -1).max(axis=1).astype(np.float32)

    def predict_anomaly_map(self, X):
        arr = np.asarray(X)
        if arr.ndim != 4:
            raise TypeError("expected (N,H,W,C) batched ndarray")
        # map = first channel intensity
        return arr[..., 0].astype(np.float32)


def test_tiled_detector_score_reduce_max() -> None:
    base = _BatchOnlyDummyDetector()
    tiled = tiling.TiledDetector(detector=base, tile_size=4, stride=3, score_reduce="max")

    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[2:4, 3:5, :] = 255

    scores = tiled.decision_function([img])
    assert scores.shape == (1,)
    assert float(scores[0]) == 255.0


def test_tiled_detector_stitches_maps() -> None:
    base = _BatchOnlyDummyDetector()
    tiled = tiling.TiledDetector(detector=base, tile_size=4, stride=3, map_reduce="max")

    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[1:3, 4:6, :] = 200

    maps = tiled.predict_anomaly_map([img])
    assert maps.shape == (1, 6, 6)
    stitched = maps[0]
    assert float(stitched[1:3, 4:6].max()) == 200.0
    assert float(stitched[:1, :].max()) == 0.0


def test_tiled_detector_stitches_maps_hann_window() -> None:
    base = _BatchOnlyDummyDetector()
    tiled = tiling.TiledDetector(detector=base, tile_size=4, stride=3, map_reduce="hann")

    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[1:3, 4:6, :] = 200

    maps = tiled.predict_anomaly_map([img])
    assert maps.shape == (1, 6, 6)
    stitched = maps[0]
    # Windowed blending is float math; allow tiny numeric noise across platforms.
    assert np.isclose(float(stitched[1:3, 4:6].max()), 200.0, atol=1e-3)
    assert float(stitched[:1, :].max()) == 0.0


def test_tiled_detector_stitches_maps_gaussian_window() -> None:
    base = _BatchOnlyDummyDetector()
    tiled = tiling.TiledDetector(detector=base, tile_size=4, stride=3, map_reduce="gaussian")

    img = np.zeros((6, 6, 3), dtype=np.uint8)
    img[1:3, 4:6, :] = 200

    maps = tiled.predict_anomaly_map([img])
    assert maps.shape == (1, 6, 6)
    stitched = maps[0]
    # Windowed blending is float math; allow tiny numeric noise across platforms.
    assert np.isclose(float(stitched[1:3, 4:6].max()), 200.0, atol=1e-3)
    assert float(stitched[:1, :].max()) == 0.0


def test_tiled_detector_caches_tile_coords(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    original = tiling.iter_tile_coords

    def _wrapped(height: int, width: int, *, tile_size: int, stride: int):
        calls["count"] += 1
        return original(height, width, tile_size=tile_size, stride=stride)

    monkeypatch.setattr(tiling, "iter_tile_coords", _wrapped)

    base = _BatchOnlyDummyDetector()
    tiled = tiling.TiledDetector(detector=base, tile_size=4, stride=3, map_reduce="max")

    img = np.zeros((6, 6, 3), dtype=np.uint8)
    tiled.decision_function([img])
    tiled.decision_function([img])
    assert calls["count"] == 1
