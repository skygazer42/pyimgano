from __future__ import annotations

import numpy as np


class _TupleOutputDetector:
    """Detector stub: decision_function returns (scores, maps)."""

    def decision_function(self, X):  # noqa: ANN001, ANN201
        items = list(X)
        if not items:
            return np.zeros((0,), dtype=np.float32), np.zeros((0, 1, 1), dtype=np.float32)

        maps = []
        scores = []
        for img in items:
            arr = np.asarray(img, dtype=np.float32)
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError(f"Expected RGB/HWC images, got shape={arr.shape}")
            gray = np.mean(arr, axis=2) / 255.0
            gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
            maps.append(gray)
            scores.append(float(np.mean(gray)))

        return np.asarray(scores, dtype=np.float32), np.stack(maps, axis=0).astype(np.float32)


def test_infer_supports_decision_function_tuple_outputs() -> None:
    from pyimgano.inference.api import infer

    det = _TupleOutputDetector()

    img0 = np.zeros((32, 48, 3), dtype=np.uint8)
    img1 = np.full((32, 48, 3), 255, dtype=np.uint8)

    results = infer(
        det,
        [img0, img1],
        input_format="rgb_u8_hwc",
        include_maps=True,
    )
    assert len(results) == 2
    assert results[0].anomaly_map is not None
    assert results[1].anomaly_map is not None

    assert results[0].anomaly_map.shape == (32, 48)
    assert results[0].anomaly_map.dtype == np.float32
    assert float(results[0].score) <= float(results[1].score)
