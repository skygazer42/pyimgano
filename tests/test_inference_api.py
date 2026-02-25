import json

import importlib.util

import numpy as np
import pytest

from pyimgano.inference.api import calibrate_threshold, infer, results_to_jsonable
from pyimgano.inputs.image_format import ImageFormat
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess


class _ScoreOnly:
    def __init__(self):
        self.threshold_ = 0.5

    def decision_function(self, X):
        assert len(X) == 2
        return np.asarray([0.1, 0.9], dtype=np.float32)


def test_infer_returns_scores_and_labels():
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC)
    assert [r.score for r in out] == pytest.approx([0.1, 0.9])
    assert [r.label for r in out] == [0, 1]


def test_calibrate_threshold_sets_detector_threshold():
    class _Cal:
        def decision_function(self, X):
            assert len(X) == 2
            return np.asarray([0.1, 0.9], dtype=np.float32)

    det = _Cal()
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    threshold = calibrate_threshold(det, imgs, input_format=ImageFormat.RGB_U8_HWC, quantile=0.5)
    assert threshold == 0.5
    assert det.threshold_ == 0.5


def test_results_to_jsonable_uses_stable_python_types():
    class _Map:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, X):
            return np.asarray([1.0], dtype=np.float32)

        def get_anomaly_map(self, item):
            _ = item
            return np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)

    imgs = [np.zeros((2, 2, 3), dtype=np.uint8)]
    results = infer(_Map(), imgs, input_format=ImageFormat.RGB_U8_HWC, include_maps=True)
    payload = results_to_jsonable(results)

    assert isinstance(payload[0]["score"], float)
    assert payload[0]["label"] == 1
    assert payload[0]["anomaly_map"]["shape"] == [2, 2]
    assert payload[0]["anomaly_map"]["dtype"] == "float32"
    assert "anomaly_map_values" not in payload[0]

    # Should be JSON-serializable without extra hooks.
    json.dumps(payload)


def test_infer_applies_postprocess_to_maps_only():
    class _Map:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, X):
            return np.asarray([1.0], dtype=np.float32)

        def get_anomaly_map(self, item):
            _ = item
            return np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)

    imgs = [np.zeros((2, 2, 3), dtype=np.uint8)]
    post = AnomalyMapPostprocess(normalize=True, normalize_method="minmax")
    results = infer(
        _Map(), imgs, input_format=ImageFormat.RGB_U8_HWC, include_maps=True, postprocess=post
    )

    out_map = results[0].anomaly_map
    assert out_map is not None
    assert float(out_map.min()) == 0.0
    assert float(out_map.max()) == 1.0


def test_infer_supports_batch_only_detectors_for_numpy_inputs():
    class _BatchOnly:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, X):
            arr = np.asarray(X)
            if arr.ndim != 4:
                raise TypeError("expected batched ndarray (N,H,W,C)")
            return np.asarray([float(arr[0].max())], dtype=np.float32)

        def predict_anomaly_map(self, X):
            arr = np.asarray(X)
            if arr.ndim != 4:
                raise TypeError("expected batched ndarray (N,H,W,C)")
            return arr[..., 0].astype(np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8)]
    imgs[0][1:3, 2:4, :] = 200

    results = infer(_BatchOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC, include_maps=True)
    assert results[0].label == 1
    assert results[0].anomaly_map is not None
    assert results[0].anomaly_map.shape == (4, 4)


def test_infer_supports_batch_size_chunking_preserves_order() -> None:
    class _Chunkable:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            # score = max pixel intensity (per image)
            scores: list[float] = []
            for item in list(X):
                arr = np.asarray(item)
                scores.append(float(arr.max()) / 255.0)
            return np.asarray(scores, dtype=np.float32)

    imgs = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8) * 100,
        np.ones((4, 4, 3), dtype=np.uint8) * 250,
    ]

    det = _Chunkable()
    out = infer(det, imgs, input_format=ImageFormat.RGB_U8_HWC, batch_size=2)
    assert [r.score for r in out] == pytest.approx([0.0, 100 / 255.0, 250 / 255.0])
    assert [r.label for r in out] == [0, 0, 1]


def test_infer_amp_is_best_effort() -> None:
    class _ScoreOnly:
        def decision_function(self, X):
            return np.asarray([0.1], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8)]
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch:
        with pytest.warns(RuntimeWarning, match=r"torch is not installed"):
            out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC, amp=True)
    else:
        out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC, amp=True)
    assert len(out) == 1
