import importlib.util
import json

import numpy as np
import pytest

from pyimgano.inference.api import calibrate_threshold, infer, infer_bgr, results_to_jsonable
from pyimgano.inputs.image_format import ImageFormat
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess


class _ScoreOnly:
    def __init__(self):
        self.threshold_ = 0.5

    def decision_function(self, x):
        assert len(x) == 2
        return np.asarray([0.1, 0.9], dtype=np.float32)


def test_infer_returns_scores_and_labels():
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC)
    assert [r.score for r in out] == pytest.approx([0.1, 0.9])
    assert [r.label for r in out] == [0, 1]


def test_calibrate_threshold_sets_detector_threshold():
    class _Cal:
        def decision_function(self, x):
            assert len(x) == 2
            return np.asarray([0.1, 0.9], dtype=np.float32)

    det = _Cal()
    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    threshold = calibrate_threshold(det, imgs, input_format=ImageFormat.RGB_U8_HWC, quantile=0.5)
    assert threshold == pytest.approx(0.5)
    assert det.threshold_ == pytest.approx(0.5)


def test_results_to_jsonable_uses_stable_python_types():
    class _Map:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, x):
            del x
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


def test_infer_can_include_label_confidence() -> None:
    class _ScoreWithConfidence:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(X) == 2
            return np.asarray([0.1, 0.9], dtype=np.float32)

        def predict_confidence(self, X):
            assert len(X) == 2
            return np.asarray([0.8, 0.9], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    out = infer(
        _ScoreWithConfidence(),
        imgs,
        input_format=ImageFormat.RGB_U8_HWC,
        include_confidence=True,
    )

    assert [r.label for r in out] == [0, 1]
    assert [r.label_confidence for r in out] == pytest.approx([0.8, 0.9])

    payload = results_to_jsonable(out)
    assert payload[0]["label_confidence"] == pytest.approx(0.8)
    assert payload[1]["label_confidence"] == pytest.approx(0.9)


def test_infer_can_reject_low_confidence_samples() -> None:
    class _ScoreWithConfidence:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(X) == 3
            return np.asarray([0.1, 0.4, 0.9], dtype=np.float32)

        def predict_confidence(self, X):
            assert len(X) == 3
            return np.asarray([0.8, 0.6, 0.9], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    out = infer(
        _ScoreWithConfidence(),
        imgs,
        input_format=ImageFormat.RGB_U8_HWC,
        reject_confidence_below=0.75,
    )

    assert [r.label for r in out] == [0, -2, 1]
    assert [r.rejected for r in out] == [False, True, False]
    assert [r.label_confidence for r in out] == pytest.approx([0.8, 0.6, 0.9])

    payload = results_to_jsonable(out)
    assert payload[0]["rejected"] is False
    assert payload[1]["rejected"] is True
    assert payload[1]["label"] == -2


def test_infer_can_export_postprocess_summary() -> None:
    class _ScoreOnly:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(X) == 1
            return np.asarray([0.1], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8)]
    out = infer(
        _ScoreOnly(),
        imgs,
        input_format=ImageFormat.RGB_U8_HWC,
        postprocess_summary={
            "maps_enabled": False,
            "runtime_postprocess_applied": False,
            "prediction_policy": {
                "reject_confidence_below": 0.75,
                "reject_label": -9,
            },
        },
    )

    payload = results_to_jsonable(out)
    assert payload[0]["postprocess_summary"] == {
        "maps_enabled": False,
        "runtime_postprocess_applied": False,
        "prediction_policy": {
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }


def test_infer_rejection_requires_confidence_support() -> None:
    class _ScoreOnly:
        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(X) == 1
            return np.asarray([0.1], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8)]

    with pytest.raises(RuntimeError, match="confidence"):
        infer(
            _ScoreOnly(),
            imgs,
            input_format=ImageFormat.RGB_U8_HWC,
            reject_confidence_below=0.75,
        )


def test_infer_applies_postprocess_to_maps_only():
    class _Map:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, x):
            del x
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
    assert float(out_map.min()) == pytest.approx(0.0)
    assert float(out_map.max()) == pytest.approx(1.0)


def test_infer_supports_batch_only_detectors_for_numpy_inputs():
    class _BatchOnly:
        def __init__(self):
            self.threshold_ = 0.0

        def decision_function(self, x):
            arr = np.asarray(x)
            if arr.ndim != 4:
                raise TypeError("expected batched ndarray (N,H,W,C)")
            return np.asarray([float(arr[0].max())], dtype=np.float32)

        def predict_anomaly_map(self, x):
            arr = np.asarray(x)
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

        def decision_function(self, x):
            # score = max pixel intensity (per image)
            scores: list[float] = []
            for item in x:
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


def test_infer_threads_u16_max_to_normalization() -> None:
    class _MaxScore:
        def __init__(self) -> None:
            self.threshold_ = 0.0

        def decision_function(self, x):
            items = list(x)
            scores: list[float] = []
            for item in items:
                arr = np.asarray(item)
                assert arr.shape == (2, 2, 3)
                assert arr.dtype == np.uint8
                scores.append(float(arr.max()))
            return np.asarray(scores, dtype=np.float32)

    gray_u16 = np.zeros((2, 2), dtype=np.uint16)
    gray_u16[0, 0] = 4095

    det = _MaxScore()

    # Default scaling (u16_max=None -> 65535) should keep this relatively small.
    out_default = infer(det, [gray_u16], input_format=ImageFormat.GRAY_U16_HW)
    assert 0.0 <= float(out_default[0].score) < 32.0

    # Industrial 12-bit sensors stored in uint16 often need u16_max=4095.
    out_12bit = infer(det, [gray_u16], input_format=ImageFormat.GRAY_U16_HW, u16_max=4095)
    assert float(out_12bit[0].score) == pytest.approx(255.0)


def test_infer_bgr_convenience_swaps_channels() -> None:
    class _FirstChannel:
        def __init__(self) -> None:
            self.threshold_ = None

        def decision_function(self, x):
            items = list(x)
            assert len(items) == 1
            arr = np.asarray(items[0])
            # inference API always passes canonical RGB/u8/HWC for numpy inputs
            assert arr.shape == (1, 1, 3)
            assert arr.dtype == np.uint8
            return np.asarray([float(arr[0, 0, 0])], dtype=np.float32)

    # BGR: [B, G, R] = [10, 20, 30] should become RGB [30, 20, 10]
    bgr = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    out = infer_bgr(_FirstChannel(), [bgr])
    assert float(out[0].score) == pytest.approx(30.0)


def test_infer_amp_is_best_effort() -> None:
    class _ScoreOnly:
        def decision_function(self, x):
            del x
            return np.asarray([0.1], dtype=np.float32)

    imgs = [np.zeros((4, 4, 3), dtype=np.uint8)]
    has_torch = importlib.util.find_spec("torch") is not None
    if not has_torch:
        with pytest.warns(RuntimeWarning, match=r"torch is not installed"):
            out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC, amp=True)
    else:
        out = infer(_ScoreOnly(), imgs, input_format=ImageFormat.RGB_U8_HWC, amp=True)
    assert len(out) == 1

