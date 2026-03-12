from __future__ import annotations

import numpy as np
import pytest

from pyimgano.inference.api import InferenceResult
from pyimgano.services.inference_service import run_inference


class _DummyDetector:
    input_mode = "numpy"

    def decision_function(self, X):
        n = len(list(X))
        return np.arange(float(n), dtype=np.float32)


def test_run_inference_returns_structured_records():
    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
    result = run_inference(detector=_DummyDetector(), inputs=X, input_format="rgb_u8_hwc")
    assert len(result.records) == 2
    assert result.records[1].score >= result.records[0].score


def test_run_inference_uses_runtime_adapter_score_path(monkeypatch):
    import pyimgano.services.inference_service as inference_service

    called = {"score_and_maps": 0}

    def fake_score_and_maps(detector, inputs, *, include_maps):
        _ = detector
        _ = include_maps
        called["score_and_maps"] += 1
        return np.asarray([0.1 for _ in inputs], dtype=np.float32), None

    monkeypatch.setattr(inference_service, "score_and_maps", fake_score_and_maps, raising=False)

    out = inference_service.run_inference(
        detector=_DummyDetector(),
        inputs=[np.zeros((4, 4, 3), dtype=np.uint8)],
        input_format="rgb_u8_hwc",
    )

    assert called["score_and_maps"] == 1
    assert len(out.records) == 1
    assert out.records[0] == InferenceResult(
        score=pytest.approx(0.1),
        label=None,
        anomaly_map=None,
    )
