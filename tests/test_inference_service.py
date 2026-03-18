from __future__ import annotations

import numpy as np
import pytest

from pyimgano.inference.api import InferenceResult, result_to_jsonable
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
        decision_summary={
            "decision": "score_only",
            "threshold_applied": False,
            "has_confidence": False,
            "rejected": False,
            "requires_review": False,
            "review_reason": "unthresholded_score",
        },
    )


def test_run_inference_emits_decision_summary_for_triage() -> None:
    class _ScoreWithConfidence:
        input_mode = "numpy"

        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(list(X)) == 3
            return np.asarray([0.1, 0.4, 0.9], dtype=np.float32)

        def predict_confidence(self, X):
            assert len(list(X)) == 3
            return np.asarray([0.8, 0.6, 0.95], dtype=np.float32)

    X = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    out = run_inference(
        detector=_ScoreWithConfidence(),
        inputs=X,
        input_format="rgb_u8_hwc",
        reject_confidence_below=0.75,
    )

    assert [record.decision_summary for record in out.records] == [
        {
            "decision": "normal",
            "threshold_applied": True,
            "has_confidence": True,
            "rejected": False,
            "requires_review": False,
            "review_reason": "none",
        },
        {
            "decision": "rejected_low_confidence",
            "threshold_applied": True,
            "has_confidence": True,
            "rejected": True,
            "requires_review": True,
            "review_reason": "low_confidence",
        },
        {
            "decision": "anomalous",
            "threshold_applied": True,
            "has_confidence": True,
            "rejected": False,
            "requires_review": True,
            "review_reason": "anomaly_label",
        },
    ]

    payload = result_to_jsonable(out.records[1])
    assert payload["decision_summary"] == {
        "decision": "rejected_low_confidence",
        "threshold_applied": True,
        "has_confidence": True,
        "rejected": True,
        "requires_review": True,
        "review_reason": "low_confidence",
    }


def test_run_inference_threads_postprocess_summary_into_records() -> None:
    class _ScoreOnly:
        input_mode = "numpy"

        def __init__(self) -> None:
            self.threshold_ = 0.5

        def decision_function(self, X):
            assert len(list(X)) == 1
            return np.asarray([0.1], dtype=np.float32)

    out = run_inference(
        detector=_ScoreOnly(),
        inputs=[np.zeros((4, 4, 3), dtype=np.uint8)],
        input_format="rgb_u8_hwc",
        postprocess_summary={
            "maps_enabled": False,
            "runtime_postprocess_applied": False,
            "map_postprocess_summary": None,
        },
    )

    payload = result_to_jsonable(out.records[0])
    assert payload["postprocess_summary"] == {
        "maps_enabled": False,
        "runtime_postprocess_applied": False,
        "map_postprocess_summary": None,
    }
