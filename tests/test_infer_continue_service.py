from __future__ import annotations

from typing import Any

import pytest

from pyimgano.inference.api import InferenceResult
from pyimgano.services import inference_service
from pyimgano.services import infer_continue_service


def test_run_continue_on_error_inference_falls_back_to_per_input_when_batch_fails() -> None:
    calls: list[list[str]] = []
    ok_records: list[tuple[int, str, float]] = []
    errors: list[tuple[int, str, str, str]] = []

    def fake_run_inference(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        calls.append(inputs)
        if len(inputs) == 2:
            raise ValueError("batch boom")
        if "bad" in inputs[0]:
            raise ValueError("single boom")
        return inference_service.InferenceRunResult(
            records=[InferenceResult(score=0.25, label=0, anomaly_map=None)],
            timing_seconds=0.2,
        )

    result = infer_continue_service.run_continue_on_error_inference(
        infer_continue_service.ContinueOnErrorInferRequest(
            detector=object(),
            inputs=["a_good.png", "b_bad.png"],
            include_maps=False,
            batch_size=2,
            amp=False,
            max_errors=0,
        ),
        process_ok_result=lambda *, index, input_path, result: ok_records.append(
            (int(index), str(input_path), float(result.score))
        ),
        handle_error=lambda *, index, input_path, exc, stage: errors.append(
            (int(index), str(input_path), str(type(exc).__name__), str(stage))
        ),
        run_inference_impl=fake_run_inference,
    )

    assert calls == [["a_good.png", "b_bad.png"], ["a_good.png"], ["b_bad.png"]]
    assert ok_records == [(0, "a_good.png", 0.25)]
    assert errors == [(1, "b_bad.png", "ValueError", "infer")]
    assert result.processed == 2
    assert result.errors == 1
    assert result.stop_early is False
    assert result.timing_seconds == pytest.approx(0.2)


def test_run_continue_on_error_inference_records_artifact_stage_errors() -> None:
    errors: list[tuple[int, str, str, str]] = []
    ok_inputs: list[str] = []

    def fake_run_inference(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        return inference_service.InferenceRunResult(
            records=[
                InferenceResult(score=0.1 + i, label=0, anomaly_map=None)
                for i, _ in enumerate(inputs)
            ],
            timing_seconds=0.4,
        )

    def process_ok_result(*, index: int, input_path: str, result: Any) -> None:
        _ = result
        ok_inputs.append(str(input_path))
        if int(index) == 1:
            raise RuntimeError("artifact boom")

    result = infer_continue_service.run_continue_on_error_inference(
        infer_continue_service.ContinueOnErrorInferRequest(
            detector=object(),
            inputs=["a.png", "b.png"],
            include_maps=False,
            batch_size=2,
            amp=False,
            max_errors=0,
        ),
        process_ok_result=process_ok_result,
        handle_error=lambda *, index, input_path, exc, stage: errors.append(
            (int(index), str(input_path), str(type(exc).__name__), str(stage))
        ),
        run_inference_impl=fake_run_inference,
    )

    assert ok_inputs == ["a.png", "b.png"]
    assert errors == [(1, "b.png", "RuntimeError", "artifacts")]
    assert result.processed == 2
    assert result.errors == 1
    assert result.stop_early is False
    assert result.timing_seconds == pytest.approx(0.4)


def test_run_continue_on_error_inference_stops_early_after_max_errors() -> None:
    calls: list[list[str]] = []
    errors: list[tuple[int, str, str]] = []

    def fake_run_inference(**kwargs):
        inputs = [str(item) for item in kwargs["inputs"]]
        calls.append(inputs)
        raise ValueError(f"boom:{inputs[0]}")

    result = infer_continue_service.run_continue_on_error_inference(
        infer_continue_service.ContinueOnErrorInferRequest(
            detector=object(),
            inputs=["a.png", "b.png", "c.png"],
            include_maps=False,
            batch_size=None,
            amp=False,
            max_errors=2,
        ),
        process_ok_result=lambda *, index, input_path, result: None,
        handle_error=lambda *, index, input_path, exc, stage: errors.append(
            (int(index), str(input_path), str(stage))
        ),
        run_inference_impl=fake_run_inference,
    )

    assert calls == [["a.png"], ["a.png"], ["b.png"], ["b.png"]]
    assert errors == [(0, "a.png", "infer"), (1, "b.png", "infer")]
    assert result.processed == 2
    assert result.errors == 2
    assert result.stop_early is True
    assert result.timing_seconds == pytest.approx(0.0)
