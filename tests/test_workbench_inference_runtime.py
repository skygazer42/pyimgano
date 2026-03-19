from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyimgano.workbench.inference_runtime import run_workbench_inference


def test_workbench_inference_runtime_resizes_maps_for_pixel_evaluation(monkeypatch) -> None:
    import pyimgano.workbench.inference_runtime as inference_runtime

    calls: dict[str, object] = {}

    def _fake_infer(
        detector, test_inputs, *, input_format=None, include_maps=False, postprocess=None
    ):  # noqa: ANN001
        calls["infer"] = {
            "detector": detector,
            "test_inputs": list(test_inputs),
            "input_format": input_format,
            "include_maps": include_maps,
            "postprocess": postprocess,
        }
        return [SimpleNamespace(score=0.7, anomaly_map=np.ones((2, 2), dtype=np.float32))]

    def _fake_evaluate_detector(
        test_labels,
        scores,
        *,
        threshold,
        find_best_threshold,
        pixel_labels=None,
        pixel_scores=None,
    ):  # noqa: ANN001
        calls["evaluate"] = {
            "test_labels": np.asarray(test_labels),
            "scores": np.asarray(scores),
            "threshold": threshold,
            "find_best_threshold": find_best_threshold,
            "pixel_labels": None if pixel_labels is None else np.asarray(pixel_labels),
            "pixel_scores": None if pixel_scores is None else np.asarray(pixel_scores),
        }
        return {"threshold": threshold, "auroc": 1.0}

    monkeypatch.setattr(inference_runtime, "infer", _fake_infer)
    monkeypatch.setattr(inference_runtime, "evaluate_detector", _fake_evaluate_detector)

    outputs = run_workbench_inference(
        detector=object(),
        test_inputs=["sample.png"],
        input_format="rgb_u8_hwc",
        postprocess=object(),
        save_maps=False,
        test_labels=np.asarray([1], dtype=np.int64),
        test_masks=np.ones((1, 4, 4), dtype=np.uint8),
        threshold=0.5,
    )

    assert calls["infer"] == {
        "detector": outputs.detector,
        "test_inputs": ["sample.png"],
        "input_format": "rgb_u8_hwc",
        "include_maps": True,
        "postprocess": calls["infer"]["postprocess"],
    }
    assert np.isclose(calls["evaluate"]["threshold"], 0.5)
    assert calls["evaluate"]["find_best_threshold"] is False
    assert calls["evaluate"]["pixel_scores"].shape == (1, 4, 4)
    assert np.allclose(np.asarray(outputs.scores), np.asarray([0.7], dtype=np.float32))
    assert outputs.maps is not None
    assert outputs.eval_results == {"threshold": 0.5, "auroc": 1.0}


def test_workbench_inference_runtime_skips_map_collection_when_not_needed(monkeypatch) -> None:
    import pyimgano.workbench.inference_runtime as inference_runtime

    calls: dict[str, object] = {}

    def _fake_infer(
        detector, test_inputs, *, input_format=None, include_maps=False, postprocess=None
    ):  # noqa: ANN001
        del detector, test_inputs, input_format, postprocess
        calls["include_maps"] = include_maps
        return [SimpleNamespace(score=0.2, anomaly_map=None)]

    def _fake_evaluate_detector(
        test_labels,
        scores,
        *,
        threshold,
        find_best_threshold,
        pixel_labels=None,
        pixel_scores=None,
    ):  # noqa: ANN001
        del test_labels, scores, find_best_threshold, pixel_labels
        calls["pixel_scores"] = pixel_scores
        return {"threshold": threshold, "average_precision": 0.6}

    monkeypatch.setattr(inference_runtime, "infer", _fake_infer)
    monkeypatch.setattr(inference_runtime, "evaluate_detector", _fake_evaluate_detector)

    outputs = run_workbench_inference(
        detector=object(),
        test_inputs=["sample.png"],
        input_format=None,
        postprocess=None,
        save_maps=False,
        test_labels=np.asarray([0], dtype=np.int64),
        test_masks=None,
        threshold=0.2,
    )

    assert calls["include_maps"] is False
    assert calls["pixel_scores"] is None
    assert outputs.maps is None
    assert np.allclose(np.asarray(outputs.scores), np.asarray([0.2], dtype=np.float32))
    assert outputs.eval_results == {"threshold": 0.2, "average_precision": 0.6}
