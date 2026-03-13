from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pyimgano.evaluation import evaluate_detector
from pyimgano.inference.api import infer


@dataclass(frozen=True)
class WorkbenchInferenceResult:
    detector: Any
    scores: np.ndarray
    maps: list[np.ndarray | None] | None
    eval_results: dict[str, Any]


def _maybe_resize_maps_to_masks(maps: Sequence[np.ndarray], masks: np.ndarray) -> np.ndarray:
    import cv2

    if masks.ndim != 3:
        raise ValueError(f"Expected masks (N,H,W), got {masks.shape}")

    out: list[np.ndarray] = []
    for i, anomaly_map in enumerate(maps):
        arr = np.asarray(anomaly_map, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got {arr.shape}")
        target_h, target_w = int(masks[i].shape[0]), int(masks[i].shape[1])
        if arr.shape != (target_h, target_w):
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        out.append(np.asarray(arr, dtype=np.float32))
    return np.stack(out, axis=0)


def run_workbench_inference(
    *,
    detector: Any,
    test_inputs: Sequence[Any],
    input_format: str | None,
    postprocess: Any,
    save_maps: bool,
    test_labels: np.ndarray,
    test_masks: np.ndarray | None,
    threshold: float,
) -> WorkbenchInferenceResult:
    include_maps = bool(save_maps or (postprocess is not None))
    results = infer(
        detector,
        test_inputs,
        input_format=input_format,
        include_maps=include_maps,
        postprocess=postprocess,
    )

    scores = np.asarray([r.score for r in results], dtype=np.float32)
    maps_list = [r.anomaly_map for r in results] if include_maps else None

    pixel_scores = None
    if test_masks is not None and maps_list is not None and all(m is not None for m in maps_list):
        maps_arr = [np.asarray(m, dtype=np.float32) for m in maps_list if m is not None]
        pixel_scores = _maybe_resize_maps_to_masks(maps_arr, np.asarray(test_masks))

    eval_results = evaluate_detector(
        np.asarray(test_labels),
        scores,
        threshold=float(threshold),
        find_best_threshold=False,
        pixel_labels=test_masks,
        pixel_scores=pixel_scores,
    )

    return WorkbenchInferenceResult(
        detector=detector,
        scores=scores,
        maps=maps_list,
        eval_results=dict(eval_results),
    )


__all__ = ["WorkbenchInferenceResult", "run_workbench_inference"]
