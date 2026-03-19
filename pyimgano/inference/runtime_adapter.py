from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


def _stack_numpy_inputs(inputs: Sequence[Any]) -> np.ndarray | None:
    if not inputs or not isinstance(inputs[0], np.ndarray):
        return None
    try:
        return np.stack([np.asarray(x) for x in inputs], axis=0)
    except Exception:
        return None


def _call_with_numpy_batch_fallback(callable_obj: Any, inputs: Sequence[Any]) -> Any:
    try:
        return callable_obj(inputs)
    except Exception as exc:
        batch = _stack_numpy_inputs(inputs)
        if batch is None:
            raise
        try:
            return callable_obj(batch)
        except Exception:
            raise exc


def _call_decision_function_best_effort(detector: Any, inputs: Sequence[Any]) -> Any:
    return _call_with_numpy_batch_fallback(detector.decision_function, inputs)


def _normalize_extracted_maps(
    maps: Any,
    *,
    n_expected: int,
) -> list[np.ndarray | None] | None:
    try:
        normalized = normalize_anomaly_maps(maps, n_expected=n_expected)
    except Exception:
        return None
    return [np.asarray(normalized[i], dtype=np.float32) for i in range(normalized.shape[0])]


def extract_maps_best_effort(detector: Any, inputs: Sequence[Any]) -> list[np.ndarray | None] | None:
    if hasattr(detector, "predict_anomaly_map"):
        try:
            maps = _call_with_numpy_batch_fallback(detector.predict_anomaly_map, inputs)
        except Exception:
            maps = None
        normalized = _normalize_extracted_maps(maps, n_expected=len(inputs))
        if normalized is not None:
            return normalized

    if hasattr(detector, "get_anomaly_map"):
        out: list[np.ndarray | None] = []
        for item in inputs:
            try:
                out.append(np.asarray(detector.get_anomaly_map(item), dtype=np.float32))
            except Exception:
                out.append(None)
        if any(m is not None for m in out):
            return out

    return None


def score_and_maps(
    detector: Any,
    inputs: Sequence[Any],
    *,
    include_maps: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    n_expected = int(len(inputs))
    if n_expected == 0:
        return np.zeros((0,), dtype=np.float32), None

    out = _call_decision_function_best_effort(detector, inputs)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        scores_any, maps_any = out
        scores = normalize_scores(scores_any, n_expected=n_expected)
        maps = normalize_anomaly_maps(maps_any, n_expected=n_expected)
        return scores, maps

    scores = normalize_scores(out, n_expected=n_expected)
    if not include_maps:
        return scores, None

    extracted = extract_maps_best_effort(detector, inputs)
    if extracted is None or any(m is None for m in extracted):
        return scores, None

    maps = normalize_anomaly_maps([m for m in extracted if m is not None], n_expected=n_expected)
    return scores, maps


__all__ = ["extract_maps_best_effort", "score_and_maps"]
