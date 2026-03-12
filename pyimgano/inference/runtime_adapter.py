from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


def _call_decision_function_best_effort(detector: Any, inputs: Sequence[Any]) -> Any:
    try:
        return detector.decision_function(inputs)
    except Exception as exc:
        if inputs and isinstance(inputs[0], np.ndarray):
            try:
                batch = np.stack([np.asarray(x) for x in inputs], axis=0)
            except Exception:
                raise exc
            try:
                return detector.decision_function(batch)
            except Exception:
                raise exc
        raise


def extract_maps_best_effort(detector: Any, inputs: Sequence[Any]) -> list[np.ndarray | None] | None:
    if hasattr(detector, "predict_anomaly_map"):
        maps = None
        try:
            maps = detector.predict_anomaly_map(inputs)
        except Exception:
            if inputs and isinstance(inputs[0], np.ndarray):
                try:
                    batch = np.stack([np.asarray(x) for x in inputs], axis=0)
                except Exception:
                    maps = None
                else:
                    try:
                        maps = detector.predict_anomaly_map(batch)
                    except Exception:
                        maps = None
        if maps is not None:
            arr = np.asarray(maps)
            if arr.ndim == 3 and arr.shape[0] == len(inputs):
                return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]

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
