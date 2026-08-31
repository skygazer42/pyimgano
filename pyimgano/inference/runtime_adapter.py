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


def _call_score_and_maps_hook(
    detector: Any,
    inputs: Sequence[Any],
    *,
    include_maps: bool,
) -> Any | None:
    hook = getattr(detector, "score_and_maps", None)
    if not callable(hook):
        return None
    try:
        return hook(inputs, include_maps=bool(include_maps))
    except TypeError as exc:
        # Compatibility for early detector hooks that accepted only the batch.
        try:
            return hook(inputs)
        except TypeError:
            raise exc


def _normalize_hook_maps(maps: Any, *, n_expected: int) -> np.ndarray | list[np.ndarray] | None:
    if maps is None:
        return None
    if isinstance(maps, (list, tuple)):
        if len(maps) != n_expected:
            raise ValueError(f"Expected {n_expected} anomaly maps, got {len(maps)}.")
        normalized = [np.asarray(item, dtype=np.float32) for item in maps]
        for item in normalized:
            if item.ndim != 2:
                raise ValueError(
                    f"Expected each anomaly map to be 2D, got shape {tuple(item.shape)}."
                )
        if len({tuple(item.shape) for item in normalized}) > 1:
            return normalized
    return normalize_anomaly_maps(maps, n_expected=n_expected)


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


def extract_maps_best_effort(
    detector: Any, inputs: Sequence[Any]
) -> list[np.ndarray | None] | None:
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
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]:
    n_expected = int(len(inputs))
    if n_expected == 0:
        return np.zeros((0,), dtype=np.float32), None

    hooked = _call_score_and_maps_hook(
        detector,
        inputs,
        include_maps=bool(include_maps),
    )
    if hooked is not None:
        if not isinstance(hooked, (tuple, list)) or len(hooked) != 2:
            raise TypeError("detector.score_and_maps() must return (scores, maps).")
        scores_any, maps_any = hooked
        scores = normalize_scores(scores_any, n_expected=n_expected)
        maps = _normalize_hook_maps(maps_any, n_expected=n_expected) if include_maps else None
        return scores, maps

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
