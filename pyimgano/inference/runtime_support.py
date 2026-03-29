from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format

ImageInput = Union[str, Path, np.ndarray]


def normalize_inputs(
    inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None,
    u16_max: int | None = None,
) -> list[str] | list[np.ndarray]:
    if not inputs:
        return []

    first = inputs[0]
    if isinstance(first, (str, Path)):
        out: list[str] = []
        for item in inputs:
            if not isinstance(item, (str, Path)):
                raise TypeError("Mixed input types are not supported (paths + arrays).")
            out.append(str(item))
        return out

    if isinstance(first, np.ndarray):
        if input_format is None:
            raise ValueError("input_format is required when passing numpy images.")
        fmt = parse_image_format(input_format)
        out_arr: list[np.ndarray] = []
        for item in inputs:
            if not isinstance(item, np.ndarray):
                raise TypeError("Mixed input types are not supported (paths + arrays).")
            out_arr.append(normalize_numpy_image(item, input_format=fmt, u16_max=u16_max))
        return out_arr

    raise TypeError(f"Unsupported input type: {type(first)}. Expected str|Path|np.ndarray.")


def best_effort_label_confidence(
    detector: Any,
    inputs: Sequence[Any],
    *,
    scores: np.ndarray,
    labels: np.ndarray | None,
) -> np.ndarray | None:
    helper = getattr(detector, "_label_confidence_from_scores", None)
    if callable(helper):
        try:
            conf = helper(scores, labels=labels)
            arr = np.asarray(conf, dtype=np.float64).reshape(-1)
            if arr.shape[0] == int(scores.shape[0]):
                return np.clip(arr, 0.0, 1.0)
        except Exception:
            pass

    predictor = getattr(detector, "predict_confidence", None)
    if callable(predictor):
        try:
            conf = predictor(inputs)
            arr = np.asarray(conf, dtype=np.float64).reshape(-1)
            if arr.shape[0] == int(scores.shape[0]):
                return np.clip(arr, 0.0, 1.0)
        except Exception:
            return None

    return None


def resolve_rejection_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"reject_confidence_below must be in (0, 1], got {value!r}")
    return threshold


def apply_rejection_policy(
    *,
    detector: Any,
    labels: np.ndarray | None,
    confidences: np.ndarray | None,
    reject_confidence_below: float | None,
    reject_label: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    threshold = resolve_rejection_threshold(reject_confidence_below)
    if threshold is None:
        return labels, None
    if labels is None:
        raise RuntimeError("Confidence rejection requires threshold-based label predictions.")
    if confidences is None:
        raise RuntimeError("Confidence rejection requires detector confidence support.")

    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    conf_arr = np.asarray(confidences, dtype=np.float64).reshape(-1)
    rejected = conf_arr < float(threshold)
    marker = int(getattr(detector, "reject_label", -2) if reject_label is None else reject_label)
    labels_arr[rejected] = marker
    return labels_arr, rejected.astype(bool)


__all__ = [
    "ImageInput",
    "apply_rejection_policy",
    "best_effort_label_confidence",
    "normalize_inputs",
    "resolve_rejection_threshold",
]
