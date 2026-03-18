from __future__ import annotations

import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from pyimgano.inference.api import InferenceResult, InferenceTiming
from pyimgano.inference.runtime_adapter import extract_maps_best_effort, score_and_maps
from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class InferenceRunResult:
    records: list[InferenceResult]
    timing_seconds: float


def _normalize_inputs(
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


def _torch_inference_context(*, amp: bool) -> ExitStack:
    stack = ExitStack()
    torch = None
    try:
        import sys

        torch = sys.modules.get("torch")
    except Exception:
        torch = None

    if torch is None:
        if not bool(amp):
            return stack
        try:
            import torch as _torch

            torch = _torch
        except Exception:
            warnings.warn(
                "AMP requested but torch is not installed; continuing without AMP.\n"
                "Install torch (and optionally CUDA) for autocast acceleration.",
                RuntimeWarning,
            )
            return stack

    try:
        stack.enter_context(torch.inference_mode())
    except Exception:
        pass

    if not bool(amp):
        return stack

    try:
        if torch.cuda.is_available():
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                try:
                    stack.enter_context(torch.amp.autocast(device_type="cuda"))
                except TypeError:
                    stack.enter_context(torch.amp.autocast("cuda"))
            else:
                stack.enter_context(torch.cuda.amp.autocast())
        else:
            warnings.warn(
                "AMP requested but CUDA is not available; continuing without autocast.",
                RuntimeWarning,
            )
    except Exception:
        warnings.warn(
            "AMP context failed to initialize; continuing without autocast.", RuntimeWarning
        )

    return stack


def _best_effort_label_confidence(
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


def _resolve_rejection_threshold(value: float | None) -> float | None:
    if value is None:
        return None
    thr = float(value)
    if not 0.0 < thr <= 1.0:
        raise ValueError(f"reject_confidence_below must be in (0, 1], got {value!r}")
    return thr


def _apply_rejection_policy(
    *,
    detector: Any,
    labels: np.ndarray | None,
    confidences: np.ndarray | None,
    reject_confidence_below: float | None,
    reject_label: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    thr = _resolve_rejection_threshold(reject_confidence_below)
    if thr is None:
        return labels, None
    if labels is None:
        raise RuntimeError("Confidence rejection requires threshold-based label predictions.")
    if confidences is None:
        raise RuntimeError("Confidence rejection requires detector confidence support.")

    labels_arr = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
    conf_arr = np.asarray(confidences, dtype=np.float64).reshape(-1)
    rejected = conf_arr < float(thr)
    marker = int(getattr(detector, "reject_label", -2) if reject_label is None else reject_label)
    labels_arr[rejected] = marker
    return labels_arr, rejected.astype(bool)


def _build_decision_summary(
    *,
    label: int | None,
    label_confidence: float | None,
    rejected: bool | None,
) -> dict[str, Any]:
    rejected_flag = bool(rejected)
    threshold_applied = label is not None
    has_confidence = label_confidence is not None

    if rejected_flag:
        decision = "rejected_low_confidence"
        requires_review = True
    elif label is None:
        decision = "score_only"
        requires_review = False
    elif int(label) == 0:
        decision = "normal"
        requires_review = False
    else:
        decision = "anomalous"
        requires_review = True

    return {
        "decision": str(decision),
        "threshold_applied": bool(threshold_applied),
        "has_confidence": bool(has_confidence),
        "rejected": bool(rejected_flag),
        "requires_review": bool(requires_review),
    }


def iter_inference_records(
    *,
    detector: Any,
    inputs: Sequence[ImageInput],
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    batch_size: int | None = None,
    amp: bool = False,
    timing: InferenceTiming | None = None,
):
    normalized = _normalize_inputs(inputs, input_format=input_format, u16_max=u16_max)
    threshold = getattr(detector, "threshold_", None)
    rejection_threshold = _resolve_rejection_threshold(reject_confidence_below)

    bs: int | None
    if batch_size is None:
        bs = None
    else:
        bsv = int(batch_size)
        bs = bsv if bsv > 0 else None

    def _yield_chunk(chunk: Sequence[Any]):
        t0 = time.perf_counter()
        with _torch_inference_context(amp=bool(amp)):
            scores, maps_from_runtime = score_and_maps(detector, chunk, include_maps=include_maps)

            labels: Optional[np.ndarray]
            if threshold is not None:
                labels = (scores >= float(threshold)).astype(int)
            else:
                labels = None

            confidences: np.ndarray | None = None
            if bool(include_confidence) or rejection_threshold is not None:
                confidences = _best_effort_label_confidence(
                    detector,
                    chunk,
                    scores=np.asarray(scores, dtype=np.float64).reshape(-1),
                    labels=(
                        None
                        if labels is None
                        else np.asarray(labels, dtype=np.int64).reshape(-1)
                    ),
                )
            labels, rejected = _apply_rejection_policy(
                detector=detector,
                labels=labels,
                confidences=confidences,
                reject_confidence_below=rejection_threshold,
                reject_label=reject_label,
            )

            maps: list[np.ndarray | None] | None = None
            if include_maps:
                extracted: list[np.ndarray | None] | None
                if maps_from_runtime is not None:
                    extracted = [
                        np.asarray(maps_from_runtime[i], dtype=np.float32)
                        for i in range(int(scores.shape[0]))
                    ]
                else:
                    extracted = extract_maps_best_effort(detector, chunk)

                if extracted is not None:
                    maps = []
                    for m in extracted:
                        if m is None:
                            maps.append(None)
                            continue
                        processed = m
                        if postprocess is not None:
                            processed = postprocess(processed)
                        maps.append(np.asarray(processed, dtype=np.float32))

        if timing is not None:
            timing.seconds += time.perf_counter() - t0

        for i in range(int(scores.shape[0])):
            label = int(labels[i]) if labels is not None else None
            label_confidence = None if confidences is None else float(confidences[i])
            rejected_flag = None if rejected is None else bool(rejected[i])
            anomaly_map = maps[i] if maps is not None else None
            yield InferenceResult(
                score=float(scores[i]),
                label=label,
                label_confidence=label_confidence,
                rejected=rejected_flag,
                anomaly_map=anomaly_map,
                decision_summary=_build_decision_summary(
                    label=label,
                    label_confidence=label_confidence,
                    rejected=rejected_flag,
                ),
            )

    if bs is None or bs >= len(normalized):
        yield from _yield_chunk(normalized)
        return

    for start in range(0, len(normalized), int(bs)):
        chunk = normalized[start : start + int(bs)]
        yield from _yield_chunk(chunk)


def run_inference(
    *,
    detector: Any,
    inputs: Sequence[ImageInput],
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    batch_size: int | None = None,
    amp: bool = False,
) -> InferenceRunResult:
    """Run inference and return structured records without any CLI side effects."""

    timing = InferenceTiming()
    records = list(
        iter_inference_records(
            detector=detector,
            inputs=inputs,
            input_format=input_format,
            u16_max=u16_max,
            include_maps=include_maps,
            include_confidence=include_confidence,
            reject_confidence_below=reject_confidence_below,
            reject_label=reject_label,
            postprocess=postprocess,
            batch_size=batch_size,
            amp=amp,
            timing=timing,
        )
    )
    return InferenceRunResult(records=records, timing_seconds=float(timing.seconds))


__all__ = [
    "InferenceRunResult",
    "iter_inference_records",
    "run_inference",
]
