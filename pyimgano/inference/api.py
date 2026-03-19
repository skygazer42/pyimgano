from __future__ import annotations

import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union, cast

import numpy as np

from pyimgano.inference.runtime_adapter import extract_maps_best_effort, score_and_maps
from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class InferenceResult:
    score: float
    label: Optional[int] = None
    label_confidence: Optional[float] = None
    rejected: Optional[bool] = None
    anomaly_map: Optional[np.ndarray] = None
    postprocess_summary: dict[str, Any] | None = None
    decision_summary: dict[str, Any] | None = None


@dataclass
class InferenceTiming:
    """Accumulate best-effort wallclock inference time (seconds)."""

    seconds: float = 0.0


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
        # Path-based callers: keep as strings, let detectors handle loading.
        out = []
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


def calibrate_threshold(
    detector: Any,
    calibration_inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    quantile: float = 0.995,
    batch_size: int | None = None,
    amp: bool = False,
) -> float:
    """Calibrate `detector.threshold_` by a score quantile on normal samples."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must be in (0,1), got {quantile}")

    scores = collect_calibration_scores(
        detector,
        calibration_inputs,
        input_format=input_format,
        u16_max=u16_max,
        batch_size=batch_size,
        amp=amp,
    )
    threshold = float(np.quantile(scores, float(quantile)))
    setattr(detector, "threshold_", threshold)
    return threshold


def collect_calibration_scores(
    detector: Any,
    calibration_inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    batch_size: int | None = None,
    amp: bool = False,
) -> np.ndarray:
    """Collect raw calibration scores on normal samples without mutating the detector."""

    normalized = _normalize_inputs(
        calibration_inputs,
        input_format=input_format,
        u16_max=u16_max,
    )
    if not normalized:
        raise ValueError("calibration_inputs must be non-empty.")

    bs: int | None
    if batch_size is None:
        bs = None
    else:
        bsv = int(batch_size)
        bs = bsv if bsv > 0 else None

    chunks = [normalized] if bs is None or bs >= len(normalized) else []
    if not chunks:
        assert bs is not None
        for start in range(0, len(normalized), bs):
            chunks.append(normalized[start : start + bs])

    scores_all: list[np.ndarray] = []
    with _torch_inference_context(amp=bool(amp)):
        for chunk in chunks:
            scores, _maps = score_and_maps(detector, chunk, include_maps=False)
            scores_all.append(scores)
    scores = np.concatenate(scores_all, axis=0)
    if scores.size == 0:
        raise ValueError("No scores produced for calibration inputs.")
    return np.asarray(scores, dtype=np.float64).reshape(-1)


def _torch_inference_context(*, amp: bool) -> ExitStack:
    """Best-effort torch inference/autocast context.

    - If torch is missing, yields without error (and warns when amp=True).
    - If CUDA is unavailable, runs without autocast (and warns when amp=True).
    """

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

    # Always disable grad for torch-backed detectors (best-effort).
    try:
        stack.enter_context(torch.inference_mode())
    except Exception:
        # Best-effort: keep going even if the context manager is unavailable/broken.
        pass

    if not bool(amp):
        return stack

    try:
        if torch.cuda.is_available():
            # Prefer the newer API: torch.amp.autocast(device_type="cuda")
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
        # Best-effort: never fail inference due to AMP wrapper issues.
        warnings.warn(
            "AMP context failed to initialize; continuing without autocast.", RuntimeWarning
        )

    return stack


def _call_decision_function(detector: Any, inputs: Sequence[Any]) -> np.ndarray:
    """Best-effort wrapper around `decision_function` for list-vs-batch conventions."""

    scores, _maps = score_and_maps(detector, inputs, include_maps=False)
    return cast(np.ndarray, np.asarray(scores, dtype=np.float32))


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
        review_reason = "low_confidence"
    elif label is None:
        decision = "score_only"
        requires_review = False
        review_reason = "unthresholded_score"
    elif int(label) == 0:
        decision = "normal"
        requires_review = False
        review_reason = "none"
    else:
        decision = "anomalous"
        requires_review = True
        review_reason = "anomaly_label"

    return {
        "decision": str(decision),
        "threshold_applied": bool(threshold_applied),
        "has_confidence": bool(has_confidence),
        "rejected": bool(rejected_flag),
        "requires_review": bool(requires_review),
        "review_reason": str(review_reason),
    }


def _maybe_build_decision_summary(
    *,
    label: int | None,
    label_confidence: float | None,
    rejected: bool | None,
) -> dict[str, Any] | None:
    if label is None and label_confidence is None and rejected is None:
        return None
    return _build_decision_summary(
        label=label,
        label_confidence=label_confidence,
        rejected=rejected,
    )


def infer(
    detector: Any,
    inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    postprocess_summary: dict[str, Any] | None = None,
    batch_size: int | None = None,
    amp: bool = False,
) -> list[InferenceResult]:
    """Run `decision_function` + optional anomaly-map extraction.

    - For numpy inputs, `input_format` is required and images are normalized to canonical RGB/u8/HWC.
    - For path inputs, `input_format` is ignored and paths are forwarded as-is.
    """

    return list(
        infer_iter(
            detector,
            inputs,
            input_format=input_format,
            u16_max=u16_max,
            include_maps=include_maps,
            include_confidence=include_confidence,
            reject_confidence_below=reject_confidence_below,
            reject_label=reject_label,
            postprocess=postprocess,
            postprocess_summary=postprocess_summary,
            batch_size=batch_size,
            amp=amp,
        )
    )


def infer_bgr(
    detector: Any,
    inputs: Sequence[ImageInput],
    *,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    postprocess_summary: dict[str, Any] | None = None,
    batch_size: int | None = None,
    amp: bool = False,
) -> list[InferenceResult]:
    """Convenience wrapper for the most common OpenCV numpy input: `bgr_u8_hwc`.

    This keeps the strict "explicit ImageFormat" contract while making the
    common case ergonomic.
    """

    return infer(
        detector,
        inputs,
        input_format=ImageFormat.BGR_U8_HWC,
        include_maps=include_maps,
        include_confidence=include_confidence,
        reject_confidence_below=reject_confidence_below,
        reject_label=reject_label,
        postprocess=postprocess,
        postprocess_summary=postprocess_summary,
        batch_size=batch_size,
        amp=amp,
    )


def infer_iter_bgr(
    detector: Any,
    inputs: Sequence[ImageInput],
    *,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    postprocess_summary: dict[str, Any] | None = None,
    batch_size: int | None = None,
    amp: bool = False,
    timing: InferenceTiming | None = None,
):
    """Iterator convenience wrapper for `bgr_u8_hwc` numpy inputs."""

    return infer_iter(
        detector,
        inputs,
        input_format=ImageFormat.BGR_U8_HWC,
        include_maps=include_maps,
        include_confidence=include_confidence,
        reject_confidence_below=reject_confidence_below,
        reject_label=reject_label,
        postprocess=postprocess,
        postprocess_summary=postprocess_summary,
        batch_size=batch_size,
        amp=amp,
        timing=timing,
    )


def calibrate_threshold_bgr(
    detector: Any,
    calibration_inputs: Sequence[ImageInput],
    *,
    quantile: float = 0.995,
    batch_size: int | None = None,
    amp: bool = False,
) -> float:
    """Convenience wrapper for calibrating from `bgr_u8_hwc` numpy images."""

    return calibrate_threshold(
        detector,
        calibration_inputs,
        input_format=ImageFormat.BGR_U8_HWC,
        quantile=quantile,
        batch_size=batch_size,
        amp=amp,
    )


def infer_iter(
    detector: Any,
    inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    u16_max: int | None = None,
    include_maps: bool = False,
    include_confidence: bool = False,
    reject_confidence_below: float | None = None,
    reject_label: int | None = None,
    postprocess: AnomalyMapPostprocess | None = None,
    postprocess_summary: dict[str, Any] | None = None,
    batch_size: int | None = None,
    amp: bool = False,
    timing: InferenceTiming | None = None,
):
    """Iterator version of :func:`infer` (doesn't store all outputs).

    This is recommended for large runs with anomaly maps, since keeping per-image
    float32 maps in memory can be expensive.
    """

    normalized = _normalize_inputs(inputs, input_format=input_format, u16_max=u16_max)
    threshold = getattr(detector, "threshold_", None)
    rejection_threshold = _resolve_rejection_threshold(reject_confidence_below)
    summary_payload = (
        dict(postprocess_summary) if postprocess_summary is not None else None
    )

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
                postprocess_summary=summary_payload,
                decision_summary=_maybe_build_decision_summary(
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


def result_to_jsonable(
    result: InferenceResult,
    *,
    anomaly_map_path: str | None = None,
    include_anomaly_map_values: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"score": float(result.score)}
    if result.label is not None:
        payload["label"] = int(result.label)
    if result.label_confidence is not None:
        payload["label_confidence"] = float(result.label_confidence)
    if result.rejected is not None:
        payload["rejected"] = bool(result.rejected)
    if result.postprocess_summary is not None:
        payload["postprocess_summary"] = dict(result.postprocess_summary)
    if result.decision_summary is not None:
        payload["decision_summary"] = dict(result.decision_summary)

    if result.anomaly_map is not None:
        meta: dict[str, Any] = {
            "shape": [int(d) for d in result.anomaly_map.shape],
            "dtype": str(result.anomaly_map.dtype),
        }
        if anomaly_map_path is not None:
            meta["path"] = str(anomaly_map_path)
        payload["anomaly_map"] = meta

        if include_anomaly_map_values:
            payload["anomaly_map_values"] = result.anomaly_map.tolist()

    return payload


def results_to_jsonable(
    results: Sequence[InferenceResult],
    *,
    anomaly_map_paths: Sequence[str | None] | None = None,
    include_anomaly_map_values: bool = False,
) -> list[dict[str, Any]]:
    if anomaly_map_paths is not None and len(anomaly_map_paths) != len(results):
        raise ValueError("anomaly_map_paths length must match results length")

    out: list[dict[str, Any]] = []
    for i, result in enumerate(results):
        path = anomaly_map_paths[i] if anomaly_map_paths is not None else None
        out.append(
            result_to_jsonable(
                result,
                anomaly_map_path=path,
                include_anomaly_map_values=include_anomaly_map_values,
            )
        )
    return out
