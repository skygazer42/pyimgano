from __future__ import annotations

import warnings
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np

from pyimgano.inputs.image_format import ImageFormat, normalize_numpy_image, parse_image_format
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

ImageInput = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class InferenceResult:
    score: float
    label: Optional[int] = None
    anomaly_map: Optional[np.ndarray] = None


def _normalize_inputs(
    inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None,
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
            out_arr.append(normalize_numpy_image(item, input_format=fmt))
        return out_arr

    raise TypeError(f"Unsupported input type: {type(first)}. Expected str|Path|np.ndarray.")


def calibrate_threshold(
    detector: Any,
    calibration_inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    quantile: float = 0.995,
    amp: bool = False,
) -> float:
    """Calibrate `detector.threshold_` by a score quantile on normal samples."""

    if not 0.0 < float(quantile) < 1.0:
        raise ValueError(f"quantile must be in (0,1), got {quantile}")

    normalized = _normalize_inputs(calibration_inputs, input_format=input_format)
    with _torch_inference_context(amp=bool(amp)):
        scores = _call_decision_function(detector, normalized)
    if scores.size == 0:
        raise ValueError("No scores produced for calibration inputs.")

    threshold = float(np.quantile(scores, float(quantile)))
    setattr(detector, "threshold_", threshold)
    return threshold


def _torch_inference_context(*, amp: bool) -> ExitStack:
    """Best-effort torch inference/autocast context.

    - If torch is missing, yields without error (and warns when amp=True).
    - If CUDA is unavailable, runs without autocast (and warns when amp=True).
    """

    stack = ExitStack()
    if not bool(amp):
        return stack

    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        warnings.warn(
            "AMP requested but torch is not installed; continuing without AMP.\n"
            "Install torch (and optionally CUDA) for autocast acceleration.",
            RuntimeWarning,
        )
        return stack

    stack.enter_context(torch.inference_mode())

    try:
        if torch.cuda.is_available():
            stack.enter_context(torch.cuda.amp.autocast())
        else:
            warnings.warn(
                "AMP requested but CUDA is not available; continuing without autocast.",
                RuntimeWarning,
            )
    except Exception:
        # Best-effort: never fail inference due to AMP wrapper issues.
        warnings.warn("AMP context failed to initialize; continuing without autocast.", RuntimeWarning)

    return stack


def _call_decision_function(detector: Any, inputs: Sequence[Any]) -> np.ndarray:
    """Best-effort wrapper around `decision_function` for list-vs-batch conventions."""

    try:
        return np.asarray(detector.decision_function(inputs), dtype=np.float32)
    except Exception as exc:
        # Some detectors expect an ndarray batch (N,H,W,C) instead of a list of arrays.
        if inputs and isinstance(inputs[0], np.ndarray):
            try:
                batch = np.stack([np.asarray(x) for x in inputs], axis=0)
            except Exception:
                raise exc
            try:
                return np.asarray(detector.decision_function(batch), dtype=np.float32)
            except Exception:
                raise exc
        raise


def _try_get_maps(detector: Any, inputs: Sequence[Any]) -> list[np.ndarray | None] | None:
    if hasattr(detector, "predict_anomaly_map"):
        maps = None
        try:
            maps = detector.predict_anomaly_map(inputs)
        except Exception:
            # Some detectors expect a batched ndarray for predict_anomaly_map.
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


def infer(
    detector: Any,
    inputs: Sequence[ImageInput],
    *,
    input_format: str | ImageFormat | None = None,
    include_maps: bool = False,
    postprocess: AnomalyMapPostprocess | None = None,
    batch_size: int | None = None,
    amp: bool = False,
) -> list[InferenceResult]:
    """Run `decision_function` + optional anomaly-map extraction.

    - For numpy inputs, `input_format` is required and images are normalized to canonical RGB/u8/HWC.
    - For path inputs, `input_format` is ignored and paths are forwarded as-is.
    """

    normalized = _normalize_inputs(inputs, input_format=input_format)
    threshold = getattr(detector, "threshold_", None)

    bs: int | None
    if batch_size is None:
        bs = None
    else:
        bsv = int(batch_size)
        bs = (bsv if bsv > 0 else None)

    def _infer_chunk(chunk: Sequence[Any]) -> list[InferenceResult]:
        with _torch_inference_context(amp=bool(amp)):
            scores = _call_decision_function(detector, chunk)

            labels: Optional[np.ndarray]
            if threshold is not None:
                labels = (scores >= float(threshold)).astype(int)
            else:
                labels = None

            maps: list[np.ndarray | None] | None = None
            if include_maps:
                extracted = _try_get_maps(detector, chunk)
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

        out: list[InferenceResult] = []
        for i in range(int(scores.shape[0])):
            label = int(labels[i]) if labels is not None else None
            anomaly_map = maps[i] if maps is not None else None
            out.append(InferenceResult(score=float(scores[i]), label=label, anomaly_map=anomaly_map))
        return out

    if bs is None or bs >= len(normalized):
        return _infer_chunk(normalized)

    results: list[InferenceResult] = []
    for start in range(0, len(normalized), int(bs)):
        chunk = normalized[start : start + int(bs)]
        results.extend(_infer_chunk(chunk))
    return results


def result_to_jsonable(
    result: InferenceResult,
    *,
    anomaly_map_path: str | None = None,
    include_anomaly_map_values: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"score": float(result.score)}
    if result.label is not None:
        payload["label"] = int(result.label)

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
