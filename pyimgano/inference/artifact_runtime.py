from __future__ import annotations

import math
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pyimgano.inference.image_decode import load_path_as_rgb_u8_hwc
from pyimgano.models.protocols import normalize_anomaly_maps, normalize_scores


class ArtifactRuntimeError(RuntimeError):
    """Raised when a validated artifact cannot satisfy its runtime contract."""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _decode_rgb_image(item: Any) -> np.ndarray:
    if isinstance(item, (str, Path)):
        return np.asarray(load_path_as_rgb_u8_hwc(item), dtype=np.uint8)
    if not isinstance(item, np.ndarray):
        raise TypeError(
            "Artifact runtimes accept path inputs or canonical RGB uint8/HWC numpy images; "
            f"got {type(item).__name__}."
        )
    image = np.asarray(item)
    if image.dtype != np.uint8:
        raise TypeError(
            "ArtifactRuntime numpy inputs must already be canonical RGB uint8/HWC. "
            "Use pyimgano.inference.infer(..., input_format=...) for other formats."
        )
    if image.ndim != 3 or int(image.shape[2]) != 3:
        raise ValueError(
            f"ArtifactRuntime numpy inputs must have shape (H, W, 3); got {tuple(image.shape)}."
        )
    return np.ascontiguousarray(image)


def _pil_interpolation(name: str) -> int:
    from PIL import Image

    resampling = getattr(Image, "Resampling", Image)
    values = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
        "area": resampling.BOX,
    }
    key = str(name).strip().lower()
    if key not in values:
        raise ArtifactRuntimeError(f"Unsupported resize interpolation: {name!r}")
    return int(values[key])


def _resize_rgb(image: np.ndarray, contract: Mapping[str, Any]) -> np.ndarray:
    size = contract.get("size")
    if size is None:
        return np.ascontiguousarray(image)
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ArtifactRuntimeError("input_contract.size must be [height, width] or null.")
    target_h, target_w = int(size[0]), int(size[1])
    if target_h <= 0 or target_w <= 0:
        raise ArtifactRuntimeError("input_contract.size dimensions must be positive.")

    from PIL import Image

    resize = _mapping(contract.get("resize"))
    mode = str(resize.get("mode", "stretch")).strip().lower()
    interpolation = _pil_interpolation(str(resize.get("interpolation", "bilinear")))
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    if mode == "stretch":
        return np.asarray(pil.resize((target_w, target_h), resample=interpolation), dtype=np.uint8)

    source_h, source_w = int(image.shape[0]), int(image.shape[1])
    if mode in {"letterbox", "pad"}:
        scale = min(target_w / float(source_w), target_h / float(source_h))
        resized_w = max(1, int(round(source_w * scale)))
        resized_h = max(1, int(round(source_h * scale)))
        resized = np.asarray(
            pil.resize((resized_w, resized_h), resample=interpolation), dtype=np.uint8
        )
        fill = resize.get("fill", [0, 0, 0])
        if isinstance(fill, (int, float)):
            fill = [int(fill)] * 3
        if not isinstance(fill, (list, tuple)) or len(fill) != 3:
            raise ArtifactRuntimeError("letterbox resize.fill must be a scalar or RGB triplet.")
        canvas = np.empty((target_h, target_w, 3), dtype=np.uint8)
        canvas[...] = np.asarray([int(v) for v in fill], dtype=np.uint8)
        top = (target_h - resized_h) // 2
        left = (target_w - resized_w) // 2
        canvas[top : top + resized_h, left : left + resized_w] = resized
        return canvas

    if mode in {"center_crop", "shorter_side_center_crop"}:
        scale = max(target_w / float(source_w), target_h / float(source_h))
        resized_w = max(target_w, int(round(source_w * scale)))
        resized_h = max(target_h, int(round(source_h * scale)))
        resized = np.asarray(
            pil.resize((resized_w, resized_h), resample=interpolation), dtype=np.uint8
        )
        top = (resized_h - target_h) // 2
        left = (resized_w - target_w) // 2
        return np.ascontiguousarray(resized[top : top + target_h, left : left + target_w])

    raise ArtifactRuntimeError(f"Unsupported resize mode: {mode!r}")


def prepare_image_batch(
    inputs: Sequence[Any], input_contract: Mapping[str, Any]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Apply a manifest tensor contract exactly once to raw image inputs."""

    if not inputs:
        dtype = np.dtype(str(input_contract.get("dtype", "float32")))
        return np.empty((0,), dtype=dtype), []

    layout = str(input_contract.get("layout", "NCHW")).strip().upper()
    if layout not in {"NCHW", "NHWC"}:
        raise ArtifactRuntimeError(f"Unsupported input layout: {layout!r}")
    color = str(input_contract.get("color_space", "RGB")).strip().upper()
    if color not in {"RGB", "BGR", "GRAY"}:
        raise ArtifactRuntimeError(f"Unsupported input color space: {color!r}")

    scale = _mapping(input_contract.get("scale"))
    divisor = float(scale.get("divisor", 1.0))
    multiplier = float(scale.get("multiplier", 1.0))
    offset = float(scale.get("offset", 0.0))
    if not math.isfinite(divisor) or divisor == 0.0:
        raise ArtifactRuntimeError("input_contract.scale.divisor must be finite and non-zero.")

    normalize = _mapping(input_contract.get("normalize"))
    mean_raw = normalize.get("mean")
    std_raw = normalize.get("std")
    mean = None if mean_raw is None else np.asarray(mean_raw, dtype=np.float32).reshape(1, 1, -1)
    std = None if std_raw is None else np.asarray(std_raw, dtype=np.float32).reshape(1, 1, -1)
    if (mean is None) != (std is None):
        raise ArtifactRuntimeError("input_contract.normalize must provide both mean and std.")
    if mean is not None:
        if mean.shape[-1] not in {1, 3} or std is None or std.shape != mean.shape:
            raise ArtifactRuntimeError("input_contract normalize mean/std must have length 1 or 3.")
        if np.any(std == 0):
            raise ArtifactRuntimeError("input_contract normalize std values must be non-zero.")

    dtype_name = str(input_contract.get("dtype", "float32")).strip().lower()
    allowed_dtypes = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "uint8": np.uint8,
        "int8": np.int8,
        "int32": np.int32,
        "int64": np.int64,
    }
    if dtype_name not in allowed_dtypes:
        raise ArtifactRuntimeError(f"Unsupported input dtype: {dtype_name!r}")

    rows: list[np.ndarray] = []
    source_shapes: list[tuple[int, int]] = []
    for item in inputs:
        image = _decode_rgb_image(item)
        source_shapes.append((int(image.shape[0]), int(image.shape[1])))
        image = _resize_rgb(image, input_contract)
        if color == "BGR":
            image = image[..., ::-1]
        elif color == "GRAY":
            # The public input boundary remains canonical RGB. Conversion to a
            # one-channel tensor is an executable preprocessing-contract detail.
            image = np.asarray(
                np.round(
                    image[..., 0].astype(np.float32) * 0.299
                    + image[..., 1].astype(np.float32) * 0.587
                    + image[..., 2].astype(np.float32) * 0.114
                ),
                dtype=np.uint8,
            )[..., None]
        row = image.astype(np.float32)
        row = (row / divisor) * multiplier + offset
        if mean is not None and std is not None:
            row = (row - mean) / std
        if layout == "NCHW":
            row = np.transpose(row, (2, 0, 1))
        rows.append(np.asarray(row, dtype=allowed_dtypes[dtype_name]))

    try:
        batch = np.stack(rows, axis=0)
    except ValueError as exc:
        raise ArtifactRuntimeError(
            "Manifest preprocessing produced different tensor shapes in one batch; "
            "declare a fixed resize or infer with batch_size=1."
        ) from exc
    return np.ascontiguousarray(batch), source_shapes


def apply_output_transform(value: Any, contract: Mapping[str, Any]) -> np.ndarray:
    array = np.asarray(value)
    transform = str(contract.get("transform", "identity")).strip().lower()
    if transform in {"select_index", "softmax_select"}:
        axis = int(contract.get("axis", -1))
        index = int(contract.get("index", 0))
        if transform == "softmax_select":
            shifted = array - np.max(array, axis=axis, keepdims=True)
            exp = np.exp(shifted)
            array = exp / np.sum(exp, axis=axis, keepdims=True)
        array = np.take(array, indices=index, axis=axis)
    elif transform == "sigmoid":
        array = 1.0 / (1.0 + np.exp(-array))
    elif transform == "negate":
        array = -array
    elif transform != "identity":
        raise ArtifactRuntimeError(f"Unsupported output transform: {transform!r}")
    return np.asarray(array)


def normalize_score_output(
    value: Any, contract: Mapping[str, Any], *, batch_size: int
) -> np.ndarray:
    array = apply_output_transform(value, contract)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 0 and batch_size == 1:
        array = array.reshape(1)
    if array.ndim != 1 or int(array.shape[0]) != int(batch_size):
        raise ArtifactRuntimeError(
            "Score output contract must resolve to exactly one score per image; "
            f"got shape {tuple(array.shape)} for batch {batch_size}."
        )
    scores = np.asarray(array, dtype=np.float32)
    order = str(contract.get("score_order", "higher_is_more_anomalous")).strip().lower()
    if order == "lower_is_more_anomalous":
        scores = -scores
    elif order != "higher_is_more_anomalous":
        raise ArtifactRuntimeError(f"Unsupported score_order: {order!r}")
    return scores


def _resize_float_map(anomaly_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if anomaly_map.shape == (target_h, target_w):
        return np.asarray(anomaly_map, dtype=np.float32)
    from PIL import Image

    image = Image.fromarray(np.asarray(anomaly_map, dtype=np.float32), mode="F")
    resampling = getattr(Image, "Resampling", Image)
    return np.asarray(
        image.resize((target_w, target_h), resample=resampling.BILINEAR), dtype=np.float32
    )


def normalize_map_output(
    value: Any,
    contract: Mapping[str, Any],
    *,
    batch_size: int,
    source_shapes: Sequence[tuple[int, int]],
) -> np.ndarray | list[np.ndarray]:
    array = apply_output_transform(value, contract)
    layout = str(contract.get("layout", "NHW")).strip().upper()
    channel = contract.get("channel", contract.get("channel_index"))
    if layout == "NCHW":
        if array.ndim != 4:
            raise ArtifactRuntimeError(f"Expected NCHW anomaly map, got {tuple(array.shape)}")
        channel_index = int(channel) if channel is not None else 0
        if array.shape[1] != 1 and channel is None:
            raise ArtifactRuntimeError("Multi-channel anomaly map requires channel selection.")
        array = array[:, channel_index]
    elif layout == "NHWC":
        if array.ndim != 4:
            raise ArtifactRuntimeError(f"Expected NHWC anomaly map, got {tuple(array.shape)}")
        channel_index = int(channel) if channel is not None else 0
        if array.shape[-1] != 1 and channel is None:
            raise ArtifactRuntimeError("Multi-channel anomaly map requires channel selection.")
        array = array[..., channel_index]
    elif layout == "NHW":
        if array.ndim != 3:
            raise ArtifactRuntimeError(f"Expected NHW anomaly map, got {tuple(array.shape)}")
    elif layout == "HW" and batch_size == 1:
        if array.ndim != 2:
            raise ArtifactRuntimeError(f"Expected HW anomaly map, got {tuple(array.shape)}")
        array = array[None, ...]
    else:
        raise ArtifactRuntimeError(f"Unsupported anomaly-map layout: {layout!r}")

    maps = normalize_anomaly_maps(array, n_expected=batch_size)
    if not bool(contract.get("resize_to_source", False)):
        return maps
    resized = [_resize_float_map(maps[i], source_shapes[i]) for i in range(batch_size)]
    shapes = {tuple(item.shape) for item in resized}
    if len(shapes) == 1:
        return np.stack(resized, axis=0).astype(np.float32, copy=False)
    return resized


def extract_policy_threshold(policy: Mapping[str, Any]) -> float | None:
    postprocess = _mapping(policy.get("postprocess"))
    image_threshold = _mapping(postprocess.get("image_threshold"))
    value = image_threshold.get("threshold")
    if value is None:
        value = policy.get("threshold")
    return None if value is None else float(value)


def _build_map_postprocess(payload: Mapping[str, Any]) -> Any | None:
    if not payload:
        return None
    if str(payload.get("method", "")).strip().lower() in {"none", "identity"}:
        return None
    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

    percentile = payload.get("percentile_range", (1.0, 99.0))
    if not isinstance(percentile, (list, tuple)) or len(percentile) != 2:
        raise ArtifactRuntimeError("map_postprocess.percentile_range must contain two values.")
    return AnomalyMapPostprocess(
        normalize=bool(payload.get("normalize", True)),
        normalize_method=str(payload.get("normalize_method", "minmax")),
        percentile_range=(float(percentile[0]), float(percentile[1])),
        gaussian_sigma=float(payload.get("gaussian_sigma", 0.0)),
        morph_open_ksize=int(payload.get("morph_open_ksize", 0)),
        morph_close_ksize=int(payload.get("morph_close_ksize", 0)),
        component_threshold=(
            None
            if payload.get("component_threshold") is None
            else float(payload["component_threshold"])
        ),
        min_component_area=int(payload.get("min_component_area", 0)),
    )


def build_inference_defaults(policy: Mapping[str, Any]) -> dict[str, Any]:
    postprocess = _mapping(policy.get("postprocess"))
    adaptation = _mapping(policy.get("adaptation"))
    review = _mapping(postprocess.get("review_policy")) or _mapping(policy.get("prediction"))
    map_payload = _mapping(postprocess.get("map_postprocess")) or _mapping(
        adaptation.get("postprocess")
    )
    include_maps = bool(adaptation.get("save_maps", False))
    defaults: dict[str, Any] = {
        "include_maps": include_maps,
        "include_confidence": bool(review.get("reject_confidence_below") is not None),
        "reject_confidence_below": review.get("reject_confidence_below"),
        "reject_label": review.get("reject_label"),
        "postprocess": _build_map_postprocess(map_payload),
        "postprocess_summary": dict(map_payload) if map_payload else None,
    }
    return defaults


class ArtifactRuntime:
    """Detector-compatible facade over a verified executable artifact backend."""

    input_mode = "numpy"

    def __init__(
        self,
        backend_runtime: Any,
        *,
        manifest: Mapping[str, Any],
        infer_config: Mapping[str, Any],
        artifact_root: str | Path,
        model_name: str | None = None,
        runtime_info: Mapping[str, Any] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self.backend_runtime = backend_runtime
        self.manifest = dict(manifest)
        self.infer_config = dict(infer_config)
        self.artifact_root = Path(artifact_root)
        model = _mapping(self.manifest.get("model"))
        self.model_name = model_name or model.get("registry_name") or model.get("name")
        self.threshold_ = extract_policy_threshold(self.infer_config)
        self.inference_defaults = build_inference_defaults(self.infer_config)
        inherited_info = _mapping(getattr(backend_runtime, "runtime_info", None))
        inherited_info.update(dict(runtime_info or {}))
        inherited_info.setdefault("layout", self.manifest.get("layout"))
        inherited_info.setdefault("artifact_id", self.manifest.get("artifact_id"))
        self.runtime_info = inherited_info
        self._cleanup = cleanup
        self._finalizer = weakref.finalize(self, cleanup) if cleanup is not None else None

    @property
    def supports_maps(self) -> bool:
        output = _mapping(self.manifest.get("output_contract"))
        return isinstance(output.get("anomaly_map"), Mapping)

    def close(self) -> None:
        finalizer = self._finalizer
        if finalizer is not None and finalizer.alive:
            finalizer()
        self._cleanup = None

    def __enter__(self) -> "ArtifactRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def score_and_maps(
        self, inputs: Sequence[Any], *, include_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]:
        items = list(inputs)
        # Only use a hook implemented by the actual wrapper class. Thin policy
        # wrappers delegate unknown attributes via ``__getattr__``; treating a
        # delegated backend hook as the wrapper's own would bypass preprocessing.
        declared_hook = getattr(type(self.backend_runtime), "score_and_maps", None)
        hook = (
            getattr(self.backend_runtime, "score_and_maps", None)
            if callable(declared_hook)
            else None
        )
        if callable(hook):
            scores, maps = hook(items, include_maps=bool(include_maps))
        else:
            scores = self.backend_runtime.decision_function(items)
            maps = None
            if include_maps:
                map_fn = getattr(self.backend_runtime, "predict_anomaly_map", None)
                if callable(map_fn):
                    maps = map_fn(items)
                else:
                    single_map_fn = getattr(self.backend_runtime, "get_anomaly_map", None)
                    if callable(single_map_fn):
                        maps = [single_map_fn(item) for item in items]
        normalized_scores = normalize_scores(scores, n_expected=len(items))
        if maps is None or not include_maps:
            return normalized_scores, None
        if isinstance(maps, list) and len({tuple(np.asarray(m).shape) for m in maps}) > 1:
            return normalized_scores, [np.asarray(m, dtype=np.float32) for m in maps]
        return normalized_scores, normalize_anomaly_maps(maps, n_expected=len(items))

    def decision_function(self, inputs: Sequence[Any]) -> np.ndarray:
        scores, _maps = self.score_and_maps(inputs, include_maps=False)
        return scores

    def predict(self, inputs: Sequence[Any]) -> np.ndarray:
        if self.threshold_ is None:
            raise ArtifactRuntimeError(
                "This artifact is score-only and has no operating threshold. "
                "Bind a validated infer policy before calling predict(); "
                "decision_function() and infer() remain available."
            )
        scores = self.decision_function(inputs)
        return (scores > float(self.threshold_)).astype(np.int64)

    def predict_anomaly_map(self, inputs: Sequence[Any]) -> np.ndarray | list[np.ndarray]:
        if not self.supports_maps:
            raise ArtifactRuntimeError(
                "This artifact does not declare anomaly-map output capability."
            )
        _scores, maps = self.score_and_maps(inputs, include_maps=True)
        if maps is None:
            raise ArtifactRuntimeError(
                "Artifact backend did not produce its declared anomaly-map output."
            )
        return maps


def probe_artifact_policy(path: str | Path, *, trust_checkpoint: bool = False) -> dict[str, Any]:
    """Load and close a staged artifact to verify runtime/policy compatibility."""

    from pyimgano.services.artifact_load_service import load_artifact

    runtime = load_artifact(path, trust_checkpoint=bool(trust_checkpoint))
    try:
        return {
            "kind": "runtime_policy_probe",
            "status": "passed",
            "runtime_id": runtime.manifest.get("runtime_id"),
            "policy_id": runtime.manifest.get("policy_id"),
            "artifact_id": runtime.manifest.get("artifact_id"),
            "backend": runtime.runtime_info.get("backend"),
        }
    finally:
        runtime.close()


__all__ = [
    "ArtifactRuntime",
    "ArtifactRuntimeError",
    "apply_output_transform",
    "build_inference_defaults",
    "extract_policy_threshold",
    "normalize_map_output",
    "normalize_score_output",
    "prepare_image_batch",
    "probe_artifact_policy",
]
