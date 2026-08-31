from __future__ import annotations

"""Strict schema-v1 validation for artifact tensor I/O contracts.

The manifest loader owns error translation and identity validation.  This module
only validates and normalizes the JSON-shaped input/output contract fragments so
the same rules can also be reused by importers and runtime construction code.
"""

import math
from typing import Any, Mapping

MAX_IMAGE_DIMENSION = 65_535

_ARTIFACT_LAYOUTS = {"native_detector", "single_graph", "composite"}
_BACKENDS = {"pyimgano", "onnxruntime", "torchscript", "openvino"}
_GRAPH_BACKENDS = {"onnxruntime", "torchscript", "openvino"}
_INPUT_DTYPES = {"float16", "float32", "float64", "uint8"}
_INPUT_LAYOUTS = {"NCHW", "NHWC"}
_COLOR_SPACES = {"RGB", "BGR", "GRAY"}
_RESIZE_MODES = {"stretch", "letterbox", "center_crop"}
_INTERPOLATIONS = {"nearest", "bilinear", "bicubic", "area", "lanczos"}
_OUTPUT_TRANSFORMS = {
    "identity",
    "select_index",
    "sigmoid",
    "softmax_select",
    "negate",
}
_SELECTION_TRANSFORMS = {"select_index", "softmax_select"}
_SCORE_ORDERS = {"higher_is_more_anomalous", "lower_is_more_anomalous"}
_MAP_LAYOUTS = {"NHW", "NCHW", "NHWC"}


class ArtifactIOContractError(ValueError):
    """Raised when an artifact schema-v1 input or output contract is invalid."""


def _field_error(path: str, message: str) -> ArtifactIOContractError:
    return ArtifactIOContractError(f"{path}: {message}")


def _mapping(
    value: Any,
    *,
    path: str,
    allowed_keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _field_error(path, "expected a JSON object")
    non_string = [key for key in value if not isinstance(key, str)]
    if non_string:
        raise _field_error(path, "JSON object keys must be strings")
    payload = dict(value)
    unknown = sorted(set(payload) - allowed_keys)
    if unknown:
        raise _field_error(path, f"unknown keys: {unknown}")
    return payload


def _nonempty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _field_error(path, "expected a non-empty string")
    return value.strip()


def _enum(value: Any, *, path: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise _field_error(path, f"must be one of {sorted(choices)}")
    return value


def _boolean(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise _field_error(path, "expected a JSON boolean")
    return value


def _integer(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _field_error(path, "expected an integer")
    if minimum is not None and value < minimum:
        raise _field_error(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise _field_error(path, f"must be <= {maximum}")
    return value


def _finite_number(value: Any, *, path: str) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise _field_error(path, "expected a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _field_error(path, "expected a finite number")
    return value


def _validate_layout_backend(*, layout: Any, backend: Any) -> tuple[str, str]:
    normalized_layout = _enum(layout, path="layout", choices=_ARTIFACT_LAYOUTS)
    normalized_backend = _enum(backend, path="runtime.backend", choices=_BACKENDS)
    if normalized_layout == "native_detector" and normalized_backend != "pyimgano":
        raise _field_error("runtime.backend", "native_detector input contracts require 'pyimgano'")
    if normalized_layout == "single_graph" and normalized_backend not in _GRAPH_BACKENDS:
        raise _field_error(
            "runtime.backend",
            "single_graph input contracts require onnxruntime, torchscript, or openvino",
        )
    if normalized_layout == "composite" and normalized_backend != "pyimgano":
        raise _field_error("runtime.backend", "composite input contracts require 'pyimgano'")
    return normalized_layout, normalized_backend


def _validate_native_input_contract(value: Any) -> dict[str, Any]:
    path = "input_contract"
    payload = _mapping(
        value,
        path=path,
        allowed_keys={"kind", "dtype", "layout", "color_space"},
    )
    payload["kind"] = _enum(payload.get("kind"), path=f"{path}.kind", choices={"image_batch"})
    payload["dtype"] = _enum(payload.get("dtype"), path=f"{path}.dtype", choices={"uint8"})
    payload["layout"] = _enum(payload.get("layout"), path=f"{path}.layout", choices={"HWC"})
    if "color_space" in payload:
        payload["color_space"] = _enum(
            payload["color_space"], path=f"{path}.color_space", choices={"RGB"}
        )
    return payload


def _validate_size(value: Any, *, path: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 2:
        raise _field_error(path, "expected [height, width]")
    return [
        _integer(
            item,
            path=f"{path}[{index}]",
            minimum=1,
            maximum=MAX_IMAGE_DIMENSION,
        )
        for index, item in enumerate(value)
    ]


def _validate_dynamic_axes(value: Any, *, path: str) -> dict[str, bool]:
    payload = _mapping(value, path=path, allowed_keys={"batch", "spatial"})
    return {key: _boolean(item, path=f"{path}.{key}") for key, item in payload.items()}


def _validate_fill(value: Any, *, path: str) -> int | list[int]:
    if isinstance(value, list):
        if len(value) != 3:
            raise _field_error(path, "expected an integer or an RGB triplet")
        return [
            _integer(item, path=f"{path}[{index}]", minimum=0, maximum=255)
            for index, item in enumerate(value)
        ]
    return _integer(value, path=path, minimum=0, maximum=255)


def _validate_resize(value: Any, *, path: str) -> dict[str, Any]:
    payload = _mapping(
        value,
        path=path,
        allowed_keys={"mode", "interpolation", "fill"},
    )
    mode = _enum(payload.get("mode"), path=f"{path}.mode", choices=_RESIZE_MODES)
    payload["mode"] = mode
    payload["interpolation"] = _enum(
        payload.get("interpolation"),
        path=f"{path}.interpolation",
        choices=_INTERPOLATIONS,
    )
    if "fill" in payload:
        if mode != "letterbox":
            raise _field_error(f"{path}.fill", "is valid only when resize.mode is 'letterbox'")
        payload["fill"] = _validate_fill(payload["fill"], path=f"{path}.fill")
    return payload


def _validate_scale(value: Any, *, path: str) -> dict[str, int | float]:
    payload = _mapping(
        value,
        path=path,
        allowed_keys={"divisor", "multiplier", "offset"},
    )
    normalized = {key: _finite_number(item, path=f"{path}.{key}") for key, item in payload.items()}
    if normalized.get("divisor") == 0:
        raise _field_error(f"{path}.divisor", "must not be zero")
    return normalized


def _validate_numeric_vector(
    value: Any,
    *,
    path: str,
    length: int,
) -> list[int | float]:
    if not isinstance(value, list) or len(value) != length:
        raise _field_error(path, f"expected exactly {length} values")
    return [_finite_number(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def _validate_normalize(value: Any, *, path: str, channels: int) -> dict[str, Any]:
    payload = _mapping(value, path=path, allowed_keys={"mean", "std"})
    if "mean" not in payload:
        raise _field_error(f"{path}.mean", "is required when normalize is declared")
    if "std" not in payload:
        raise _field_error(f"{path}.std", "is required when normalize is declared")
    mean = _validate_numeric_vector(payload["mean"], path=f"{path}.mean", length=channels)
    std = _validate_numeric_vector(payload["std"], path=f"{path}.std", length=channels)
    for index, item in enumerate(std):
        if item == 0:
            raise _field_error(f"{path}.std[{index}]", "must not be zero")
    return {"mean": mean, "std": std}


def _validate_graph_input_contract(value: Any) -> dict[str, Any]:
    path = "input_contract"
    payload = _mapping(
        value,
        path=path,
        allowed_keys={
            "kind",
            "name",
            "dtype",
            "layout",
            "color_space",
            "size",
            "dynamic_axes",
            "resize",
            "scale",
            "normalize",
        },
    )
    payload["kind"] = _enum(payload.get("kind"), path=f"{path}.kind", choices={"image_batch"})
    payload["name"] = _nonempty_string(payload.get("name"), path=f"{path}.name")
    payload["dtype"] = _enum(payload.get("dtype"), path=f"{path}.dtype", choices=_INPUT_DTYPES)
    payload["layout"] = _enum(payload.get("layout"), path=f"{path}.layout", choices=_INPUT_LAYOUTS)
    payload["color_space"] = _enum(
        payload.get("color_space"),
        path=f"{path}.color_space",
        choices=_COLOR_SPACES,
    )
    payload["size"] = _validate_size(payload.get("size"), path=f"{path}.size")
    if "dynamic_axes" in payload:
        payload["dynamic_axes"] = _validate_dynamic_axes(
            payload["dynamic_axes"], path=f"{path}.dynamic_axes"
        )
    if "resize" in payload:
        payload["resize"] = _validate_resize(payload["resize"], path=f"{path}.resize")
    if "scale" in payload:
        payload["scale"] = _validate_scale(payload["scale"], path=f"{path}.scale")
    if "normalize" in payload:
        channels = 1 if payload["color_space"] == "GRAY" else 3
        payload["normalize"] = _validate_normalize(
            payload["normalize"],
            path=f"{path}.normalize",
            channels=channels,
        )
    return payload


def validate_artifact_input_contract(
    value: Any,
    *,
    layout: str,
    backend: str,
) -> dict[str, Any]:
    """Validate and return a normalized artifact schema-v1 input contract."""

    normalized_layout, _normalized_backend = _validate_layout_backend(
        layout=layout,
        backend=backend,
    )
    if normalized_layout == "native_detector":
        return _validate_native_input_contract(value)
    return _validate_graph_input_contract(value)


def _validate_output_transform_fields(
    payload: dict[str, Any],
    *,
    path: str,
    transform_required: bool,
) -> None:
    if transform_required or "transform" in payload:
        transform = _enum(
            payload.get("transform"),
            path=f"{path}.transform",
            choices=_OUTPUT_TRANSFORMS,
        )
        payload["transform"] = transform
    else:
        transform = "identity"

    if transform in _SELECTION_TRANSFORMS:
        if "axis" not in payload:
            raise _field_error(f"{path}.axis", f"is required for transform {transform!r}")
        if "index" not in payload:
            raise _field_error(f"{path}.index", f"is required for transform {transform!r}")
        payload["axis"] = _integer(payload["axis"], path=f"{path}.axis")
        payload["index"] = _integer(payload["index"], path=f"{path}.index", minimum=0)
    else:
        for field in ("axis", "index"):
            if field in payload:
                raise _field_error(
                    f"{path}.{field}",
                    "is valid only for select_index or softmax_select",
                )


def _validate_output_index(payload: dict[str, Any], *, path: str) -> None:
    if "output_index" in payload:
        payload["output_index"] = _integer(
            payload["output_index"], path=f"{path}.output_index", minimum=0
        )


def _validate_score(value: Any) -> dict[str, Any]:
    path = "output_contract.score"
    payload = _mapping(
        value,
        path=path,
        allowed_keys={
            "name",
            "transform",
            "score_order",
            "axis",
            "index",
            "output_index",
        },
    )
    payload["name"] = _nonempty_string(payload.get("name"), path=f"{path}.name")
    _validate_output_transform_fields(payload, path=path, transform_required=True)
    payload["score_order"] = _enum(
        payload.get("score_order"),
        path=f"{path}.score_order",
        choices=_SCORE_ORDERS,
    )
    _validate_output_index(payload, path=path)
    return payload


def _validate_map(value: Any) -> dict[str, Any]:
    path = "output_contract.anomaly_map"
    payload = _mapping(
        value,
        path=path,
        allowed_keys={
            "name",
            "layout",
            "channel",
            "resize_to_source",
            "transform",
            "axis",
            "index",
            "output_index",
        },
    )
    payload["name"] = _nonempty_string(payload.get("name"), path=f"{path}.name")
    payload["layout"] = _enum(payload.get("layout"), path=f"{path}.layout", choices=_MAP_LAYOUTS)
    if "channel" in payload:
        if payload["layout"] == "NHW":
            raise _field_error(f"{path}.channel", "must not be declared for NHW maps")
        payload["channel"] = _integer(payload["channel"], path=f"{path}.channel", minimum=0)
    if "resize_to_source" not in payload:
        raise _field_error(f"{path}.resize_to_source", "is required")
    payload["resize_to_source"] = _boolean(
        payload["resize_to_source"], path=f"{path}.resize_to_source"
    )
    _validate_output_transform_fields(payload, path=path, transform_required=False)
    _validate_output_index(payload, path=path)
    return payload


def validate_artifact_output_contract(value: Any) -> dict[str, Any]:
    """Validate and return a normalized artifact schema-v1 output contract."""

    path = "output_contract"
    payload = _mapping(
        value,
        path=path,
        allowed_keys={"score", "anomaly_map"},
    )
    if "score" not in payload:
        raise _field_error(f"{path}.score", "is required")
    payload["score"] = _validate_score(payload["score"])
    if "anomaly_map" in payload:
        payload["anomaly_map"] = _validate_map(payload["anomaly_map"])
        if payload["score"]["name"] == payload["anomaly_map"]["name"]:
            raise _field_error(
                f"{path}.anomaly_map.name",
                "must differ from output_contract.score.name",
            )
    return payload


__all__ = [
    "ArtifactIOContractError",
    "MAX_IMAGE_DIMENSION",
    "validate_artifact_input_contract",
    "validate_artifact_output_contract",
]
