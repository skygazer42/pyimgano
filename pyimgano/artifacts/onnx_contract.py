from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

_LAYOUTS = {"NCHW", "NHWC"}
_DTYPES = {"float16", "float32", "float64", "uint8"}
_TRANSFORMS = {"identity", "select_index", "sigmoid", "softmax_select", "negate"}
_MAP_LAYOUTS = {"NHW", "NCHW", "NHWC"}


class ONNXImportContractError(ValueError):
    pass


def _mapping(
    value: Any,
    *,
    name: str,
    allowed_keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ONNXImportContractError(f"{name} must be a JSON object/dict.")
    payload = dict(value)
    if allowed_keys is not None:
        unknown = set(payload) - allowed_keys
        if unknown:
            raise ONNXImportContractError(f"{name} contains unknown keys: {sorted(unknown)}")
    return payload


def _nonempty(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ONNXImportContractError(f"{name} must be a non-empty string.")
    return value.strip()


def _enum(value: Any, *, name: str, choices: set[str], upper: bool = False) -> str:
    text = _nonempty(value, name=name)
    normalized = text.upper() if upper else text.lower()
    if normalized not in choices:
        raise ONNXImportContractError(f"{name} must be one of: {sorted(choices)}")
    return normalized


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ONNXImportContractError(f"{name} must be a JSON boolean.")
    return value


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ONNXImportContractError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ONNXImportContractError(f"{name} must be >= {minimum}.")
    return int(value)


def _finite_float(value: Any, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ONNXImportContractError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise ONNXImportContractError(f"{name} must be a finite number.")
    return number


def _load_contract(value: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"ONNX import contract not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ONNXImportContractError(f"Invalid JSON in ONNX import contract: {path}") from exc
    return _mapping(payload, name="ONNX import contract")


def _normalize_input(value: Any) -> dict[str, Any]:
    payload = _mapping(
        value,
        name="input",
        allowed_keys={
            "name",
            "dtype",
            "layout",
            "color_space",
            "size",
            "dynamic_batch",
            "dynamic_spatial",
            "resize",
            "scale",
            "normalize",
        },
    )
    name = _nonempty(payload.get("name"), name="input.name")
    dtype = _enum(payload.get("dtype"), name="input.dtype", choices=_DTYPES)
    layout = _enum(payload.get("layout"), name="input.layout", choices=_LAYOUTS, upper=True)
    color = _enum(
        payload.get("color_space"),
        name="input.color_space",
        choices={"RGB", "BGR", "GRAY"},
        upper=True,
    )
    size = payload.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ONNXImportContractError("input.size must be [height, width].")
    height = _integer(size[0], name="input.size[0]", minimum=1)
    width = _integer(size[1], name="input.size[1]", minimum=1)

    resize = _mapping(
        payload.get("resize", {}),
        name="input.resize",
        allowed_keys={"mode", "interpolation"},
    )
    resize_mode = _enum(
        resize.get("mode", "stretch"),
        name="input.resize.mode",
        choices={"stretch", "letterbox", "center_crop"},
    )
    interpolation = _enum(
        resize.get("interpolation", "bilinear"),
        name="input.resize.interpolation",
        choices={"nearest", "bilinear", "bicubic", "area"},
    )

    scale = _mapping(
        payload.get("scale", {"divisor": 255.0}),
        name="input.scale",
        allowed_keys={"divisor"},
    )
    divisor = _finite_float(scale.get("divisor", 1.0), name="input.scale.divisor")
    if divisor == 0.0:
        raise ONNXImportContractError("input.scale.divisor must not be zero.")

    normalize = _mapping(
        payload.get("normalize", {}),
        name="input.normalize",
        allowed_keys={"mean", "std"},
    )
    channels = 1 if color == "GRAY" else 3
    mean = normalize.get("mean", [0.0] * channels)
    std = normalize.get("std", [1.0] * channels)
    if not isinstance(mean, (list, tuple)) or len(mean) != channels:
        raise ONNXImportContractError(f"input.normalize.mean must contain {channels} values.")
    if not isinstance(std, (list, tuple)) or len(std) != channels:
        raise ONNXImportContractError(f"input.normalize.std must contain {channels} values.")
    normalized_mean = [
        _finite_float(item, name=f"input.normalize.mean[{index}]")
        for index, item in enumerate(mean)
    ]
    normalized_std = [
        _finite_float(item, name=f"input.normalize.std[{index}]") for index, item in enumerate(std)
    ]
    if any(item == 0.0 for item in normalized_std):
        raise ONNXImportContractError("input.normalize.std values must not be zero.")

    dynamic_batch = _boolean(payload.get("dynamic_batch", False), name="input.dynamic_batch")
    dynamic_spatial = _boolean(payload.get("dynamic_spatial", False), name="input.dynamic_spatial")

    return {
        "kind": "image_batch",
        "name": name,
        "dtype": dtype,
        "layout": layout,
        "color_space": color,
        "size": [height, width],
        "dynamic_axes": {
            "batch": dynamic_batch,
            "spatial": dynamic_spatial,
        },
        "resize": {"mode": resize_mode, "interpolation": interpolation},
        "scale": {"divisor": divisor},
        "normalize": {
            "mean": normalized_mean,
            "std": normalized_std,
        },
    }


def _normalize_score(value: Any) -> dict[str, Any]:
    payload = _mapping(
        value,
        name="outputs.score",
        allowed_keys={"name", "transform", "score_order", "axis", "index"},
    )
    transform = _enum(
        payload.get("transform", "identity"),
        name="outputs.score.transform",
        choices=_TRANSFORMS,
    )
    score_order = _enum(
        payload.get("score_order"),
        name="outputs.score.score_order",
        choices={"higher_is_more_anomalous", "lower_is_more_anomalous"},
    )
    out: dict[str, Any] = {
        "name": _nonempty(payload.get("name"), name="outputs.score.name"),
        "transform": transform,
        "score_order": score_order,
    }
    if transform in {"select_index", "softmax_select"}:
        if payload.get("axis") is None or payload.get("index") is None:
            raise ONNXImportContractError(f"{transform} requires outputs.score.axis and index.")
        # Negative axes use NumPy/ONNX convention and are normalized against
        # the graph rank during import. Selection indices are never negative.
        out["axis"] = _integer(payload["axis"], name="outputs.score.axis")
        out["index"] = _integer(payload["index"], name="outputs.score.index", minimum=0)
    elif "axis" in payload or "index" in payload:
        raise ONNXImportContractError(
            "outputs.score.axis/index are valid only for select_index or softmax_select."
        )
    return out


def _normalize_map(value: Any) -> dict[str, Any]:
    payload = _mapping(
        value,
        name="outputs.anomaly_map",
        allowed_keys={"name", "layout", "resize_to_source", "channel"},
    )
    layout = _enum(
        payload.get("layout"),
        name="outputs.anomaly_map.layout",
        choices=_MAP_LAYOUTS,
        upper=True,
    )
    out: dict[str, Any] = {
        "name": _nonempty(payload.get("name"), name="outputs.anomaly_map.name"),
        "layout": layout,
        "resize_to_source": _boolean(
            payload.get("resize_to_source", True),
            name="outputs.anomaly_map.resize_to_source",
        ),
    }
    if payload.get("channel") is not None:
        out["channel"] = _integer(payload["channel"], name="outputs.anomaly_map.channel", minimum=0)
    return out


def normalize_onnx_import_contract(
    value: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(
        _load_contract(value),
        name="ONNX import contract",
        allowed_keys={"schema_family", "schema_version", "input", "outputs"},
    )
    if payload.get("schema_family") != "pyimgano-onnx-import":
        raise ONNXImportContractError(
            "ONNX import contract schema_family must be 'pyimgano-onnx-import'."
        )
    if (
        not isinstance(payload.get("schema_version"), int)
        or isinstance(payload.get("schema_version"), bool)
        or payload["schema_version"] != 1
    ):
        raise ONNXImportContractError("Only ONNX import contract schema_version=1 is supported.")
    outputs = _mapping(
        payload.get("outputs"),
        name="outputs",
        allowed_keys={"score", "anomaly_map"},
    )
    if "score" not in outputs:
        raise ONNXImportContractError("ONNX import contract requires outputs.score.")
    normalized_outputs: dict[str, Any] = {"score": _normalize_score(outputs["score"])}
    if outputs.get("anomaly_map") is not None:
        normalized_outputs["anomaly_map"] = _normalize_map(outputs["anomaly_map"])
        if normalized_outputs["score"]["name"] == normalized_outputs["anomaly_map"]["name"]:
            raise ONNXImportContractError(
                "outputs.score.name and outputs.anomaly_map.name must be distinct."
            )
    return {
        "schema_family": "pyimgano-onnx-import",
        "schema_version": 1,
        "input": _normalize_input(payload.get("input")),
        "outputs": normalized_outputs,
    }


__all__ = ["ONNXImportContractError", "normalize_onnx_import_contract"]
