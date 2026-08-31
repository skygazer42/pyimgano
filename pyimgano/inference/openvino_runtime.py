from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.inference.artifact_runtime import (
    ArtifactRuntimeError,
    normalize_map_output,
    normalize_score_output,
    prepare_image_batch,
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _port_names(port: Any) -> set[str]:
    get_names = getattr(port, "get_names", None)
    if callable(get_names):
        return {str(item) for item in get_names()}
    any_name = getattr(port, "any_name", None)
    return {str(any_name)} if any_name else set()


_OPENVINO_DEVICE_RE = re.compile(r"^[A-Z][A-Z0-9_.:-]*$")
_OPENVINO_DTYPE_ALIASES = {
    "boolean": "bool",
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
}


def _openvino_provider_key(spec: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": str(spec["name"]), "options": dict(spec.get("options", {}))},
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_openvino_provider_specs(value: Any, *, field: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, Mapping)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ArtifactRuntimeError(f"{field} must be a provider spec or list of provider specs.")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if isinstance(raw, str):
            raw_name, options = raw, {}
        elif isinstance(raw, Mapping):
            raw_name = str(raw.get("name", ""))
            raw_options = raw.get("options", {})
            if not isinstance(raw_options, Mapping):
                raise ArtifactRuntimeError(f"{field} provider options must be a mapping.")
            options = dict(raw_options)
        else:
            raise ArtifactRuntimeError(f"{field} entries must be strings or objects.")
        name = raw_name.strip().upper()
        if not name or _OPENVINO_DEVICE_RE.fullmatch(name) is None:
            raise ArtifactRuntimeError(f"{field} contains invalid OpenVINO device {raw_name!r}.")
        # Schema v1 has no OpenVINO compile-option allowlist. Reject rather than
        # silently ignoring a signed provider option during parity/runtime use.
        if options:
            raise ArtifactRuntimeError(
                f"{field} OpenVINO provider options are unsupported in schema v1."
            )
        spec = {"name": name, "options": {}}
        key = _openvino_provider_key(spec)
        if key in seen:
            raise ArtifactRuntimeError(f"{field} contains a duplicate provider spec: {spec!r}.")
        seen.add(key)
        specs.append(spec)
    return specs


def resolve_openvino_device(
    *,
    allowed: Any,
    verified: Any,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve one exact allowed-and-verified OpenVINO device without fallback."""

    allowed_specs = _normalize_openvino_provider_specs(
        allowed,
        field="runtime.allowed_providers",
    )
    verified_specs = _normalize_openvino_provider_specs(
        verified,
        field="runtime.verified_providers",
    )
    if not allowed_specs or not verified_specs:
        raise ArtifactRuntimeError(
            "OpenVINO artifacts require non-empty allowed_providers and verified_providers."
        )
    allowed_keys = {_openvino_provider_key(item) for item in allowed_specs}
    verified_keys = {_openvino_provider_key(item) for item in verified_specs}
    if not verified_keys.issubset(allowed_keys):
        raise ArtifactRuntimeError(
            "runtime.verified_providers must be an exact subset of allowed_providers."
        )
    if device is not None:
        requested = _normalize_openvino_provider_specs(device, field="device")[0]
        key = _openvino_provider_key(requested)
        if key not in allowed_keys:
            raise ArtifactRuntimeError(
                f"OpenVINO device {device!r} is not allowed by the artifact."
            )
        if key not in verified_keys:
            raise ArtifactRuntimeError(
                f"OpenVINO device {device!r} is not release-verified by the artifact."
            )
        selected = requested
    else:
        candidates = [
            item for item in allowed_specs if _openvino_provider_key(item) in verified_keys
        ]
        if not candidates:
            raise ArtifactRuntimeError(
                "OpenVINO artifact has no allowed-and-verified device provider."
            )
        selected = candidates[0]
    return str(selected["name"]), selected


def _dimension_value(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value) if value > 0 else None
    is_dynamic = getattr(value, "is_dynamic", None)
    if callable(is_dynamic):
        is_dynamic = is_dynamic()
    if is_dynamic is True:
        return None
    get_length = getattr(value, "get_length", None)
    if callable(get_length):
        try:
            length = int(get_length())
        except (TypeError, ValueError, RuntimeError):
            return None
        return length if length > 0 else None
    text = str(value).strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def _port_shape(port: Any, *, field: str) -> list[int | None]:
    shape = getattr(port, "partial_shape", None)
    if shape is None:
        get_shape = getattr(port, "get_partial_shape", None)
        if callable(get_shape):
            shape = get_shape()
    if shape is None:
        shape = getattr(port, "shape", None)
    if shape is None:
        get_shape = getattr(port, "get_shape", None)
        if callable(get_shape):
            shape = get_shape()
    if shape is None:
        raise ArtifactRuntimeError(f"{field} does not expose tensor shape metadata.")
    try:
        return [_dimension_value(item) for item in shape]
    except TypeError as exc:
        raise ArtifactRuntimeError(f"{field} shape metadata is not iterable.") from exc


def _normalize_openvino_dtype(value: Any) -> str:
    to_dtype = getattr(value, "to_dtype", None)
    if callable(to_dtype):
        try:
            return np.dtype(to_dtype()).name
        except (TypeError, ValueError):
            pass
    get_type_name = getattr(value, "get_type_name", None)
    if callable(get_type_name):
        value = get_type_name()
    text = str(value).strip().lower()
    match = re.fullmatch(r"<type:\s*['\"]([^'\"]+)['\"]>", text)
    if match is not None:
        text = match.group(1)
    text = _OPENVINO_DTYPE_ALIASES.get(text, text)
    try:
        return np.dtype(text).name
    except (TypeError, ValueError):
        return text


def _port_dtype(port: Any, *, field: str) -> str:
    element_type = getattr(port, "element_type", None)
    if element_type is None:
        get_element_type = getattr(port, "get_element_type", None)
        if callable(get_element_type):
            element_type = get_element_type()
    if element_type is None:
        raise ArtifactRuntimeError(f"{field} does not expose tensor dtype metadata.")
    return _normalize_openvino_dtype(element_type)


def _validate_batch_dimension(shape: Sequence[int | None], *, dynamic: bool, field: str) -> None:
    if not shape:
        raise ArtifactRuntimeError(f"{field} must include a batch dimension.")
    if dynamic and shape[0] is not None:
        raise ArtifactRuntimeError(
            f"{field} has a static batch dimension but the manifest declares dynamic batch."
        )
    if not dynamic and shape[0] is None:
        raise ArtifactRuntimeError(
            f"{field} has a dynamic batch dimension but the manifest declares static batch."
        )


def _validate_openvino_score_port(
    port: Any,
    contract: Mapping[str, Any],
    *,
    dynamic_batch: bool,
) -> None:
    dtype = _port_dtype(port, field="OpenVINO score output")
    if np.dtype(dtype).kind != "f":
        raise ArtifactRuntimeError(f"OpenVINO score output must be floating point; got {dtype!r}.")
    shape = _port_shape(port, field="OpenVINO score output")
    _validate_batch_dimension(
        shape,
        dynamic=dynamic_batch,
        field="OpenVINO score output",
    )
    transform = str(contract.get("transform", "identity")).strip().lower()
    if transform in {"select_index", "softmax_select"}:
        if len(shape) != 2:
            raise ArtifactRuntimeError(
                f"OpenVINO score transform {transform!r} requires rank-2 [batch, classes]."
            )
        axis = int(contract.get("axis", 0))
        axis = axis + len(shape) if axis < 0 else axis
        if axis != 1:
            raise ArtifactRuntimeError("OpenVINO score selection must use non-batch axis 1.")
        index = int(contract.get("index", -1))
        classes = shape[axis]
        if index < 0 or (classes is not None and index >= classes):
            raise ArtifactRuntimeError(
                f"OpenVINO score selection index {index} is invalid for shape {shape!r}."
            )
        return
    if len(shape) == 1 or (len(shape) == 2 and shape[1] == 1):
        return
    raise ArtifactRuntimeError(
        "OpenVINO score output must be [batch] or [batch, 1] unless a class-selection "
        f"transform is declared; got {shape!r}."
    )


def _validate_openvino_map_port(
    port: Any,
    contract: Mapping[str, Any],
    *,
    dynamic_batch: bool,
) -> None:
    dtype = _port_dtype(port, field="OpenVINO anomaly-map output")
    if np.dtype(dtype).kind != "f":
        raise ArtifactRuntimeError(
            f"OpenVINO anomaly-map output must be floating point; got {dtype!r}."
        )
    shape = _port_shape(port, field="OpenVINO anomaly-map output")
    _validate_batch_dimension(
        shape,
        dynamic=dynamic_batch,
        field="OpenVINO anomaly-map output",
    )
    layout = str(contract.get("layout", "")).strip().upper()
    expected_rank = {"NHW": 3, "NCHW": 4, "NHWC": 4}.get(layout)
    if expected_rank is None or len(shape) != expected_rank:
        raise ArtifactRuntimeError(
            f"OpenVINO anomaly-map layout {layout!r} does not match shape {shape!r}."
        )
    if layout == "NHW":
        if contract.get("channel", contract.get("channel_index")) is not None:
            raise ArtifactRuntimeError("NHW anomaly maps must not declare a channel index.")
        return
    channel_axis = 1 if layout == "NCHW" else 3
    channels = shape[channel_axis]
    channel = contract.get("channel", contract.get("channel_index"))
    if channel is None:
        if channels != 1:
            raise ArtifactRuntimeError(
                "Multi-channel or dynamic-channel OpenVINO anomaly maps require channel selection."
            )
        return
    selected = int(channel)
    if selected < 0 or (channels is not None and selected >= channels):
        raise ArtifactRuntimeError(
            f"OpenVINO anomaly-map channel {selected} is invalid for shape {shape!r}."
        )


class OpenVINOArtifactRuntime:
    """Manifest-driven OpenVINO IR full-detector runtime."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        allowed_providers: Any = None,
        verified_providers: Any = None,
        device: str | None = None,
        openvino_module: Any | None = None,
        core: Any | None = None,
        compiled_model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        device_name, selected_provider = resolve_openvino_device(
            allowed=allowed_providers,
            verified=verified_providers,
            device=device,
        )
        if compiled_model is None:
            if openvino_module is None:
                from pyimgano.utils.optional_deps import require

                openvino_module = require(
                    "openvino",
                    extra="openvino-runtime",
                    purpose="loading an OpenVINO artifact",
                )
            core = core or openvino_module.Core()
            available_devices = getattr(core, "available_devices", None)
            if available_devices is not None:
                available = {str(item).strip().upper() for item in available_devices}
                if device_name not in available:
                    raise ArtifactRuntimeError(
                        f"Artifact-selected OpenVINO device {device_name!r} is unavailable; "
                        f"available={sorted(available)}."
                    )
            model = core.read_model(str(self.model_path))
            compiled_model = core.compile_model(model, device_name)
        self.compiled_model = compiled_model
        self.device = device_name
        self._validate_io_contract()
        self.runtime_info = {
            "backend": "openvino",
            "device": device_name,
            "providers": [str(selected_provider["name"])],
            "selected_provider": str(selected_provider["name"]),
        }

    def _inputs(self) -> list[Any]:
        value = getattr(self.compiled_model, "inputs", None)
        return list(value() if callable(value) else value or [])

    def _outputs(self) -> list[Any]:
        value = getattr(self.compiled_model, "outputs", None)
        return list(value() if callable(value) else value or [])

    def _validate_io_contract(self) -> None:
        inputs = self._inputs()
        if len(inputs) != 1:
            raise ArtifactRuntimeError(
                f"OpenVINO artifact must expose exactly one runtime input; got {len(inputs)}."
            )
        expected = str(self.input_contract.get("name", "")).strip()
        if not expected:
            raise ArtifactRuntimeError("input_contract.name is required for OpenVINO artifacts.")
        names = _port_names(inputs[0])
        if expected not in names:
            raise ArtifactRuntimeError(
                f"OpenVINO input {expected!r} does not match model names {sorted(names)}."
            )
        expected_dtype = _normalize_openvino_dtype(
            str(self.input_contract.get("dtype", "")).strip().lower()
        )
        actual_dtype = _port_dtype(inputs[0], field="OpenVINO input")
        if not expected_dtype or actual_dtype != expected_dtype:
            raise ArtifactRuntimeError(
                f"OpenVINO input dtype mismatch: manifest={expected_dtype!r}, "
                f"graph={actual_dtype!r}."
            )
        shape = _port_shape(inputs[0], field="OpenVINO input")
        if len(shape) != 4:
            raise ArtifactRuntimeError(f"OpenVINO image input must have rank 4; got {shape!r}.")
        layout = str(self.input_contract.get("layout", "")).strip().upper()
        if layout not in {"NCHW", "NHWC"}:
            raise ArtifactRuntimeError(f"Unsupported OpenVINO input layout: {layout!r}.")
        axes = _mapping(self.input_contract.get("dynamic_axes"))
        dynamic_batch = bool(axes.get("batch", False))
        dynamic_spatial = bool(axes.get("spatial", False))
        _validate_batch_dimension(shape, dynamic=dynamic_batch, field="OpenVINO input")
        channel_axis = 1 if layout == "NCHW" else 3
        height_axis, width_axis = (2, 3) if layout == "NCHW" else (1, 2)
        color = str(self.input_contract.get("color_space", "")).strip().upper()
        if color not in {"RGB", "BGR", "GRAY"}:
            raise ArtifactRuntimeError(f"Unsupported OpenVINO color_space: {color!r}.")
        expected_channels = 1 if color == "GRAY" else 3
        if shape[channel_axis] != expected_channels:
            raise ArtifactRuntimeError(
                "OpenVINO input channel dimension does not match manifest color_space."
            )
        size = self.input_contract.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ArtifactRuntimeError("OpenVINO input_contract.size must be [height, width].")
        for axis, expected_size, name in (
            (height_axis, int(size[0]), "height"),
            (width_axis, int(size[1]), "width"),
        ):
            actual = shape[axis]
            if dynamic_spatial and actual is not None:
                raise ArtifactRuntimeError(
                    f"OpenVINO input {name} is static but manifest declares dynamic spatial axes."
                )
            if not dynamic_spatial and actual != expected_size:
                raise ArtifactRuntimeError(
                    f"OpenVINO input {name} mismatch: "
                    f"manifest={expected_size}, graph={actual!r}."
                )

        outputs = self._outputs()
        output_names = sorted({name for output in outputs for name in _port_names(output)})

        def declared_port(
            contract: Mapping[str, Any],
            *,
            field: str,
            default_index: int,
        ) -> Any:
            name = str(contract.get("name", "")).strip()
            matches = [
                (index, output)
                for index, output in enumerate(outputs)
                if name and name in _port_names(output)
            ]
            if len(matches) != 1:
                raise ArtifactRuntimeError(
                    f"Declared OpenVINO {field} output {name!r} is not uniquely present; "
                    f"graph outputs={output_names}."
                )
            index, output = matches[0]
            declared_index = int(contract.get("output_index", default_index))
            if declared_index != index:
                raise ArtifactRuntimeError(
                    f"OpenVINO {field} output_index mismatch: "
                    f"manifest={declared_index}, graph={index}."
                )
            return output

        score = _mapping(self.output_contract.get("score"))
        score_port = declared_port(score, field="score", default_index=0)
        _validate_openvino_score_port(
            score_port,
            score,
            dynamic_batch=dynamic_batch,
        )
        anomaly_map = self.output_contract.get("anomaly_map")
        if isinstance(anomaly_map, Mapping):
            map_port = declared_port(anomaly_map, field="anomaly_map", default_index=1)
            if map_port is score_port:
                raise ArtifactRuntimeError(
                    "OpenVINO score and anomaly-map contracts must select distinct outputs."
                )
            _validate_openvino_map_port(
                map_port,
                anomaly_map,
                dynamic_batch=dynamic_batch,
            )

    def _infer(self, batch: np.ndarray) -> Any:
        input_name = str(self.input_contract.get("name", "")).strip()
        try:
            return self.compiled_model({input_name: batch})
        except Exception:
            # Some OpenVINO versions accept a positional list but not tensor names.
            return self.compiled_model([batch])

    def _select_result(
        self, result: Any, contract: Mapping[str, Any], *, default_index: int
    ) -> Any:
        name = str(contract.get("name", "")).strip()
        if isinstance(result, Mapping):
            if name in result:
                return result[name]
            for key, value in result.items():
                if name in _port_names(key):
                    return value
            values = list(result.values())
        elif isinstance(result, (list, tuple)):
            values = list(result)
        else:
            values = [result]
        index = int(contract.get("output_index", default_index))
        if index < 0 or index >= len(values):
            raise ArtifactRuntimeError(
                f"OpenVINO output {name!r} could not be selected from {len(values)} result(s)."
            )
        return values[index]

    def score_and_maps(
        self, inputs: Sequence[Any], *, include_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]:
        items = list(inputs)
        if not items:
            return np.zeros((0,), dtype=np.float32), None
        batch, source_shapes = prepare_image_batch(items, self.input_contract)
        result = self._infer(batch)
        score_contract = _mapping(self.output_contract.get("score"))
        score_value = self._select_result(result, score_contract, default_index=0)
        scores = normalize_score_output(score_value, score_contract, batch_size=len(items))
        maps = None
        map_contract = _mapping(self.output_contract.get("anomaly_map"))
        if include_maps and map_contract:
            map_value = self._select_result(result, map_contract, default_index=1)
            maps = normalize_map_output(
                map_value,
                map_contract,
                batch_size=len(items),
                source_shapes=source_shapes,
            )
        return scores, maps

    def decision_function(self, inputs: Sequence[Any]) -> np.ndarray:
        return self.score_and_maps(inputs, include_maps=False)[0]

    def predict_anomaly_map(self, inputs: Sequence[Any]) -> np.ndarray | list[np.ndarray]:
        _scores, maps = self.score_and_maps(inputs, include_maps=True)
        if maps is None:
            raise ArtifactRuntimeError("OpenVINO artifact has no declared anomaly-map output.")
        return maps


__all__ = ["OpenVINOArtifactRuntime", "resolve_openvino_device"]
