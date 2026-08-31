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

_SESSION_OPTION_KEYS = {
    "intra_op_num_threads",
    "inter_op_num_threads",
    "execution_mode",
    "graph_optimization_level",
    "enable_mem_pattern",
    "enable_cpu_mem_arena",
    "log_severity_level",
    "log_verbosity_level",
    "session_config_entries",
}
_SESSION_CONFIG_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_CUDA_PROVIDER_OPTION_KEYS = {
    "device_id",
    "arena_extend_strategy",
    "gpu_mem_limit",
    "cudnn_conv_algo_search",
    "do_copy_in_default_stream",
    "cudnn_conv_use_max_workspace",
    "cudnn_conv1d_pad_to_nc1d",
    "enable_cuda_graph",
    "prefer_nhwc",
}
_TENSORRT_PROVIDER_OPTION_KEYS = {
    "device_id",
    "trt_max_workspace_size",
    "trt_fp16_enable",
    "trt_int8_enable",
    "trt_engine_cache_enable",
    "trt_engine_cache_path",
}
_ORT_TENSOR_TYPES = {
    "float16": "tensor(float16)",
    "float32": "tensor(float)",
    "float64": "tensor(double)",
    "uint8": "tensor(uint8)",
}
_ORT_FLOAT_TENSOR_TYPES = frozenset({"tensor(float16)", "tensor(float)", "tensor(double)"})


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_provider_specs(value: Any, *, field: str) -> list[dict[str, Any]]:
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
            name, options = raw, {}
        elif isinstance(raw, Mapping):
            name = str(raw.get("name", "")).strip()
            options = _mapping(raw.get("options"))
        else:
            raise ArtifactRuntimeError(f"{field} entries must be strings or objects.")
        if not name or name in seen:
            raise ArtifactRuntimeError(f"{field} contains an empty or duplicate provider: {name!r}")
        seen.add(name)
        allowed_keys: set[str]
        if name == "CPUExecutionProvider":
            allowed_keys = set()
        elif name == "CUDAExecutionProvider":
            allowed_keys = _CUDA_PROVIDER_OPTION_KEYS
        elif name == "TensorrtExecutionProvider":
            allowed_keys = _TENSORRT_PROVIDER_OPTION_KEYS
        else:
            # Unknown providers are permitted only without options. This avoids
            # passing backend-specific path/library controls through a manifest.
            allowed_keys = set()
        unknown = sorted(set(options) - allowed_keys)
        if unknown:
            raise ArtifactRuntimeError(
                f"{field} provider {name!r} contains unsupported option(s): {unknown}"
            )
        for key, option in options.items():
            if not isinstance(option, (str, int, float, bool)):
                raise ArtifactRuntimeError(
                    f"{field} provider option {name}.{key} must be a scalar value."
                )
        specs.append({"name": name, "options": dict(options)})
    return specs


def _provider_names(specs: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(item["name"]) for item in specs]


def _provider_spec_key(spec: Mapping[str, Any]) -> str:
    """Return a type-preserving identity for an already-normalized spec."""

    return json.dumps(
        {"name": str(spec["name"]), "options": dict(spec.get("options", {}))},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _static_dimension(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def _node_shape(node: Any, *, field: str) -> list[Any]:
    shape = getattr(node, "shape", None)
    if not isinstance(shape, (list, tuple)):
        raise ArtifactRuntimeError(f"{field} does not expose tensor shape metadata.")
    return list(shape)


def _validate_batch_dimension(shape: Sequence[Any], *, dynamic: bool, field: str) -> None:
    if not shape:
        raise ArtifactRuntimeError(f"{field} must include a batch dimension.")
    static = _static_dimension(shape[0])
    if dynamic and static is not None:
        raise ArtifactRuntimeError(
            f"{field} has a static batch dimension but the manifest declares dynamic batch."
        )
    if not dynamic and static is None:
        raise ArtifactRuntimeError(
            f"{field} has a dynamic batch dimension but the manifest declares static batch."
        )


def _validate_score_metadata(node: Any, contract: Mapping[str, Any], *, dynamic: bool) -> None:
    output_type = str(getattr(node, "type", "")).strip().lower()
    if output_type not in _ORT_FLOAT_TENSOR_TYPES:
        raise ArtifactRuntimeError(
            f"ONNX score output must be floating point; got {output_type!r}."
        )
    shape = _node_shape(node, field="ONNX score output")
    _validate_batch_dimension(shape, dynamic=dynamic, field="ONNX score output")
    transform = str(contract.get("transform", "identity")).strip().lower()
    if transform in {"select_index", "softmax_select"}:
        if len(shape) != 2:
            raise ArtifactRuntimeError(
                f"ONNX score transform {transform!r} requires a rank-2 [batch, classes] output."
            )
        axis = int(contract.get("axis", 0))
        axis = axis + len(shape) if axis < 0 else axis
        if axis != 1:
            raise ArtifactRuntimeError("ONNX score selection must use the non-batch axis 1.")
        index = int(contract.get("index", -1))
        classes = _static_dimension(shape[axis])
        if index < 0 or (classes is not None and index >= classes):
            raise ArtifactRuntimeError(
                f"ONNX score selection index {index} is invalid for shape {shape!r}."
            )
        return
    if len(shape) == 1:
        return
    if len(shape) == 2 and _static_dimension(shape[1]) == 1:
        return
    raise ArtifactRuntimeError(
        "ONNX score output must be [batch] or [batch, 1] unless a class-selection "
        f"transform is declared; got {shape!r}."
    )


def _validate_map_metadata(node: Any, contract: Mapping[str, Any], *, dynamic: bool) -> None:
    output_type = str(getattr(node, "type", "")).strip().lower()
    if output_type not in _ORT_FLOAT_TENSOR_TYPES:
        raise ArtifactRuntimeError(
            f"ONNX anomaly-map output must be floating point; got {output_type!r}."
        )
    shape = _node_shape(node, field="ONNX anomaly-map output")
    _validate_batch_dimension(shape, dynamic=dynamic, field="ONNX anomaly-map output")
    layout = str(contract.get("layout", "")).strip().upper()
    expected_rank = {"NHW": 3, "NCHW": 4, "NHWC": 4}.get(layout)
    if expected_rank is None or len(shape) != expected_rank:
        raise ArtifactRuntimeError(
            f"ONNX anomaly-map layout {layout!r} does not match shape {shape!r}."
        )
    if layout == "NHW":
        if contract.get("channel") is not None:
            raise ArtifactRuntimeError("NHW anomaly maps must not declare a channel index.")
        return
    channel_axis = 1 if layout == "NCHW" else 3
    channels = _static_dimension(shape[channel_axis])
    channel = contract.get("channel", contract.get("channel_index"))
    if channel is None:
        if channels != 1:
            raise ArtifactRuntimeError(
                "Multi-channel or dynamic-channel ONNX anomaly maps require channel selection."
            )
        return
    selected = int(channel)
    if selected < 0 or (channels is not None and selected >= channels):
        raise ArtifactRuntimeError(
            f"ONNX anomaly-map channel {selected} is invalid for shape {shape!r}."
        )


def _device_provider(device: str) -> str:
    value = str(device).strip().lower()
    if value in {"cpu"}:
        return "CPUExecutionProvider"
    if value in {"cuda", "gpu"} or value.startswith("cuda:"):
        return "CUDAExecutionProvider"
    raise ArtifactRuntimeError(f"Unsupported ONNX Runtime device override: {device!r}")


def resolve_onnx_providers(
    ort: Any,
    *,
    allowed: Any = None,
    verified: Any = None,
    providers: Any = None,
    device: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve strict provider precedence and return specs plus warnings."""

    if providers is not None and device is not None:
        raise ArtifactRuntimeError("Pass either providers or device, not both.")
    allowed_specs = _normalize_provider_specs(allowed, field="runtime.allowed_providers")
    verified_specs = _normalize_provider_specs(verified, field="runtime.verified_providers")
    allowed_names = set(_provider_names(allowed_specs))
    allowed_keys = {_provider_spec_key(item) for item in allowed_specs}
    verified_keys = {_provider_spec_key(item) for item in verified_specs}
    if not verified_keys.issubset(allowed_keys):
        raise ArtifactRuntimeError(
            "runtime.verified_providers must be a subset of allowed_providers."
        )

    available = {str(item) for item in ort.get_available_providers()}
    warnings: list[str] = []
    if providers is not None:
        chosen = _normalize_provider_specs(providers, field="providers")
        names = _provider_names(chosen)
        disallowed = [name for name in names if allowed_names and name not in allowed_names]
        if disallowed:
            raise ArtifactRuntimeError(
                f"Explicit ONNX provider(s) are not allowed by the artifact: {disallowed}"
            )
        unverified = [
            item
            for item in chosen
            if verified_specs and _provider_spec_key(item) not in verified_keys
        ]
        if unverified or (allowed_specs and not verified_specs):
            raise ArtifactRuntimeError(
                "Explicit ONNX provider/options are not release-verified by the artifact: "
                f"{unverified or chosen}"
            )
        unavailable = [name for name in names if name not in available]
        if unavailable:
            raise ArtifactRuntimeError(
                f"Explicit ONNX provider(s) are unavailable: {unavailable}; "
                f"available={sorted(available)}"
            )
    elif device is not None:
        name = _device_provider(device)
        if allowed_names and name not in allowed_names:
            raise ArtifactRuntimeError(
                f"Device {device!r} maps to {name}, which is not allowed by the artifact."
            )
        device_spec = {"name": name, "options": {}}
        if verified_specs and _provider_spec_key(device_spec) not in verified_keys:
            raise ArtifactRuntimeError(
                f"Device {device!r} maps to an ONNX provider/options spec that is not "
                "release-verified by the artifact."
            )
        if allowed_specs and not verified_specs:
            raise ArtifactRuntimeError(
                "Artifact declares allowed ONNX providers but no release-verified provider."
            )
        if name not in available:
            raise ArtifactRuntimeError(
                f"Device {device!r} requires unavailable provider {name}; "
                f"available={sorted(available)}"
            )
        chosen = [device_spec]
    elif allowed_specs:
        chosen = [
            item
            for item in allowed_specs
            if str(item["name"]) in available and _provider_spec_key(item) in verified_keys
        ]
        if not chosen:
            raise ArtifactRuntimeError(
                "None of the artifact's allowed and release-verified ONNX providers are "
                "available; "
                f"allowed={_provider_names(allowed_specs)}, available={sorted(available)}"
            )
    else:
        defaults = list(ort.get_available_providers())
        if not defaults:
            raise ArtifactRuntimeError("ONNX Runtime reports no available execution providers.")
        chosen = [{"name": str(defaults[0]), "options": {}}]

    return chosen, warnings


def _normalize_onnx_session_options(
    options: Mapping[str, Any] | None,
    *,
    field: str,
) -> dict[str, Any] | None:
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise ArtifactRuntimeError(f"{field} must be a mapping.")
    values = dict(options)
    unknown = sorted(set(values) - _SESSION_OPTION_KEYS)
    if unknown:
        raise ArtifactRuntimeError(f"Unsupported ONNX {field} option(s): {unknown}")
    for key in (
        "intra_op_num_threads",
        "inter_op_num_threads",
        "log_severity_level",
        "log_verbosity_level",
    ):
        if values.get(key) is not None:
            value = values[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ArtifactRuntimeError(f"ONNX {field}.{key} must be an integer >= 0.")
            values[key] = int(value)
    for key in ("enable_mem_pattern", "enable_cpu_mem_arena"):
        if values.get(key) is not None:
            if not isinstance(values[key], bool):
                raise ArtifactRuntimeError(f"ONNX {field}.{key} must be a boolean.")
            values[key] = bool(values[key])

    execution = values.get("execution_mode")
    if execution is not None:
        name = str(execution).strip().lower()
        if name not in {"sequential", "parallel"}:
            raise ArtifactRuntimeError(f"Unsupported ONNX {field}.execution_mode: {execution!r}")
        values["execution_mode"] = name

    optimization = values.get("graph_optimization_level")
    if optimization is not None:
        name = str(optimization).strip().lower()
        if name not in {"disable", "basic", "extended", "all"}:
            raise ArtifactRuntimeError(
                f"Unsupported ONNX {field}.graph_optimization_level: {optimization!r}"
            )
        values["graph_optimization_level"] = name
    entries = values.get("session_config_entries")
    if entries is not None:
        if not isinstance(entries, Mapping):
            raise ArtifactRuntimeError(f"ONNX {field}.session_config_entries must be a mapping.")
        normalized_entries: dict[str, str] = {}
        for raw_key, raw_value in entries.items():
            key = str(raw_key)
            if not _SESSION_CONFIG_KEY_RE.fullmatch(key) or not isinstance(raw_value, str):
                raise ArtifactRuntimeError(
                    f"ONNX {field}.session_config_entries requires safe names and string values."
                )
            normalized_entries[key] = raw_value
        values["session_config_entries"] = normalized_entries
    return values


def resolve_onnx_session_options(
    declared: Mapping[str, Any] | None,
    requested: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve only the signed schema-v1 ONNX session-options contract."""

    declared_options = _normalize_onnx_session_options(
        declared,
        field="runtime.session_options",
    )
    if requested is None:
        return declared_options
    requested_options = _normalize_onnx_session_options(requested, field="session_options")
    if requested_options != declared_options:
        raise ArtifactRuntimeError(
            "Requested ONNX session_options must exactly match the signed "
            "runtime.session_options declaration."
        )
    return requested_options


def build_onnx_session_options(ort: Any, options: Mapping[str, Any] | None) -> Any | None:
    values = _normalize_onnx_session_options(options, field="session_options")
    if values is None:
        return None
    session_options = ort.SessionOptions()
    for key in (
        "intra_op_num_threads",
        "inter_op_num_threads",
        "log_severity_level",
        "log_verbosity_level",
    ):
        if values.get(key) is not None:
            setattr(session_options, key, int(values[key]))
    for key in ("enable_mem_pattern", "enable_cpu_mem_arena"):
        if values.get(key) is not None:
            setattr(session_options, key, bool(values[key]))
    if values.get("execution_mode") is not None:
        session_options.execution_mode = {
            "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
            "parallel": ort.ExecutionMode.ORT_PARALLEL,
        }[str(values["execution_mode"])]
    if values.get("graph_optimization_level") is not None:
        session_options.graph_optimization_level = {
            "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }[str(values["graph_optimization_level"])]
    for key, raw_value in dict(values.get("session_config_entries", {})).items():
        session_options.add_session_config_entry(key, raw_value)
    return session_options


class OnnxArtifactRuntime:
    """Manifest-driven full-detector ONNX Runtime adapter."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_contract: Mapping[str, Any],
        output_contract: Mapping[str, Any],
        allowed_providers: Any = None,
        verified_providers: Any = None,
        providers: Any = None,
        device: str | None = None,
        session_options: Mapping[str, Any] | None = None,
        expected_onnx_ir: int | None = None,
        expected_onnx_opset: int | None = None,
        ort_module: Any | None = None,
        session: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_contract = dict(input_contract)
        self.output_contract = dict(output_contract)
        if session is None:
            from pyimgano.artifacts.onnx_graph import validate_onnx_graph_contract

            try:
                graph_info = validate_onnx_graph_contract(
                    self.model_path,
                    input_contract=self.input_contract,
                    output_contract=self.output_contract,
                )
                for field, expected, actual in (
                    ("compatibility.onnx_ir", expected_onnx_ir, graph_info.ir_version),
                    (
                        "compatibility.onnx_opset",
                        expected_onnx_opset,
                        graph_info.default_opset,
                    ),
                ):
                    if expected is None:
                        continue
                    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
                        raise ValueError(f"{field} must be a positive integer.")
                    if int(expected) != int(actual):
                        raise ValueError(
                            f"{field}={expected} does not match staged graph value {actual}."
                        )
            except (OSError, TypeError, ValueError) as exc:
                raise ArtifactRuntimeError(
                    f"Staged ONNX graph failed pre-runtime validation: {exc}"
                ) from exc
            if ort_module is None:
                from pyimgano.utils.optional_deps import require

                ort_module = require(
                    "onnxruntime",
                    extra="onnx-runtime",
                    purpose="loading an ONNX artifact",
                )
            specs, provider_warnings = resolve_onnx_providers(
                ort_module,
                allowed=allowed_providers,
                verified=verified_providers,
                providers=providers,
                device=device,
            )
            names = _provider_names(specs)
            options = [dict(item["options"]) for item in specs]
            sess_options = build_onnx_session_options(ort_module, session_options)
            kwargs: dict[str, Any] = {"providers": names, "provider_options": options}
            if sess_options is not None:
                kwargs["sess_options"] = sess_options
            session = ort_module.InferenceSession(str(self.model_path), **kwargs)
        else:
            provider_warnings = []
        self.session = session
        self._validate_io_contract()
        selected = []
        get_providers = getattr(self.session, "get_providers", None)
        if callable(get_providers):
            selected = [str(item) for item in get_providers()]
        self.runtime_info = {
            "backend": "onnxruntime",
            "providers": selected,
            "selected_provider": selected[0] if selected else None,
            "warnings": list(provider_warnings),
        }

    def _validate_io_contract(self) -> None:
        inputs = list(self.session.get_inputs())
        if len(inputs) != 1:
            raise ArtifactRuntimeError(
                f"ONNX artifact must expose exactly one runtime input; got {len(inputs)}."
            )
        expected_input = str(self.input_contract.get("name", "")).strip()
        if not expected_input:
            raise ArtifactRuntimeError("input_contract.name is required for ONNX artifacts.")
        if str(inputs[0].name) != expected_input:
            raise ArtifactRuntimeError(
                f"ONNX input mismatch: manifest={expected_input!r}, graph={inputs[0].name!r}."
            )
        expected_type = _ORT_TENSOR_TYPES.get(
            str(self.input_contract.get("dtype", "")).strip().lower()
        )
        actual_type = str(getattr(inputs[0], "type", "")).strip().lower()
        if expected_type is None or actual_type != expected_type:
            raise ArtifactRuntimeError(
                f"ONNX input dtype mismatch: manifest={expected_type!r}, graph={actual_type!r}."
            )
        input_shape = _node_shape(inputs[0], field="ONNX input")
        if len(input_shape) != 4:
            raise ArtifactRuntimeError(f"ONNX image input must have rank 4; got {input_shape!r}.")
        layout = str(self.input_contract.get("layout", "")).strip().upper()
        if layout not in {"NCHW", "NHWC"}:
            raise ArtifactRuntimeError(f"Unsupported ONNX input layout: {layout!r}.")
        axes = _mapping(self.input_contract.get("dynamic_axes"))
        dynamic_batch = bool(axes.get("batch", False))
        dynamic_spatial = bool(axes.get("spatial", False))
        _validate_batch_dimension(input_shape, dynamic=dynamic_batch, field="ONNX input")
        channel_axis = 1 if layout == "NCHW" else 3
        height_axis, width_axis = (2, 3) if layout == "NCHW" else (1, 2)
        color = str(self.input_contract.get("color_space", "")).strip().upper()
        expected_channels = 1 if color == "GRAY" else 3
        if _static_dimension(input_shape[channel_axis]) != expected_channels:
            raise ArtifactRuntimeError(
                "ONNX input channel dimension does not match manifest color_space."
            )
        size = self.input_contract.get("size")
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ArtifactRuntimeError("ONNX input_contract.size must be [height, width].")
        for axis, expected, name in (
            (height_axis, int(size[0]), "height"),
            (width_axis, int(size[1]), "width"),
        ):
            actual = _static_dimension(input_shape[axis])
            if dynamic_spatial and actual is not None:
                raise ArtifactRuntimeError(
                    f"ONNX input {name} is static but manifest declares dynamic spatial axes."
                )
            if not dynamic_spatial and actual != expected:
                raise ArtifactRuntimeError(
                    f"ONNX input {name} mismatch: manifest={expected}, graph={input_shape[axis]!r}."
                )

        outputs = {str(item.name): item for item in self.session.get_outputs()}
        score = _mapping(self.output_contract.get("score"))
        score_name = str(score.get("name", "")).strip()
        if not score_name or score_name not in outputs:
            raise ArtifactRuntimeError(
                f"Declared ONNX score output {score_name!r} not present; graph outputs={sorted(outputs)}"
            )
        _validate_score_metadata(outputs[score_name], score, dynamic=dynamic_batch)
        anomaly_map = self.output_contract.get("anomaly_map")
        if isinstance(anomaly_map, Mapping):
            map_name = str(anomaly_map.get("name", "")).strip()
            if not map_name or map_name not in outputs:
                raise ArtifactRuntimeError(
                    f"Declared ONNX map output {map_name!r} not present; graph outputs={sorted(outputs)}"
                )
            _validate_map_metadata(
                outputs[map_name],
                anomaly_map,
                dynamic=dynamic_batch,
            )

    def score_and_maps(
        self, inputs: Sequence[Any], *, include_maps: bool = True
    ) -> tuple[np.ndarray, np.ndarray | list[np.ndarray] | None]:
        items = list(inputs)
        if not items:
            return np.zeros((0,), dtype=np.float32), None
        batch, source_shapes = prepare_image_batch(items, self.input_contract)
        score_contract = _mapping(self.output_contract.get("score"))
        map_contract = _mapping(self.output_contract.get("anomaly_map"))
        output_names = [str(score_contract["name"])]
        if include_maps and map_contract:
            output_names.append(str(map_contract["name"]))
        values = self.session.run(output_names, {str(self.input_contract["name"]): batch})
        scores = normalize_score_output(values[0], score_contract, batch_size=len(items))
        maps = None
        if include_maps and map_contract:
            maps = normalize_map_output(
                values[1],
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
            raise ArtifactRuntimeError("ONNX artifact has no declared anomaly-map output.")
        return maps


__all__ = [
    "OnnxArtifactRuntime",
    "build_onnx_session_options",
    "resolve_onnx_providers",
    "resolve_onnx_session_options",
]
