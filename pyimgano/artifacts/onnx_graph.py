from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_MAX_ONNX_BYTES = 512 * 1024 * 1024
_MAX_ONNX_PROTOBUF_MESSAGES = 1_000_000
_MIN_ONNX_IR = 3
_MAX_ONNX_IR = 10
_ONNX_OPSET_RANGES = {
    "ai.onnx": (7, 21),
    "ai.onnx.ml": (1, 4),
}
_FLOAT_DTYPES = {"float16", "float32", "float64"}


@dataclass(frozen=True)
class ONNXGraphContractInfo:
    ir_version: int
    default_opset: int


def _onnx_dtype_name(onnx: Any, elem_type: int) -> str:
    try:
        return str(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).name)
    except Exception as exc:
        raise ValueError(f"Unsupported ONNX tensor dtype enum: {elem_type}") from exc


def _iter_onnx_messages(model: Any):
    stack = [model]
    visited = 0
    while stack:
        message = stack.pop()
        visited += 1
        if visited > _MAX_ONNX_PROTOBUF_MESSAGES:
            raise ValueError("ONNX protobuf contains too many nested messages.")
        descriptor = getattr(message, "DESCRIPTOR", None)
        if descriptor is None:
            continue
        yield message
        for field, value in message.ListFields():
            if getattr(field, "message_type", None) is None:
                continue
            repeated = getattr(field, "is_repeated", None)
            if repeated is None:
                repeated = int(getattr(field, "label", 0)) == int(
                    getattr(field, "LABEL_REPEATED", 3)
                )
            if repeated:
                stack.extend(reversed(value))
            else:
                stack.append(value)


def _canonical_onnx_domain(domain: str) -> str:
    return "ai.onnx" if domain in {"", "ai.onnx"} else domain


def _validate_opset(domain: str, version: int, *, field: str) -> str:
    canonical = _canonical_onnx_domain(domain)
    bounds = _ONNX_OPSET_RANGES.get(canonical)
    if bounds is None:
        raise ValueError(f"Unsupported/custom ONNX operator domain: {domain!r}.")
    minimum, maximum = bounds
    if not minimum <= version <= maximum:
        raise ValueError(
            f"Unsupported ONNX opset version for {canonical!r} at {field}: {version}; "
            f"supported range is {minimum}..{maximum}."
        )
    return canonical


def _validate_domains_and_opsets(model: Any) -> int:
    ir_version = int(model.ir_version)
    if not _MIN_ONNX_IR <= ir_version <= _MAX_ONNX_IR:
        raise ValueError(
            f"Unsupported ONNX IR version: {ir_version}; "
            f"supported range is {_MIN_ONNX_IR}..{_MAX_ONNX_IR}."
        )

    model_opsets: dict[str, int] = {}
    for index, opset in enumerate(model.opset_import):
        raw_domain = str(opset.domain or "")
        version = int(opset.version)
        canonical = _validate_opset(raw_domain, version, field=f"model.opset_import[{index}]")
        if canonical in model_opsets:
            raise ValueError(f"Duplicate ONNX opset import for domain {canonical!r}.")
        model_opsets[canonical] = version
    if "ai.onnx" not in model_opsets:
        raise ValueError("ONNX model must declare exactly one default ai.onnx opset import.")

    for message in _iter_onnx_messages(model):
        full_name = str(getattr(message.DESCRIPTOR, "full_name", ""))
        if full_name == "onnx.NodeProto":
            domain = str(message.domain or "")
            if _canonical_onnx_domain(domain) not in _ONNX_OPSET_RANGES:
                raise ValueError(
                    f"Unsupported/custom ONNX operator domain on node "
                    f"{str(message.name or message.op_type)!r}: {domain!r}."
                )
        elif full_name == "onnx.FunctionProto":
            domain = str(message.domain or "")
            if _canonical_onnx_domain(domain) not in _ONNX_OPSET_RANGES:
                raise ValueError(f"Unsupported/custom ONNX function domain: {domain!r}.")
        elif full_name == "onnx.OperatorSetIdProto":
            _validate_opset(
                str(message.domain or ""),
                int(message.version),
                field="nested opset import",
            )
    return model_opsets["ai.onnx"]


def _tensor_metadata(
    value_info: Any,
    *,
    field: str,
    onnx: Any,
) -> tuple[str, list[tuple[int | None, str | None]]]:
    type_proto = value_info.type
    if type_proto.WhichOneof("value") != "tensor_type":
        raise ValueError(f"{field} must be an ONNX tensor.")
    tensor_type = type_proto.tensor_type
    if int(tensor_type.elem_type) == 0:
        raise ValueError(f"{field} must declare a tensor dtype.")
    if not tensor_type.HasField("shape"):
        raise ValueError(f"{field} must declare tensor shape metadata.")
    dimensions: list[tuple[int | None, str | None]] = []
    for index, dimension in enumerate(tensor_type.shape.dim):
        if dimension.HasField("dim_value"):
            value = int(dimension.dim_value)
            if value <= 0:
                raise ValueError(f"{field} dimension {index} must be positive when static.")
            dimensions.append((value, None))
        elif dimension.HasField("dim_param") and str(dimension.dim_param):
            dimensions.append((None, str(dimension.dim_param)))
        else:
            dimensions.append((None, None))
    return _onnx_dtype_name(onnx, tensor_type.elem_type), dimensions


def _dimension_values(shape: list[tuple[int | None, str | None]]) -> list[int | None]:
    return [value for value, _symbol in shape]


def _validate_matching_batch(
    input_batch: tuple[int | None, str | None],
    output_batch: tuple[int | None, str | None],
    *,
    field: str,
) -> None:
    input_value, input_symbol = input_batch
    output_value, output_symbol = output_batch
    if (input_value is None) != (output_value is None):
        raise ValueError(f"{field} batch dimension must match the ONNX input batch dimension.")
    if input_value is not None and output_value != input_value:
        raise ValueError(f"{field} batch dimension must match the ONNX input batch dimension.")
    if input_symbol and output_symbol and input_symbol != output_symbol:
        raise ValueError(
            f"{field} batch symbol {output_symbol!r} does not match input {input_symbol!r}."
        )


def _validate_float_output(
    value_info: Any,
    *,
    field: str,
    input_batch: tuple[int | None, str | None],
    onnx: Any,
) -> list[tuple[int | None, str | None]]:
    dtype, shape = _tensor_metadata(value_info, field=field, onnx=onnx)
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(f"{field} must be floating point; got {dtype!r}.")
    if not shape:
        raise ValueError(f"{field} must include a batch dimension.")
    _validate_matching_batch(input_batch, shape[0], field=field)
    return shape


def _validate_score_output(
    value_info: Any,
    contract: Mapping[str, Any],
    *,
    input_batch: tuple[int | None, str | None],
    onnx: Any,
) -> None:
    shape = _validate_float_output(
        value_info,
        field="ONNX score output",
        input_batch=input_batch,
        onnx=onnx,
    )
    values = _dimension_values(shape)
    transform = str(contract["transform"])
    if transform in {"select_index", "softmax_select"}:
        if len(shape) != 2:
            raise ValueError(
                f"ONNX score transform {transform!r} requires rank-2 [batch, classes]; "
                f"got {values!r}."
            )
        raw_axis = int(contract["axis"])
        axis = raw_axis + len(shape) if raw_axis < 0 else raw_axis
        if not 0 <= axis < len(shape):
            raise ValueError(f"ONNX score selection axis {raw_axis} is out of range.")
        if axis != 1:
            raise ValueError("ONNX score selection must use the non-batch axis 1/-1.")
        index = int(contract["index"])
        classes = shape[axis][0]
        if index < 0 or (classes is not None and index >= classes):
            raise ValueError(f"ONNX score selection index {index} is invalid for shape {values!r}.")
        return
    if len(shape) == 1:
        return
    if len(shape) == 2 and shape[1][0] == 1:
        return
    raise ValueError(
        "ONNX score output must be [batch] or [batch, 1] unless a class-selection "
        f"transform is declared; got {values!r}."
    )


def _validate_map_output(
    value_info: Any,
    contract: Mapping[str, Any],
    *,
    input_batch: tuple[int | None, str | None],
    onnx: Any,
) -> None:
    shape = _validate_float_output(
        value_info,
        field="ONNX anomaly-map output",
        input_batch=input_batch,
        onnx=onnx,
    )
    values = _dimension_values(shape)
    layout = str(contract["layout"])
    expected_rank = {"NHW": 3, "NCHW": 4, "NHWC": 4}[layout]
    if len(shape) != expected_rank:
        raise ValueError(f"ONNX anomaly-map layout {layout!r} does not match shape {values!r}.")
    if layout == "NHW":
        if contract.get("channel") is not None:
            raise ValueError("NHW anomaly maps must not declare a channel index.")
        return
    channel_axis = 1 if layout == "NCHW" else 3
    channels = shape[channel_axis][0]
    channel = contract.get("channel")
    if channel is None:
        if channels != 1:
            raise ValueError(
                "Multi-channel or dynamic-channel ONNX anomaly maps require channel selection."
            )
        return
    selected = int(channel)
    if selected < 0 or (channels is not None and selected >= channels):
        raise ValueError(f"ONNX anomaly-map channel {selected} is invalid for shape {values!r}.")


def _validate_image_input(
    model: Any,
    input_contract: Mapping[str, Any],
    *,
    onnx: Any,
) -> tuple[Any, list[tuple[int | None, str | None]]]:
    initializer_names = {str(item.name) for item in model.graph.initializer}
    initializer_names.update(
        str(item.values.name) for item in model.graph.sparse_initializer if str(item.values.name)
    )
    runtime_inputs = [item for item in model.graph.input if str(item.name) not in initializer_names]
    if len(runtime_inputs) != 1:
        raise ValueError(
            "ONNX artifact schema v1 requires exactly one non-initializer runtime input; "
            f"found {len(runtime_inputs)}."
        )
    graph_input = runtime_inputs[0]
    if str(graph_input.name) != str(input_contract["name"]):
        raise ValueError(
            f"ONNX input name mismatch: graph={graph_input.name!r}, "
            f"contract={input_contract['name']!r}."
        )
    graph_dtype, graph_shape = _tensor_metadata(
        graph_input,
        field="ONNX image input",
        onnx=onnx,
    )
    if graph_dtype != str(input_contract["dtype"]):
        raise ValueError(
            f"ONNX input dtype mismatch: graph={graph_dtype}, contract={input_contract['dtype']}."
        )

    shape = _dimension_values(graph_shape)
    if len(shape) != 4:
        raise ValueError(f"ONNX image input must have rank 4, got shape={shape!r}.")
    layout = str(input_contract["layout"])
    height_axis, width_axis = (2, 3) if layout == "NCHW" else (1, 2)
    channels_axis = 1 if layout == "NCHW" else 3
    channels = 1 if input_contract["color_space"] == "GRAY" else 3
    if shape[channels_axis] != channels:
        raise ValueError("ONNX input channel count does not match contract color_space.")
    expected_size = list(input_contract["size"])
    dynamic_axes = input_contract.get("dynamic_axes")
    dynamic_axes = dict(dynamic_axes) if isinstance(dynamic_axes, Mapping) else {}
    dynamic_spatial = bool(dynamic_axes.get("spatial", False))
    for axis, expected in ((height_axis, expected_size[0]), (width_axis, expected_size[1])):
        if dynamic_spatial and shape[axis] is not None:
            raise ValueError("Contract declares dynamic spatial axes but ONNX graph is static.")
        if not dynamic_spatial and shape[axis] != int(expected):
            raise ValueError("ONNX graph spatial dimensions do not match contract input.size.")
    dynamic_batch = bool(dynamic_axes.get("batch", False))
    if shape[0] is None and not dynamic_batch:
        raise ValueError("ONNX graph has dynamic batch but contract.dynamic_batch is false.")
    if shape[0] is not None and dynamic_batch:
        raise ValueError("Contract declares dynamic batch but ONNX graph batch is static.")
    return graph_input, graph_shape


def validate_onnx_model_contract(
    model: Any,
    *,
    input_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
    onnx_module: Any,
) -> ONNXGraphContractInfo:
    """Validate a parsed schema-v1 graph without constructing an ORT session."""

    default_opset = _validate_domains_and_opsets(model)
    _graph_input, graph_shape = _validate_image_input(
        model,
        input_contract,
        onnx=onnx_module,
    )
    graph_outputs = {str(item.name): item for item in model.graph.output}
    if len(graph_outputs) != len(model.graph.output):
        raise ValueError("ONNX graph contains duplicate output names.")

    if "score" in output_contract:
        score_contract = dict(output_contract["score"])
        score_name = str(score_contract["name"])
        if score_name not in graph_outputs:
            raise ValueError(f"Declared ONNX score output not found in graph: {score_name!r}.")
        _validate_score_output(
            graph_outputs[score_name],
            score_contract,
            input_batch=graph_shape[0],
            onnx=onnx_module,
        )
        map_contract = output_contract.get("anomaly_map")
        if isinstance(map_contract, Mapping):
            map_name = str(map_contract["name"])
            if map_name not in graph_outputs:
                raise ValueError(
                    f"Declared ONNX anomaly_map output not found in graph: {map_name!r}."
                )
            _validate_map_output(
                graph_outputs[map_name],
                map_contract,
                input_batch=graph_shape[0],
                onnx=onnx_module,
            )
    elif output_contract.get("kind") == "feature_matrix":
        output_name = str(output_contract.get("name", ""))
        if not output_name or output_name not in graph_outputs:
            raise ValueError(f"Declared ONNX feature output not found in graph: {output_name!r}.")
        feature_shape = _validate_float_output(
            graph_outputs[output_name],
            field="ONNX feature output",
            input_batch=graph_shape[0],
            onnx=onnx_module,
        )
        if len(feature_shape) < 2:
            raise ValueError("ONNX feature output must have rank >= 2 including batch.")
    else:
        raise ValueError("Unsupported ONNX schema-v1 output contract.")

    return ONNXGraphContractInfo(
        ir_version=int(model.ir_version),
        default_opset=int(default_opset),
    )


def validate_onnx_graph_contract(
    model_path: str | Path,
    *,
    input_contract: Mapping[str, Any],
    output_contract: Mapping[str, Any],
) -> ONNXGraphContractInfo:
    """Parse and validate a staged ONNX path before any runtime constructor runs."""

    path = Path(model_path)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"Cannot inspect staged ONNX model: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Staged ONNX model must be a regular file: {path}")
    if int(metadata.st_size) > _MAX_ONNX_BYTES:
        raise ValueError(f"Staged ONNX model exceeds the {_MAX_ONNX_BYTES}-byte validation limit.")
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - required by onnx-runtime extra
        raise ImportError("ONNX graph validation requires pyimgano[onnx-runtime].") from exc
    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise ValueError(f"Cannot parse staged ONNX model: {exc}") from exc
    info = validate_onnx_model_contract(
        model,
        input_contract=input_contract,
        output_contract=output_contract,
        onnx_module=onnx,
    )
    try:
        # The path form resolves only the already verified/staged external-data
        # closure and rejects unknown standard-domain operators/functions before
        # ONNX Runtime parses or optimizes the graph.
        onnx.checker.check_model(str(path), full_check=False)
    except Exception as exc:
        raise ValueError(f"ONNX graph failed schema/operator validation: {exc}") from exc
    return info


__all__ = [
    "ONNXGraphContractInfo",
    "validate_onnx_graph_contract",
    "validate_onnx_model_contract",
]
