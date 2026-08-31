from __future__ import annotations

import numpy as np
import pytest

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
from pyimgano.inference.onnx_runtime import (
    OnnxArtifactRuntime,
    build_onnx_session_options,
    resolve_onnx_providers,
    resolve_onnx_session_options,
)


def _write_detector_onnx(path) -> None:  # noqa: ANN001
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    tensor = onnx.TensorProto
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0),
            helper.make_node("ReduceMean", ["input"], ["map"], axes=[1], keepdims=0),
        ],
        "artifact_detector",
        [helper.make_tensor_value_info("input", tensor.FLOAT, [None, 3, 4, 4])],
        [
            helper.make_tensor_value_info("score", tensor.FLOAT, [None]),
            helper.make_tensor_value_info("map", tensor.FLOAT, [None, 4, 4]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, path)


def _input_contract() -> dict:
    return {
        "kind": "image_batch",
        "name": "input",
        "dtype": "float32",
        "layout": "NCHW",
        "color_space": "RGB",
        "size": [4, 4],
        "dynamic_axes": {"batch": True, "spatial": False},
        "resize": {"mode": "stretch", "interpolation": "bilinear"},
        "scale": {"divisor": 255.0},
    }


def _output_contract() -> dict:
    return {
        "score": {
            "name": "score",
            "transform": "identity",
            "score_order": "higher_is_more_anomalous",
        },
        "anomaly_map": {
            "name": "map",
            "layout": "NHW",
            "transform": "identity",
            "resize_to_source": True,
        },
    }


def test_onnx_artifact_runtime_runs_score_and_map_in_one_session(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    model_path = tmp_path / "detector.onnx"
    _write_detector_onnx(model_path)
    runtime = OnnxArtifactRuntime(
        model_path,
        input_contract=_input_contract(),
        output_contract=_output_contract(),
        allowed_providers=[{"name": "CPUExecutionProvider", "options": {}}],
        verified_providers=[{"name": "CPUExecutionProvider", "options": {}}],
    )
    images = [np.full((6, 8, 3), 255, dtype=np.uint8), np.zeros((6, 8, 3), dtype=np.uint8)]

    scores, maps = runtime.score_and_maps(images)

    np.testing.assert_allclose(scores, [1.0, 0.0], atol=1e-6)
    assert maps is not None and maps.shape == (2, 6, 8)
    np.testing.assert_allclose(maps[0], 1.0, atol=1e-6)
    assert runtime.runtime_info["selected_provider"] == "CPUExecutionProvider"


class _FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]


class _ConstructorSpyOrt:
    constructor_calls = 0

    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider"]

    @classmethod
    def InferenceSession(cls, *_args, **_kwargs):  # noqa: N802, ANN206
        cls.constructor_calls += 1
        raise AssertionError("InferenceSession must not run for an invalid staged graph")


@pytest.mark.parametrize("invalid", ["custom-domain", "unknown-operator", "ir", "opset", "io"])
def test_onnx_graph_contract_is_rejected_before_session_construction(tmp_path, invalid) -> None:
    onnx = pytest.importorskip("onnx")
    model_path = tmp_path / "detector.onnx"
    _write_detector_onnx(model_path)
    model = onnx.load_model(str(model_path), load_external_data=False)
    output_contract = _output_contract()
    if invalid == "custom-domain":
        model.graph.node[0].domain = "vendor.custom"
    elif invalid == "unknown-operator":
        model.graph.node[0].op_type = "NotARealOnnxOp"
    elif invalid == "ir":
        model.ir_version = 11
    elif invalid == "opset":
        model.opset_import[0].version = 22
    else:
        output_contract["score"]["name"] = "missing-score"
    onnx.save_model(model, str(model_path))

    _ConstructorSpyOrt.constructor_calls = 0
    with pytest.raises(ArtifactRuntimeError, match="pre-runtime validation"):
        OnnxArtifactRuntime(
            model_path,
            input_contract=_input_contract(),
            output_contract=output_contract,
            allowed_providers=[{"name": "CPUExecutionProvider", "options": {}}],
            verified_providers=[{"name": "CPUExecutionProvider", "options": {}}],
            ort_module=_ConstructorSpyOrt,
        )
    assert _ConstructorSpyOrt.constructor_calls == 0


@pytest.mark.parametrize(
    ("expected_ir", "expected_opset", "field"),
    [(8, 13, "onnx_ir"), (9, 14, "onnx_opset")],
)
def test_onnx_manifest_metadata_mismatch_is_rejected_before_session(
    tmp_path, expected_ir, expected_opset, field
) -> None:
    model_path = tmp_path / "detector.onnx"
    _write_detector_onnx(model_path)
    _ConstructorSpyOrt.constructor_calls = 0

    with pytest.raises(ArtifactRuntimeError, match=field):
        OnnxArtifactRuntime(
            model_path,
            input_contract=_input_contract(),
            output_contract=_output_contract(),
            allowed_providers=[{"name": "CPUExecutionProvider", "options": {}}],
            verified_providers=[{"name": "CPUExecutionProvider", "options": {}}],
            expected_onnx_ir=expected_ir,
            expected_onnx_opset=expected_opset,
            ort_module=_ConstructorSpyOrt,
        )

    assert _ConstructorSpyOrt.constructor_calls == 0


def test_explicit_onnx_provider_must_be_allowed_and_available() -> None:
    allowed = [{"name": "CPUExecutionProvider", "options": {}}]
    with pytest.raises(ArtifactRuntimeError, match="not allowed"):
        resolve_onnx_providers(
            _FakeOrt,
            allowed=allowed,
            providers=[{"name": "CUDAExecutionProvider", "options": {}}],
        )
    with pytest.raises(ArtifactRuntimeError, match="either providers or device"):
        resolve_onnx_providers(
            _FakeOrt,
            allowed=allowed,
            providers=["CPUExecutionProvider"],
            device="cpu",
        )


class _Node:
    def __init__(self, name, type_, shape):  # noqa: ANN001
        self.name = name
        self.type = type_
        self.shape = shape


class _MetadataSession:
    def __init__(self, input_node, outputs):  # noqa: ANN001
        self._input = input_node
        self._outputs = outputs

    def get_inputs(self):  # noqa: ANN201
        return [self._input]

    def get_outputs(self):  # noqa: ANN201
        return list(self._outputs)

    def get_providers(self):  # noqa: ANN201
        return ["CPUExecutionProvider"]


@pytest.mark.parametrize(
    ("input_node", "outputs", "message"),
    [
        (
            _Node("input", "tensor(uint8)", ["batch", 3, 4, 4]),
            [
                _Node("score", "tensor(float)", ["batch"]),
                _Node("map", "tensor(float)", ["batch", 4, 4]),
            ],
            "dtype mismatch",
        ),
        (
            _Node("input", "tensor(float)", [1, 3, 4, 4]),
            [_Node("score", "tensor(float)", [1]), _Node("map", "tensor(float)", [1, 4, 4])],
            "static batch",
        ),
        (
            _Node("input", "tensor(float)", ["batch", 3, 5, 4]),
            [
                _Node("score", "tensor(float)", ["batch"]),
                _Node("map", "tensor(float)", ["batch", 4, 4]),
            ],
            "height mismatch",
        ),
        (
            _Node("input", "tensor(float)", ["batch", 3, 4, 4]),
            [
                _Node("score", "tensor(float)", ["batch", 2]),
                _Node("map", "tensor(float)", ["batch", 4, 4]),
            ],
            r"must be \[batch\]",
        ),
        (
            _Node("input", "tensor(float)", ["batch", 3, 4, 4]),
            [
                _Node("score", "tensor(float)", ["batch"]),
                _Node("map", "tensor(float)", ["batch", 2, 4, 4]),
            ],
            "layout 'NHW'",
        ),
    ],
)
def test_onnx_artifact_runtime_rejects_metadata_contract_mismatch(
    tmp_path, input_node, outputs, message
) -> None:
    session = _MetadataSession(input_node, outputs)
    with pytest.raises(ArtifactRuntimeError, match=message):
        OnnxArtifactRuntime(
            tmp_path / "detector.onnx",
            input_contract=_input_contract(),
            output_contract=_output_contract(),
            session=session,
        )


class _SessionOptions:
    def __init__(self) -> None:
        self.entries = {}

    def add_session_config_entry(self, key, value):  # noqa: ANN001
        self.entries[key] = value


class _OptionOrt:
    SessionOptions = _SessionOptions

    class ExecutionMode:
        ORT_SEQUENTIAL = "sequential"
        ORT_PARALLEL = "parallel"

    class GraphOptimizationLevel:
        ORT_DISABLE_ALL = "disable"
        ORT_ENABLE_BASIC = "basic"
        ORT_ENABLE_EXTENDED = "extended"
        ORT_ENABLE_ALL = "all"


def test_onnx_session_options_apply_safe_config_entries_and_strict_types() -> None:
    options = build_onnx_session_options(
        _OptionOrt,
        {
            "intra_op_num_threads": 2,
            "enable_mem_pattern": False,
            "session_config_entries": {"session.use_env_allocators": "1"},
        },
    )
    assert options.intra_op_num_threads == 2
    assert options.enable_mem_pattern is False
    assert options.entries == {"session.use_env_allocators": "1"}

    with pytest.raises(ArtifactRuntimeError, match="boolean"):
        build_onnx_session_options(_OptionOrt, {"enable_mem_pattern": "false"})
    with pytest.raises(ArtifactRuntimeError, match="safe names"):
        build_onnx_session_options(
            _OptionOrt,
            {"session_config_entries": {"bad/key": "value"}},
        )


def test_onnx_session_options_resolve_to_signed_declaration() -> None:
    declared = {
        "execution_mode": "sequential",
        "graph_optimization_level": "extended",
        "intra_op_num_threads": 2,
        "session_config_entries": {"session.use_env_allocators": "1"},
    }

    assert resolve_onnx_session_options(declared, None) == declared
    assert (
        resolve_onnx_session_options(
            declared,
            {
                "session_config_entries": {"session.use_env_allocators": "1"},
                "intra_op_num_threads": 2,
                "graph_optimization_level": "EXTENDED",
                "execution_mode": "SEQUENTIAL",
            },
        )
        == declared
    )


def test_onnx_session_options_reject_unsigned_or_different_override() -> None:
    with pytest.raises(ArtifactRuntimeError, match="exactly match"):
        resolve_onnx_session_options(
            {"intra_op_num_threads": 2},
            {"intra_op_num_threads": 4},
        )
    with pytest.raises(ArtifactRuntimeError, match="exactly match"):
        resolve_onnx_session_options(None, {"enable_mem_pattern": True})
    with pytest.raises(ArtifactRuntimeError, match="runtime.session_options must be a mapping"):
        resolve_onnx_session_options([("enable_mem_pattern", True)], None)  # type: ignore[arg-type]
    with pytest.raises(ArtifactRuntimeError, match="session_options must be a mapping"):
        resolve_onnx_session_options({}, "unsafe")  # type: ignore[arg-type]
