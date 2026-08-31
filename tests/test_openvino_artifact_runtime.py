from __future__ import annotations

import copy

import numpy as np
import pytest

from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
from pyimgano.inference.openvino_runtime import (
    OpenVINOArtifactRuntime,
    resolve_openvino_device,
)


class _Port:
    def __init__(self, name: str, *, dtype: str, shape: list[int | None]) -> None:
        self._name = name
        self.element_type = dtype
        self.partial_shape = shape

    def get_names(self):  # noqa: ANN201
        return {self._name}


class _CompiledModel:
    def __init__(self, *, inputs=None, outputs=None) -> None:  # noqa: ANN001
        self.inputs = inputs or [_Port("input", dtype="f32", shape=[None, 3, 4, 4])]
        self.outputs = outputs or [
            _Port("score", dtype="f32", shape=[None]),
            _Port("map", dtype="f32", shape=[None, 4, 4]),
        ]

    def __call__(self, values):  # noqa: ANN001
        batch = next(iter(values.values())) if isinstance(values, dict) else values[0]
        return {
            "score": batch.mean(axis=(1, 2, 3)),
            "map": batch.mean(axis=1),
        }


def _input_contract() -> dict:
    return {
        "name": "input",
        "dtype": "float32",
        "layout": "NCHW",
        "color_space": "RGB",
        "size": [4, 4],
        "dynamic_axes": {"batch": True, "spatial": False},
        "resize": {"mode": "stretch"},
        "scale": {"divisor": 255.0},
    }


def _output_contract() -> dict:
    return {
        "score": {
            "name": "score",
            "output_index": 0,
            "transform": "identity",
            "score_order": "higher_is_more_anomalous",
        },
        "anomaly_map": {
            "name": "map",
            "output_index": 1,
            "layout": "NHW",
            "resize_to_source": True,
        },
    }


_CPU = [{"name": "CPU", "options": {}}]


def _runtime(
    *,
    compiled_model: _CompiledModel | None = None,
    input_contract: dict | None = None,
    output_contract: dict | None = None,
) -> OpenVINOArtifactRuntime:
    return OpenVINOArtifactRuntime(
        "detector.xml",
        input_contract=input_contract or _input_contract(),
        output_contract=output_contract or _output_contract(),
        allowed_providers=_CPU,
        verified_providers=_CPU,
        compiled_model=compiled_model or _CompiledModel(),
        device="cpu",
    )


def test_openvino_artifact_runtime_uses_manifest_contract() -> None:
    runtime = _runtime()
    image = np.full((3, 6, 3), 255, dtype=np.uint8)

    scores, maps = runtime.score_and_maps([image])

    np.testing.assert_allclose(scores, [1.0])
    assert maps is not None and maps.shape == (1, 3, 6)
    assert runtime.runtime_info["selected_provider"] == "CPU"


def test_openvino_device_defaults_to_first_allowed_verified_intersection() -> None:
    name, spec = resolve_openvino_device(
        allowed=[
            {"name": "GPU", "options": {}},
            {"name": "CPU", "options": {}},
        ],
        verified=[{"name": "CPU", "options": {}}],
    )

    assert name == "CPU"
    assert spec == {"name": "CPU", "options": {}}


def test_explicit_openvino_device_requires_exact_allowed_and_verified_spec() -> None:
    with pytest.raises(ArtifactRuntimeError, match="not allowed"):
        resolve_openvino_device(
            allowed=_CPU,
            verified=_CPU,
            device="GPU",
        )

    with pytest.raises(ArtifactRuntimeError, match="not release-verified"):
        resolve_openvino_device(
            allowed=[*_CPU, {"name": "GPU", "options": {}}],
            verified=_CPU,
            device="gpu",
        )

    with pytest.raises(ArtifactRuntimeError, match="exact subset"):
        resolve_openvino_device(
            allowed=_CPU,
            verified=[{"name": "GPU", "options": {}}],
        )


def test_openvino_selected_device_must_be_available_before_model_read(monkeypatch) -> None:
    class Core:
        available_devices = ["CPU"]
        read_called = False

        def read_model(self, _path):  # noqa: ANN001, ANN201
            self.read_called = True
            raise AssertionError("read_model must not be reached")

    core = Core()

    def reject_require(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("injected OpenVINO core must not import the optional runtime")

    monkeypatch.setattr("pyimgano.utils.optional_deps.require", reject_require)
    with pytest.raises(ArtifactRuntimeError, match="GPU.*unavailable"):
        OpenVINOArtifactRuntime(
            "detector.xml",
            input_contract=_input_contract(),
            output_contract=_output_contract(),
            allowed_providers=[{"name": "GPU", "options": {}}],
            verified_providers=[{"name": "GPU", "options": {}}],
            core=core,
        )
    assert core.read_called is False


@pytest.mark.parametrize(
    ("input_port", "outputs", "message"),
    [
        (
            _Port("input", dtype="u8", shape=[None, 3, 4, 4]),
            None,
            "dtype mismatch",
        ),
        (
            _Port("input", dtype="f32", shape=[1, 3, 4, 4]),
            None,
            "static batch",
        ),
        (
            _Port("input", dtype="f32", shape=[None, 1, 4, 4]),
            None,
            "channel dimension",
        ),
        (
            _Port("input", dtype="f32", shape=[None, 3, 5, 4]),
            None,
            "height mismatch",
        ),
        (
            None,
            [
                _Port("score", dtype="i64", shape=[None]),
                _Port("map", dtype="f32", shape=[None, 4, 4]),
            ],
            "score output must be floating point",
        ),
        (
            None,
            [
                _Port("score", dtype="f32", shape=[None, 2]),
                _Port("map", dtype="f32", shape=[None, 4, 4]),
            ],
            r"must be \[batch\]",
        ),
        (
            None,
            [
                _Port("score", dtype="f32", shape=[None]),
                _Port("map", dtype="i64", shape=[None, 4, 4]),
            ],
            "anomaly-map output must be floating point",
        ),
        (
            None,
            [
                _Port("score", dtype="f32", shape=[None]),
                _Port("map", dtype="f32", shape=[None, 2, 4, 4]),
            ],
            "layout 'NHW'",
        ),
    ],
)
def test_openvino_artifact_runtime_rejects_metadata_contract_mismatch(
    input_port,
    outputs,
    message,
) -> None:  # noqa: ANN001
    model = _CompiledModel(
        inputs=[input_port] if input_port is not None else None,
        outputs=outputs,
    )

    with pytest.raises(ArtifactRuntimeError, match=message):
        _runtime(compiled_model=model)


def test_openvino_artifact_runtime_validates_output_index_and_score_selection() -> None:
    swapped = _CompiledModel(
        outputs=[
            _Port("map", dtype="f32", shape=[None, 4, 4]),
            _Port("score", dtype="f32", shape=[None]),
        ]
    )
    with pytest.raises(ArtifactRuntimeError, match="output_index mismatch"):
        _runtime(compiled_model=swapped)

    selected_contract = copy.deepcopy(_output_contract())
    selected_contract["score"].update({"transform": "select_index", "axis": 1, "index": 3})
    selected_model = _CompiledModel(
        outputs=[
            _Port("score", dtype="f32", shape=[None, 2]),
            _Port("map", dtype="f32", shape=[None, 4, 4]),
        ]
    )
    with pytest.raises(ArtifactRuntimeError, match="selection index 3 is invalid"):
        _runtime(compiled_model=selected_model, output_contract=selected_contract)
