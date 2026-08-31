from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pyimgano.inference import infer, load_artifact
from pyimgano.inference.artifact_runtime import ArtifactRuntime, ArtifactRuntimeError


class _Backend:
    runtime_info = {"backend": "test", "selected_provider": "TestProvider"}

    def __init__(self) -> None:
        self.calls = 0

    def score_and_maps(self, inputs, *, include_maps=True):  # noqa: ANN001
        self.calls += 1
        scores = np.asarray([float(np.asarray(item).mean()) for item in inputs], dtype=np.float32)
        maps = None
        if include_maps:
            maps = np.stack(
                [
                    np.full(np.asarray(item).shape[:2], score, dtype=np.float32)
                    for item, score in zip(inputs, scores)
                ]
            )
        return scores, maps


def _manifest(*, maps: bool = True) -> dict:
    output = {
        "score": {
            "name": "score",
            "transform": "identity",
            "score_order": "higher_is_more_anomalous",
        }
    }
    if maps:
        output["anomaly_map"] = {"name": "map", "layout": "NHW"}
    return {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "artifact_id": "sha256:test",
        "layout": "single_graph",
        "runtime": {"backend": "test"},
        "output_contract": output,
    }


def test_artifact_runtime_is_detector_compatible_and_uses_one_call(tmp_path) -> None:
    backend = _Backend()
    runtime = ArtifactRuntime(
        backend,
        manifest=_manifest(),
        infer_config={"postprocess": {"image_threshold": {"threshold": 10.0}}},
        artifact_root=tmp_path,
    )
    images = [np.full((3, 4, 3), 5, dtype=np.uint8), np.full((3, 4, 3), 20, dtype=np.uint8)]

    scores, maps = runtime.score_and_maps(images)

    np.testing.assert_allclose(scores, [5.0, 20.0])
    assert maps is not None and maps.shape == (2, 3, 4)
    assert backend.calls == 1
    np.testing.assert_array_equal(runtime.predict(images), [0, 1])


def test_public_infer_merges_artifact_defaults_and_allows_explicit_false(tmp_path) -> None:
    runtime = ArtifactRuntime(
        _Backend(),
        manifest=_manifest(),
        infer_config={
            "adaptation": {"save_maps": True},
            "postprocess": {"image_threshold": {"threshold": 10.0}},
        },
        artifact_root=tmp_path,
    )
    image = np.full((3, 4, 3), 20, dtype=np.uint8)

    inherited = infer(runtime, [image], input_format="rgb_u8_hwc")
    overridden = infer(runtime, [image], input_format="rgb_u8_hwc", include_maps=False)

    assert inherited[0].label == 1
    assert inherited[0].anomaly_map is not None
    assert overridden[0].anomaly_map is None


def test_score_only_artifact_has_no_label_and_predict_fails(tmp_path) -> None:
    runtime = ArtifactRuntime(
        _Backend(),
        manifest=_manifest(maps=False),
        infer_config={"postprocess": {"image_threshold": {"threshold": None}}},
        artifact_root=tmp_path,
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    assert infer(runtime, [image], input_format="rgb_u8_hwc")[0].label is None
    with pytest.raises(ArtifactRuntimeError, match="score-only"):
        runtime.predict([image])


def test_public_loader_verifies_stages_and_runs_real_onnx_artifact(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from pyimgano.artifacts import write_artifact_manifest

    artifact_root = tmp_path / "artifact"
    model_dir = artifact_root / "model"
    verification_dir = artifact_root / "verification"
    model_dir.mkdir(parents=True)
    verification_dir.mkdir()
    model_path = model_dir / "detector.onnx"
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0),
            onnx.helper.make_node("ReduceMean", ["input"], ["map"], axes=[1], keepdims=0),
        ],
        "artifact_detector",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, 3, 4, 4])],
        [
            onnx.helper.make_tensor_value_info("score", onnx.TensorProto.FLOAT, [None]),
            onnx.helper.make_tensor_value_info("map", onnx.TensorProto.FLOAT, [None, 4, 4]),
        ],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, model_path)
    verification_path = verification_dir / "runtime.json"
    verification_path.write_bytes(b"{}")

    def _attachment(path, *, role, format, serialization):  # noqa: ANN001
        data = path.read_bytes()
        return {
            "path": path.relative_to(artifact_root).as_posix(),
            "role": role,
            "format": format,
            "serialization": serialization,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    policy = {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {
            "image_threshold": {
                "threshold": 0.5,
                "score_order": "higher_is_more_anomalous",
            },
            "map_postprocess": None,
        },
    }
    payload = {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "layout": "single_graph",
        "runtime": {
            "backend": "onnxruntime",
            "entrypoint": "model/detector.onnx",
            "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
        },
        "input_contract": {
            "kind": "image_batch",
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [4, 4],
            "dynamic_axes": {"batch": True},
            "resize": {"mode": "stretch", "interpolation": "bilinear"},
            "scale": {"divisor": 255.0},
        },
        "output_contract": {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            },
            "anomaly_map": {"name": "map", "layout": "NHW", "resize_to_source": True},
        },
        "components": [
            _attachment(
                model_path,
                role="runtime_model",
                format="onnx",
                serialization="onnx",
            )
        ],
        "policy_ref": {"path": "infer_config.json"},
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": ["linux-x86_64"],
            "runtime_versions": {"onnxruntime": ">=1.17,<2"},
            "codecs": [],
        },
        "verification": {
            "level": "runtime_smoke",
            "report": {
                "path": "verification/runtime.json",
                "size_bytes": 2,
                "sha256": hashlib.sha256(b"{}").hexdigest(),
            },
        },
    }
    write_artifact_manifest(artifact_root, payload, policy)

    runtime = load_artifact(artifact_root, format="onnx", backend="onnxruntime")
    staged_model = runtime.backend_runtime.model_path
    assert staged_model != model_path
    assert staged_model.is_file()
    result = infer(
        runtime,
        [np.full((6, 8, 3), 255, dtype=np.uint8)],
        input_format="rgb_u8_hwc",
        include_maps=True,
    )[0]
    assert result.score == pytest.approx(1.0)
    assert result.label == 1
    assert result.anomaly_map is not None and result.anomaly_map.shape == (6, 8)
    assert runtime.runtime_info["selected_provider"] == "CPUExecutionProvider"

    staging_root = staged_model.parent.parent
    runtime.close()
    assert not staging_root.exists()
