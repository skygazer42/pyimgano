from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _images(count: int = 12) -> list[np.ndarray]:
    yy = np.arange(10, dtype=np.uint16)[:, None]
    xx = np.arange(11, dtype=np.uint16)[None, :]
    values: list[np.ndarray] = []
    for index in range(count):
        image = np.empty((10, 11, 3), dtype=np.uint8)
        image[..., 0] = ((index * 17 + xx * 3) % 256).astype(np.uint8)
        image[..., 1] = ((yy * 7 + index * 5) % 256).astype(np.uint8)
        image[..., 2] = ((xx * 5 + yy * 11 + index) % 256).astype(np.uint8)
        values.append(image)
    return values


def _write_onnx_embedding(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 8, 8])
    output_value = helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, 3])
    bias = numpy_helper.from_array(
        np.asarray([0.125, -0.25, 0.5], dtype=np.float32).reshape(1, 3, 1, 1),
        name="bias",
    )
    add = helper.make_node("Add", ["input", "bias"], ["adjusted"])
    reduce = helper.make_node(
        "ReduceMean",
        ["adjusted"],
        ["embedding"],
        axes=[2, 3],
        keepdims=0,
    )
    graph = helper.make_graph(
        [add, reduce],
        "embedding",
        [input_value],
        [output_value],
        initializer=[bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    (path.parent / "embedding-data").mkdir()
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="embedding-data/bias.bin",
        size_threshold=0,
    )


def _write_torchscript_embedding(path: Path) -> None:
    torch = pytest.importorskip("torch")

    class Embedding(torch.nn.Module):
        def forward(self, inputs):  # noqa: ANN001, ANN201
            return inputs.mean(dim=(2, 3))

    traced = torch.jit.trace(Embedding().eval(), torch.zeros(1, 3, 8, 8))
    traced.save(str(path))


def _config(model_name: str, *, onnx: bool) -> Any:
    model_kwargs: dict[str, Any] = {
        "batch_size": 3,
        "image_size": 8,
        "core_kwargs": {"n_jobs": 1},
    }
    if onnx:
        model_kwargs["providers"] = ["CPUExecutionProvider"]
    return SimpleNamespace(
        seed=7,
        model=SimpleNamespace(
            name=model_name,
            model_kwargs=model_kwargs,
            device="cpu",
            contamination=0.1,
            pretrained=False,
            preset=None,
        ),
    )


def _state_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).lower())
            keys.update(_state_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_state_keys(item))
    return keys


def _fresh_process_scores(
    artifact: Path,
    inputs_path: Path,
    *,
    trusted: bool,
    cwd: Path,
) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    code = """
import json
import sys
import numpy as np
from pyimgano.inference import load_artifact

artifact, inputs_path, trusted = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
inputs = list(np.load(inputs_path, allow_pickle=False))
runtime = load_artifact(artifact, trust_checkpoint=trusted)
try:
    print(json.dumps({
        "scores": runtime.decision_function(inputs).tolist(),
        "backend_type": type(runtime.backend_runtime).__name__,
        "threshold": runtime.threshold_,
        "component_backend": runtime.runtime_info["component_runtime"]["backend"],
    }, sort_keys=True))
finally:
    runtime.close()
"""
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root) if not existing else str(source_root) + os.pathsep + existing
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(artifact),
            str(inputs_path),
            "1" if trusted else "0",
        ],
        cwd=cwd,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


@pytest.mark.parametrize(
    ("model_name", "format_name", "graph_suffix", "component_backend"),
    [
        ("vision_onnx_ecod", "onnx", ".onnx", "onnxruntime"),
        (
            "vision_torchscript_ecod",
            "torchscript",
            ".pt",
            "torchscript",
        ),
    ],
)
def test_ecod_composite_export_relocate_delete_source_and_fresh_load_score_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    format_name: str,
    graph_suffix: str,
    component_backend: str,
) -> None:
    if format_name == "onnx":
        pytest.importorskip("onnxruntime")
    else:
        pytest.importorskip("torch")
    from pyimgano.artifacts import load_artifact_manifest, verify_artifact_files
    from pyimgano.artifacts.compatibility import current_platform_tag
    from pyimgano.exporting import ArtifactFormat, NativeExportContext, get_export_adapter
    from pyimgano.inference import load_artifact
    from pyimgano.inference.artifact_runtime import ArtifactRuntimeError
    from pyimgano.inference.composite_runtime import CompositeArtifactRuntime
    from pyimgano.models.registry import create_model
    from pyimgano.serialization.safe_checkpoint import load_safe_checkpoint
    from pyimgano.services.checkpoint_certification_service import (
        certify_checkpoint_for_export,
    )

    source = tmp_path / f"source graph{graph_suffix}"
    if format_name == "onnx":
        _write_onnx_embedding(source)
    else:
        _write_torchscript_embedding(source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    kwargs: dict[str, Any] = {
        "checkpoint_path": str(source),
        "device": "cpu",
        "batch_size": 3,
        "image_size": 8,
        "core_kwargs": {"n_jobs": 1},
    }
    if format_name == "onnx":
        kwargs["providers"] = ["CPUExecutionProvider"]
    detector = create_model(model_name, **kwargs)
    training = _images()
    detector.fit(training)
    probe_inputs = training[:5]
    reference = np.asarray(detector.decision_function(probe_inputs), dtype=np.float64)
    threshold = float(detector.threshold_)

    fitted_checkpoint = tmp_path / "fitted source.pyim"
    fitted_checkpoint.write_bytes(b"uncertified-placeholder")
    contract = certify_checkpoint_for_export(
        detector,
        fitted_checkpoint,
        config=_config(model_name, onnx=format_name == "onnx"),
    )
    assert contract is not None and contract.strict_exportable

    policy = {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {
            "image_threshold": {
                "threshold": threshold,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }
    context = NativeExportContext(
        model_name=model_name,
        model_kwargs={},
        category="瓶 身",
        policy=policy,
        checkpoint_contract=contract,
        verification={"level": "reference_parity", "source": "focused-test"},
        compatibility={"platforms": ["windows-x86_64"]},
    )
    adapter = get_export_adapter(model_name)
    exported = tmp_path / "original artifact"
    result = adapter.export_artifact(
        detector,
        format=ArtifactFormat(format_name),
        context=context,
        out=exported,
    )

    manifest = load_artifact_manifest(exported)
    verify_artifact_files(exported, manifest)
    assert manifest["layout"] == "composite"
    assert manifest["runtime"]["composition_adapter"] == manifest["compatibility"]["adapter"]
    assert manifest["compatibility"]["platforms"] == [current_platform_tag()]
    assert manifest["model"]["asset_bindings"] == {
        "embedding_kwargs.checkpoint_path": manifest["components"][0]["path"]
    }
    assert (
        manifest["runtime"]["allowed_providers"]
        == manifest["composition"]["nodes"][0]["runtime"]["allowed_providers"]
    )
    assert hashlib.sha256(result.graph_path.read_bytes()).hexdigest() == source_digest
    if format_name == "onnx":
        external = [item for item in manifest["components"] if item["role"] == "external_data"]
        assert [item["path"] for item in external] == ["model/embedding-data/bias.bin"]
    if format_name == "torchscript":
        assert manifest["components"][0]["serialization"] == ("executable-trust-required")

    checkpoint_payload = load_safe_checkpoint(result.state_path)
    fitted_state = checkpoint_payload["state"]
    assert set(fitted_state) == {
        "core_state",
        "feature_metadata",
        "score_calibration",
    }
    assert not any("threshold" in key or "labels" in key for key in _state_keys(fitted_state))

    detector._base_feature_extractor.input_color = "bgr"
    rejected = tmp_path / "unrepresentable artifact"
    with pytest.raises(Exception, match="canonical RGB|input_color"):
        adapter.export_artifact(
            detector,
            format=ArtifactFormat(format_name),
            context=context,
            out=rejected,
        )
    assert not rejected.exists()

    relocated = tmp_path / "relocated 空间" / "复合 artifact"
    relocated.parent.mkdir()
    shutil.copytree(exported, relocated)
    shutil.rmtree(exported)
    source.unlink()
    if format_name == "onnx":
        shutil.rmtree(tmp_path / "embedding-data")
    fitted_checkpoint.unlink()
    assert not source.exists() and not exported.exists()

    unrelated = tmp_path / "unrelated cwd"
    unrelated.mkdir()
    inputs_path = tmp_path / "probe-inputs.npy"
    np.save(inputs_path, np.stack(probe_inputs, axis=0), allow_pickle=False)

    if format_name == "torchscript":
        import pyimgano.utils.torchscript_safe as torchscript_safe

        called = False
        actual_load = torchscript_safe.load_module

        def forbidden_load(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            nonlocal called
            called = True
            raise AssertionError("torch.jit.load reached before trust validation")

        monkeypatch.setattr(torchscript_safe, "load_module", forbidden_load)
        with pytest.raises(ArtifactRuntimeError, match="requires executable deserialization"):
            load_artifact(relocated, trust_checkpoint=False)
        assert called is False
        monkeypatch.setattr(torchscript_safe, "load_module", actual_load)
    else:
        with pytest.raises(ArtifactRuntimeError, match="not allowed|release-verified"):
            load_artifact(relocated, providers=["CUDAExecutionProvider"])
        with pytest.raises(ArtifactRuntimeError, match="session_options.*exact"):
            load_artifact(relocated, session_options={"intra_op_num_threads": 2})

    fresh = _fresh_process_scores(
        relocated,
        inputs_path,
        trusted=format_name == "torchscript",
        cwd=unrelated,
    )
    assert fresh["backend_type"] == CompositeArtifactRuntime.__name__
    assert fresh["component_backend"] == component_backend
    assert fresh["threshold"] == pytest.approx(threshold)
    np.testing.assert_allclose(
        np.asarray(fresh["scores"], dtype=np.float64),
        reference,
        atol=1e-6,
        rtol=1e-5,
    )
