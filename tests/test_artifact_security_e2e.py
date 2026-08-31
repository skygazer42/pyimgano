from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_VERIFICATION_REPORT = b"{}"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy() -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {
            "image_threshold": {
                "threshold": 0.5,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }


def _verified_model_bytes() -> bytes:
    onnx = pytest.importorskip("onnx")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("ReduceMean", ["input"], ["score"], axes=[1, 2, 3], keepdims=0)],
        "verified-model",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [None, 3, 4, 4])],
        [onnx.helper.make_tensor_value_info("score", onnx.TensorProto.FLOAT, [None])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
    model.ir_version = 9
    return model.SerializeToString()


def _write_synthetic_onnx_artifact(root: Path) -> Path:
    from pyimgano.artifacts import write_artifact_manifest

    model_path = root / "model" / "detector.onnx"
    report_path = root / "verification" / "runtime.json"
    model_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)
    verified_model = _verified_model_bytes()
    model_path.write_bytes(verified_model)
    report_path.write_bytes(_VERIFICATION_REPORT)
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
            }
        },
        "components": [
            {
                "path": "model/detector.onnx",
                "role": "runtime_model",
                "format": "onnx",
                "serialization": "onnx",
                "size_bytes": len(verified_model),
                "sha256": _sha(verified_model),
            }
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
                "size_bytes": len(_VERIFICATION_REPORT),
                "sha256": _sha(_VERIFICATION_REPORT),
            },
        },
    }
    write_artifact_manifest(root, payload, _policy())
    return root


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return _write_synthetic_onnx_artifact(tmp_path / "artifact")


@pytest.mark.parametrize(
    ("embedded_path", "message"),
    [
        ("../outside.onnx", "dot-dot"),
        ("/tmp/outside.onnx", "absolute"),
    ],
)
def test_public_loader_rejects_traversal_and_absolute_component_paths(
    artifact_root: Path,
    embedded_path: str,
    message: str,
) -> None:
    from pyimgano.artifacts import ArtifactManifestError
    from pyimgano.inference import load_artifact

    manifest_path = artifact_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["components"][0]["path"] = embedded_path
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match=message):
        load_artifact(artifact_root)


def test_public_loader_rejects_component_symlink_escape(
    artifact_root: Path,
    tmp_path: Path,
) -> None:
    from pyimgano.artifacts import ArtifactSecurityError
    from pyimgano.inference import load_artifact

    outside = tmp_path / "outside.onnx"
    outside.write_bytes(_verified_model_bytes())
    component = artifact_root / "model" / "detector.onnx"
    component.unlink()
    try:
        component.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ArtifactSecurityError, match="symlink"):
        load_artifact(artifact_root)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("same-size", "SHA-256"),
        ("different-size", "size mismatch"),
    ],
)
def test_public_loader_rejects_hash_and_size_tamper(
    artifact_root: Path,
    tamper: str,
    message: str,
) -> None:
    from pyimgano.artifacts import ArtifactSecurityError
    from pyimgano.inference import load_artifact

    component = artifact_root / "model" / "detector.onnx"
    verified_model = _verified_model_bytes()
    if tamper == "same-size":
        component.write_bytes(b"X" + verified_model[1:])
    else:
        component.write_bytes(verified_model + b"X")

    with pytest.raises(ArtifactSecurityError, match=message):
        load_artifact(artifact_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_family", "vendor-artifact", "schema_family"),
        ("schema_version", 999, "unsupported schema version"),
    ],
)
def test_public_loader_rejects_unknown_artifact_schema(
    artifact_root: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    from pyimgano.artifacts import ArtifactManifestError
    from pyimgano.inference import load_artifact

    manifest_path = artifact_root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match=message):
        load_artifact(artifact_root)


def test_public_loader_rejects_oversized_manifest_before_json_parsing(
    artifact_root: Path,
) -> None:
    from pyimgano.artifacts.manifest import MAX_MANIFEST_BYTES, ArtifactManifestError
    from pyimgano.inference import load_artifact

    manifest_path = artifact_root / "artifact_manifest.json"
    manifest_path.write_bytes(b"{" + b" " * MAX_MANIFEST_BYTES)

    with pytest.raises(ArtifactManifestError, match="exceeds"):
        load_artifact(artifact_root)


def test_public_loader_rejects_duplicate_manifest_json_entries(
    artifact_root: Path,
) -> None:
    from pyimgano.artifacts import ArtifactManifestError
    from pyimgano.inference import load_artifact

    manifest_path = artifact_root / "artifact_manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    duplicate = '"schema_family":"pyimgano-artifact",'
    raw = raw.replace('"schema_family":', duplicate + '"schema_family":', 1)
    manifest_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ArtifactManifestError, match="duplicate JSON object key"):
        load_artifact(artifact_root)


def _write_untrusted_native_artifact(root: Path) -> Path:
    from pyimgano.artifacts import write_artifact_manifest

    state = b"legacy executable checkpoint"
    state_path = root / "state" / "detector.ckpt"
    report_path = root / "verification" / "parity.json"
    state_path.parent.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)
    state_path.write_bytes(state)
    report_path.write_bytes(_VERIFICATION_REPORT)
    payload = {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "layout": "native_detector",
        "model": {
            "registry_name": "test_untrusted_model",
            "constructor_kwargs": {},
        },
        "runtime": {
            "backend": "pyimgano",
            "entrypoint": "state/detector.ckpt",
            "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
        },
        "input_contract": {"kind": "image_batch", "dtype": "uint8", "layout": "HWC"},
        "output_contract": {
            "score": {
                "name": "score",
                "transform": "identity",
                "score_order": "higher_is_more_anomalous",
            }
        },
        "components": [
            {
                "path": "state/detector.ckpt",
                "role": "trained_state",
                "format": "legacy-checkpoint",
                "serialization": "executable-trust-required",
                "size_bytes": len(state),
                "sha256": _sha(state),
            }
        ],
        "policy_ref": {"path": "infer_config.json"},
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": ["linux-x86_64"],
            "runtime_versions": {},
            "adapter": {"id": "legacy-test", "version": 1},
            "codecs": [{"id": "legacy-checkpoint", "version": 1}],
        },
        "verification": {
            "level": "reference_parity",
            "reference_backend": "pyimgano",
            "report": {
                "path": "verification/parity.json",
                "size_bytes": len(_VERIFICATION_REPORT),
                "sha256": _sha(_VERIFICATION_REPORT),
            },
        },
    }
    write_artifact_manifest(root, payload, _policy())
    return root


def test_executable_serialization_is_not_deserialized_without_explicit_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyimgano.models.registry as registry
    import pyimgano.workbench.checkpoint_restore as checkpoint_restore
    from pyimgano.inference import load_artifact
    from pyimgano.inference.artifact_runtime import ArtifactRuntimeError

    root = _write_untrusted_native_artifact(tmp_path / "untrusted")
    restore_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(registry, "create_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        checkpoint_restore,
        "load_checkpoint_into_detector",
        lambda *args, **kwargs: restore_calls.append(args),
    )

    with pytest.raises(ArtifactRuntimeError, match="requires executable deserialization"):
        load_artifact(root)
    assert restore_calls == []


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="deterministic verified-bytes replacement acceptance is Linux-specific",
)
def test_backend_session_creation_consumes_staged_verified_bytes_after_source_replacement(
    artifact_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyimgano.inference.onnx_runtime as onnx_runtime
    from pyimgano.inference import load_artifact

    source = artifact_root / "model" / "detector.onnx"
    captured: dict[str, Any] = {}

    class _CapturingBackend:
        runtime_info = {"backend": "onnxruntime", "selected_provider": "test"}

        def __init__(self, model_path: str | Path, **kwargs: Any) -> None:
            del kwargs
            replacement = source.with_name("attacker-replacement.onnx")
            replacement.write_bytes(b"attacker-controlled replacement")
            os.replace(replacement, source)
            staged_path = Path(model_path)
            captured["model_path"] = staged_path
            captured["model_bytes"] = staged_path.read_bytes()

    monkeypatch.setattr(onnx_runtime, "OnnxArtifactRuntime", _CapturingBackend)
    runtime = load_artifact(artifact_root)
    try:
        staged_path = captured["model_path"]
        assert isinstance(staged_path, Path)
        assert staged_path != source
        assert staged_path.is_file()
        assert captured["model_bytes"] == _verified_model_bytes()
        assert source.read_bytes() == b"attacker-controlled replacement"
    finally:
        runtime.close()
