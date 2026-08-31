from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy(threshold: float = 0.5) -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "postprocess": {
            "image_threshold": {
                "threshold": threshold,
                "score_order": "higher_is_more_anomalous",
            }
        },
    }


def _single_graph_payload(model_bytes: bytes = b"onnx-model") -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "layout": "single_graph",
        "runtime": {
            "backend": "onnxruntime",
            "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "entrypoint": "model/detector.onnx",
        },
        "input_contract": {
            "kind": "image_batch",
            "name": "input",
            "dtype": "float32",
            "layout": "NCHW",
            "color_space": "RGB",
            "size": [224, 224],
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
                "size_bytes": len(model_bytes),
                "sha256": _sha(model_bytes),
            }
        ],
        "policy_ref": {"path": "infer_config.json"},
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": ["linux-x86_64"],
            "onnx_opset": 18,
            "onnx_ir": 9,
            "runtime_versions": {"onnxruntime": ">=1.17,<2"},
            "adapter": {"id": "vision-autoencoder", "version": 1},
            "codecs": [],
        },
        "verification": {
            "level": "runtime_smoke",
            "report": {
                "path": "verification/parity.json",
                "size_bytes": 2,
                "sha256": _sha(b"{}"),
            },
        },
    }


def test_canonical_json_normalizes_unicode_keys_and_numbers() -> None:
    from pyimgano.artifacts.manifest import canonical_json_bytes

    decomposed = "e\N{COMBINING ACUTE ACCENT}"
    composed = "\N{LATIN SMALL LETTER E WITH ACUTE}"
    left = {"z": -0.0, decomposed: [1.0, 0.1]}
    right = {composed: [1, 0.1], "z": 0}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_build_manifest_computes_three_independent_identity_layers() -> None:
    from pyimgano.artifacts.manifest import build_artifact_manifest

    policy = _policy()
    manifest = build_artifact_manifest(_single_graph_payload(), policy)

    assert manifest["runtime_id"].startswith("sha256:")
    assert manifest["policy_id"].startswith("sha256:")
    assert manifest["artifact_id"].startswith("sha256:")
    assert manifest["policy_ref"]["policy_id"] == manifest["policy_id"]

    runtime_changed_payload = copy.deepcopy(manifest)
    runtime_changed_payload.pop("runtime_id")
    runtime_changed_payload.pop("artifact_id")
    runtime_changed_payload["input_contract"]["size"] = [256, 256]
    runtime_changed = build_artifact_manifest(runtime_changed_payload, policy)
    assert runtime_changed["runtime_id"] != manifest["runtime_id"]
    assert runtime_changed["policy_id"] == manifest["policy_id"]
    assert runtime_changed["artifact_id"] != manifest["artifact_id"]

    policy_changed = build_artifact_manifest(_single_graph_payload(), _policy(0.7))
    assert policy_changed["runtime_id"] == manifest["runtime_id"]
    assert policy_changed["policy_id"] != manifest["policy_id"]
    assert policy_changed["artifact_id"] != manifest["artifact_id"]

    provenance_changed_payload = copy.deepcopy(manifest)
    provenance_changed_payload.pop("runtime_id")
    provenance_changed_payload.pop("policy_id")
    provenance_changed_payload.pop("artifact_id")
    provenance_changed_payload["provenance"] = {"producer": "different-host"}
    provenance_changed = build_artifact_manifest(provenance_changed_payload, policy)
    assert provenance_changed["runtime_id"] == manifest["runtime_id"]
    assert provenance_changed["artifact_id"] == manifest["artifact_id"]


def test_validate_manifest_rejects_identity_mutation() -> None:
    from pyimgano.artifacts.manifest import (
        ArtifactManifestError,
        build_artifact_manifest,
        validate_artifact_manifest,
    )

    policy = _policy()
    manifest = build_artifact_manifest(_single_graph_payload(), policy)
    manifest["input_contract"]["size"] = [256, 256]

    with pytest.raises(ArtifactManifestError, match="runtime_id"):
        validate_artifact_manifest(manifest, policy)


def test_provider_specs_must_be_safe_and_verified_subset_of_allowed() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _single_graph_payload()
    payload["runtime"]["verified_providers"] = [{"name": "CUDAExecutionProvider", "options": {}}]
    with pytest.raises(ArtifactManifestError, match="verified_providers"):
        build_artifact_manifest(payload, _policy())

    payload = _single_graph_payload()
    payload["runtime"]["allowed_providers"][0]["options"] = {
        "user_compute_stream": {"unsafe": "object"}
    }
    payload["runtime"]["verified_providers"] = copy.deepcopy(
        payload["runtime"]["allowed_providers"]
    )
    with pytest.raises(ArtifactManifestError, match="scalar"):
        build_artifact_manifest(payload, _policy())


def test_session_options_reject_unknown_or_code_loading_keys() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _single_graph_payload()
    payload["runtime"]["session_options"] = {"custom_op_library": "/tmp/evil.so"}
    with pytest.raises(ArtifactManifestError, match="session_options"):
        build_artifact_manifest(payload, _policy())


def test_all_component_policy_and_attachment_paths_are_unique() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _single_graph_payload()
    payload["verification"]["report"]["path"] = "model/detector.onnx"
    with pytest.raises(ArtifactManifestError, match="duplicate.*path"):
        build_artifact_manifest(payload, _policy())


def test_provenance_attachments_are_hash_addressed() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _single_graph_payload()
    payload["provenance"] = {
        "producer": "ci",
        "attachments": [{"path": "verification/provenance.json", "size_bytes": 2}],
    }
    with pytest.raises(ArtifactManifestError, match="provenance.attachments.*sha256"):
        build_artifact_manifest(payload, _policy())


def test_verification_level_and_attachment_are_strict() -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, build_artifact_manifest

    payload = _single_graph_payload()
    payload["verification"]["level"] = "structural"
    with pytest.raises(ArtifactManifestError, match="verification.level"):
        build_artifact_manifest(payload, _policy())

    payload = _single_graph_payload()
    del payload["verification"]["report"]["sha256"]
    with pytest.raises(ArtifactManifestError, match="verification.report.sha256"):
        build_artifact_manifest(payload, _policy())


def test_write_and_load_manifest_verify_policy_bytes_and_ids(tmp_path: Path) -> None:
    from pyimgano.artifacts.manifest import (
        ArtifactManifestError,
        load_artifact_manifest,
        write_artifact_manifest,
    )

    root = tmp_path / "artifact"
    (root / "model").mkdir(parents=True)
    (root / "verification").mkdir()
    (root / "model" / "detector.onnx").write_bytes(b"onnx-model")
    (root / "verification" / "parity.json").write_bytes(b"{}")

    path = write_artifact_manifest(root, _single_graph_payload(), policy=_policy())
    loaded = load_artifact_manifest(path)
    assert loaded["artifact_id"].startswith("sha256:")

    policy_path = root / "infer_config.json"
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy_payload["postprocess"]["image_threshold"]["threshold"] = 0.9
    policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match=r"policy_ref\.(size_bytes|sha256)"):
        load_artifact_manifest(root)


def test_load_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    from pyimgano.artifacts.manifest import ArtifactManifestError, load_artifact_manifest

    path = tmp_path / "artifact_manifest.json"
    path.write_text(
        '{"schema_family":"pyimgano-artifact","schema_family":"other"}',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactManifestError, match="duplicate"):
        load_artifact_manifest(path)
