from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _policy(threshold: float) -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact-policy",
        "schema_version": 1,
        "model": {
            "registry_name": "ae_resnet_unet",
            "category": "bottle",
            "constructor_kwargs": {"device": "cpu"},
        },
        "threshold": threshold,
        "postprocess": {
            "image_threshold": {
                "threshold": threshold,
                "score_order": "higher_is_more_anomalous",
            },
            "map_postprocess": {"method": "none"},
        },
    }


def _payload() -> dict[str, object]:
    return {
        "schema_family": "pyimgano-artifact",
        "schema_version": 1,
        "layout": "native_detector",
        "model": {
            "registry_name": "ae_resnet_unet",
            "category": "bottle",
            "constructor_kwargs": {"device": "cpu"},
        },
        "runtime": {
            "backend": "pyimgano",
            "allowed_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "verified_providers": [{"name": "CPUExecutionProvider", "options": {}}],
            "entrypoint": "state/detector.pyim",
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
                "id": "state",
                "path": "state/detector.pyim",
                "role": "trained_state",
                "format": "pyimgano-state",
                "serialization": "safe-data",
                "size_bytes": 5,
                "sha256": _sha(b"state"),
            }
        ],
        "policy_ref": {"path": "infer_config.json"},
        "compatibility": {
            "pyimgano": ">=0.10,<0.11",
            "python": ">=3.9,<3.13",
            "platforms": ["linux-x86_64"],
            "runtime_versions": {},
            "adapter": {"id": "reference", "version": 1},
            "codecs": [{"id": "tensor-state", "version": 1}],
        },
        "verification": {
            "level": "reference_parity",
            "reference_backend": "pyimgano",
            "report": {
                "path": "verification/parity.json",
                "size_bytes": 2,
                "sha256": _sha(b"{}"),
            },
        },
    }


def _write_source(root: Path) -> dict[str, object]:
    from pyimgano.artifacts.manifest import write_artifact_manifest

    (root / "state").mkdir(parents=True)
    (root / "verification").mkdir()
    (root / "state" / "detector.pyim").write_bytes(b"state")
    (root / "verification" / "parity.json").write_bytes(b"{}")
    write_artifact_manifest(root, _payload(), policy=_policy(0.5))
    return json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))


def test_policy_rejects_external_reconstruction_authority() -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, validate_artifact_policy

    for forbidden in (
        {"checkpoint_path": "../model.pkl"},
        {"source_run": "/tmp/run"},
        {"nested": {"import_path": "evil.Payload"}},
        {"model": {"registry_name": "x", "module": "evil"}},
        {"runtime": {"backend": "onnxruntime"}},
    ):
        payload = _policy(0.5)
        payload.update(forbidden)
        with pytest.raises(ArtifactPolicyError):
            validate_artifact_policy(payload)


def test_policy_rejects_conflicting_threshold_and_model_mirrors() -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, validate_artifact_policy

    payload = _policy(0.5)
    payload["threshold"] = 0.7
    with pytest.raises(ArtifactPolicyError, match="threshold"):
        validate_artifact_policy(payload)

    payload = _policy(0.5)
    with pytest.raises(ArtifactPolicyError, match="model"):
        validate_artifact_policy(payload, manifest_model={"registry_name": "other"})


def test_bind_policy_atomically_clones_and_rebinds_only_policy_identity(tmp_path: Path) -> None:
    from pyimgano.artifacts.manifest import load_artifact_manifest
    from pyimgano.artifacts.policy import bind_policy

    source = tmp_path / "source"
    old_manifest = _write_source(source)
    out = tmp_path / "bound"
    probes: list[Path] = []

    result = bind_policy(source, _policy(0.7), out, probe=lambda path: probes.append(path))

    new_manifest = load_artifact_manifest(result)
    assert new_manifest["runtime_id"] == old_manifest["runtime_id"]
    assert new_manifest["policy_id"] != old_manifest["policy_id"]
    assert new_manifest["artifact_id"] != old_manifest["artifact_id"]
    assert (source / "infer_config.json").read_bytes() != (out / "infer_config.json").read_bytes()
    assert (out / "state" / "detector.pyim").read_bytes() == b"state"
    assert len(probes) == 1
    assert probes[0] != out
    assert not probes[0].exists()


def test_bind_policy_accepts_a_json_policy_path(tmp_path: Path) -> None:
    from pyimgano.artifacts.manifest import load_artifact_manifest
    from pyimgano.artifacts.policy import bind_policy

    source = tmp_path / "source"
    old_manifest = _write_source(source)
    policy_path = tmp_path / "production-policy.json"
    policy_path.write_text(json.dumps(_policy(0.8)), encoding="utf-8")

    out = bind_policy(source, policy_path, out=tmp_path / "bound", probe=lambda _path: None)

    new_manifest = load_artifact_manifest(out)
    assert new_manifest["runtime_id"] == old_manifest["runtime_id"]
    assert new_manifest["policy_id"] != old_manifest["policy_id"]


def test_bind_policy_rejects_duplicate_keys_in_policy_json(tmp_path: Path) -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, bind_policy

    source = tmp_path / "source"
    _write_source(source)
    policy_path = tmp_path / "bad-policy.json"
    policy_path.write_text(
        '{"schema_family":"pyimgano-artifact-policy",'
        '"schema_family":"other","schema_version":1,"postprocess":{}}',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactPolicyError, match="duplicate"):
        bind_policy(source, policy_path, out=tmp_path / "bound", probe=lambda _path: None)


def test_bind_policy_probe_failure_leaves_no_partial_output(tmp_path: Path) -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, bind_policy

    source = tmp_path / "source"
    _write_source(source)
    out = tmp_path / "bound"

    def _fail(_path: Path) -> None:
        raise RuntimeError("parity failed")

    with pytest.raises(ArtifactPolicyError, match="probe"):
        bind_policy(source, _policy(0.7), out, probe=_fail)
    assert not out.exists()


def test_bind_policy_refuses_existing_output_without_mutating_it(tmp_path: Path) -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, bind_policy

    source = tmp_path / "source"
    _write_source(source)
    out = tmp_path / "bound"
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ArtifactPolicyError, match="exists"):
        bind_policy(source, _policy(0.7), out, probe=lambda _path: None)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_bind_policy_rejects_policy_model_conflict_before_cloning(tmp_path: Path) -> None:
    from pyimgano.artifacts.policy import ArtifactPolicyError, bind_policy

    source = tmp_path / "source"
    _write_source(source)
    out = tmp_path / "bound"
    bad = copy.deepcopy(_policy(0.7))
    bad["model"]["registry_name"] = "other"

    with pytest.raises(ArtifactPolicyError, match="model"):
        bind_policy(source, bad, out, probe=lambda _path: None)
    assert not out.exists()


def test_default_policy_probe_forwards_explicit_executable_trust(monkeypatch) -> None:
    from types import SimpleNamespace

    from pyimgano.artifacts import policy as policy_module

    calls: list[tuple[Path, bool]] = []
    runtime_module = SimpleNamespace(
        probe_artifact_policy=lambda path, *, trust_checkpoint=False: calls.append(
            (Path(path), bool(trust_checkpoint))
        )
    )
    monkeypatch.setattr(
        policy_module.importlib,
        "import_module",
        lambda _name: runtime_module,
    )

    artifact = Path("artifact")
    policy_module._default_probe(artifact, trust_checkpoint=True)

    assert calls == [(artifact, True)]
