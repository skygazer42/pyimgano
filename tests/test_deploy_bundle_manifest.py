import json
from pathlib import Path

import numpy as np
import pytest

from pyimgano.models.registry import MODEL_REGISTRY
from pyimgano.services.train_service import TrainRunRequest, run_train_request


def test_run_train_request_writes_deploy_bundle_manifest(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)
            self.threshold_ = None

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            _ = epochs, lr
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ckpt", encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_deploy_bundle_manifest_dummy_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    run_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "seed": 123,
                "dataset": {
                    "name": "custom",
                    "root": str(root),
                    "category": "custom",
                    "resize": [16, 16],
                    "input_mode": "paths",
                    "limit_train": 2,
                    "limit_test": 2,
                },
                "model": {
                    "name": "test_deploy_bundle_manifest_dummy_detector",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                },
                "training": {
                    "enabled": True,
                    "epochs": 2,
                    "lr": 0.001,
                    "checkpoint_name": "model.pt",
                },
                "output": {"output_dir": str(run_dir), "save_run": True, "per_image_jsonl": False},
            }
        ),
        encoding="utf-8",
    )

    payload = run_train_request(
        TrainRunRequest(
            config_path=str(cfg_path),
            export_infer_config=True,
            export_deploy_bundle=True,
        )
    )

    bundle_dir = Path(payload["deploy_bundle_dir"])
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["entries"]
    rel_paths = {item["path"] for item in entries}

    assert manifest["schema_version"] == 1
    assert manifest["source_run"]["run_dir"] == str(run_dir)
    assert manifest["source_run"]["environment_fingerprint_sha256"]
    assert manifest["source_run"]["artifact_refs"]["infer_config"] == "artifacts/infer_config.json"
    assert (
        manifest["source_run"]["artifact_refs"]["calibration_card"]
        == "artifacts/calibration_card.json"
    )
    assert (
        manifest["source_run"]["artifact_refs"]["operator_contract"]
        == "artifacts/operator_contract.json"
    )
    assert manifest["bundle_artifact_refs"]["infer_config"] == "infer_config.json"
    assert manifest["bundle_artifact_refs"]["calibration_card"] == "calibration_card.json"
    assert manifest["bundle_artifact_refs"]["operator_contract"] == "operator_contract.json"
    assert manifest["bundle_artifact_refs"]["handoff_report"] == "handoff_report.json"
    assert manifest["required_source_artifacts_present"] is True
    assert manifest["required_bundle_artifacts_present"] is True
    digests = manifest["operator_contract_digests"]
    assert len(str(digests["source_run_operator_contract_sha256"])) == 64
    assert len(str(digests["bundle_operator_contract_sha256"])) == 64
    assert len(str(digests["bundle_infer_operator_contract_sha256"])) == 64
    assert digests["bundle_operator_contract_consistent"] is True
    assert digests["source_run_matches_bundle_operator_contract"] is True
    assert manifest["artifact_roles"]["infer_config"] == ["infer_config.json"]
    assert manifest["artifact_roles"]["calibration_card"] == ["calibration_card.json"]
    assert manifest["artifact_roles"]["operator_contract"] == ["operator_contract.json"]
    assert manifest["artifact_roles"]["handoff_report"] == ["handoff_report.json"]
    assert any(path.endswith("model.pt") for path in manifest["artifact_roles"]["checkpoint"])
    assert (bundle_dir / "handoff_report.json").exists()
    handoff_report = json.loads((bundle_dir / "handoff_report.json").read_text(encoding="utf-8"))
    assert handoff_report["schema_version"] == 1
    assert handoff_report["bundle_type"] == "cpu-offline-qc"
    assert handoff_report["files"]["bundle_manifest"] == "bundle_manifest.json"
    assert handoff_report["files"]["infer_config"] == "infer_config.json"
    assert handoff_report["files"]["handoff_report"] == "handoff_report.json"
    assert "infer_config.json" in rel_paths
    assert "calibration_card.json" in rel_paths
    assert "operator_contract.json" in rel_paths
    assert "handoff_report.json" in rel_paths
    assert "report.json" in rel_paths
    assert "config.json" in rel_paths
    assert "environment.json" in rel_paths
    assert any(path.endswith("model.pt") for path in rel_paths)
    assert all(int(item["size_bytes"]) >= 0 for item in entries)
    assert all(len(str(item["sha256"])) == 64 for item in entries)


def test_run_train_request_keeps_supporting_bundle_files_after_helper_extraction(tmp_path):
    import cv2

    class _DummyDetector:
        def __init__(self, **kwargs):  # noqa: ANN003 - test stub
            self.kwargs = dict(kwargs)
            self.threshold_ = None

        def fit(self, X, *, epochs=None, lr=None):  # noqa: ANN001 - test stub
            _ = epochs, lr
            self.fit_inputs = list(X)
            return self

        def decision_function(self, X):  # noqa: ANN001
            return np.linspace(0.0, 1.0, num=len(list(X)), dtype=np.float32)

        def save_checkpoint(self, path):  # noqa: ANN001 - test stub
            Path(path).write_text("ckpt", encoding="utf-8")

    MODEL_REGISTRY.register(
        "test_deploy_bundle_supporting_files_dummy_detector",
        _DummyDetector,
        tags=("vision",),
        overwrite=True,
    )

    root = tmp_path / "custom"
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 121),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((16, 16, 3), dtype=np.uint8) * int(value)
        cv2.imwrite(str(p), img)

    run_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "seed": 123,
                "dataset": {
                    "name": "custom",
                    "root": str(root),
                    "category": "custom",
                    "resize": [16, 16],
                    "input_mode": "paths",
                    "limit_train": 2,
                    "limit_test": 2,
                },
                "model": {
                    "name": "test_deploy_bundle_supporting_files_dummy_detector",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                },
                "training": {
                    "enabled": True,
                    "epochs": 2,
                    "lr": 0.001,
                    "checkpoint_name": "model.pt",
                },
                "output": {"output_dir": str(run_dir), "save_run": True, "per_image_jsonl": False},
            }
        ),
        encoding="utf-8",
    )

    payload = run_train_request(
        TrainRunRequest(
            config_path=str(cfg_path),
            export_infer_config=True,
            export_deploy_bundle=True,
        )
    )
    bundle_dir = Path(payload["deploy_bundle_dir"])

    assert (bundle_dir / "report.json").is_file()
    assert (bundle_dir / "config.json").is_file()
    assert (bundle_dir / "environment.json").is_file()
    assert (bundle_dir / "calibration_card.json").is_file()
    assert (bundle_dir / "operator_contract.json").is_file()


def test_run_train_request_rejects_invalid_deploy_bundle_request_before_recipe_runs(tmp_path):
    from pyimgano.recipes.registry import RECIPE_REGISTRY

    calls: list[str] = []

    def _dummy_recipe(cfg):  # noqa: ANN001 - test stub
        calls.append(str(cfg.output.output_dir))
        run_dir = Path(str(cfg.output.output_dir))
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_dir": str(run_dir),
            "threshold": 0.5,
            "threshold_provenance": {"method": "fixed"},
        }

    RECIPE_REGISTRY.register(
        "test_deploy_bundle_requires_pixel_threshold_precheck_recipe",
        _dummy_recipe,
        overwrite=True,
    )

    root = tmp_path / "custom"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_path / "run_out"
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "test_deploy_bundle_requires_pixel_threshold_precheck_recipe",
                "dataset": {
                    "name": "custom",
                    "root": str(root),
                    "category": "custom",
                    "resize": [16, 16],
                },
                "model": {
                    "name": "vision_ecod",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                },
                "defects": {
                    "enabled": True,
                    "pixel_threshold": None,
                    "pixel_threshold_strategy": "normal_pixel_quantile",
                },
                "output": {"output_dir": str(run_dir), "save_run": True, "per_image_jsonl": False},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="pixel_threshold"):
        run_train_request(
            TrainRunRequest(
                config_path=str(cfg_path),
                export_deploy_bundle=True,
            )
        )

    assert calls == []


def test_validate_deploy_bundle_manifest_detects_tampered_file(tmp_path):
    from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    infer_cfg = bundle_dir / "infer_config.json"
    infer_cfg.write_text("{}", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_run": {"run_dir": "/tmp/run", "environment_fingerprint_sha256": "f" * 64},
        "entries": [
            {
                "path": "infer_config.json",
                "role": "infer_config",
                "size_bytes": int(infer_cfg.stat().st_size),
                "sha256": "0" * 64,
            }
        ],
    }

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=True)

    assert any("SHA256 mismatch" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_missing_bundle_artifact_ref(tmp_path):
    from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    infer_cfg = bundle_dir / "infer_config.json"
    infer_cfg.write_text("{}", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_run": {"run_dir": "/tmp/run", "environment_fingerprint_sha256": "f" * 64},
        "bundle_artifact_refs": {
            "infer_config": "infer_config.json",
            "calibration_card": "missing_calibration_card.json",
        },
        "entries": [
            {
                "path": "infer_config.json",
                "role": "infer_config",
                "size_bytes": int(infer_cfg.stat().st_size),
                "sha256": "0" * 64,
            }
        ],
    }

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("bundle_artifact_refs.calibration_card" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_incorrect_roles_and_completeness_flags(tmp_path):
    from pyimgano.reporting.deploy_bundle import validate_deploy_bundle_manifest

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    infer_cfg = bundle_dir / "infer_config.json"
    infer_cfg.write_text("{}", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "source_run": {
            "run_dir": "/tmp/run",
            "environment_fingerprint_sha256": "f" * 64,
            "artifact_refs": {"infer_config": "artifacts/infer_config.json"},
        },
        "bundle_artifact_refs": {
            "infer_config": "infer_config.json",
        },
        "artifact_roles": {
            "checkpoint": ["infer_config.json"],
        },
        "required_source_artifacts_present": True,
        "required_bundle_artifacts_present": True,
        "entries": [
            {
                "path": "infer_config.json",
                "role": "infer_config",
                "size_bytes": int(infer_cfg.stat().st_size),
                "sha256": "0" * 64,
            }
        ],
    }

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("artifact_roles" in item for item in errors)
    assert any("required_source_artifacts_present" in item for item in errors)
    assert any("required_bundle_artifacts_present" in item for item in errors)


def test_build_deploy_bundle_manifest_classifies_weight_audit_files(tmp_path):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model_card.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "weights_manifest.json").write_text("{}", encoding="utf-8")

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    roles = {item["path"]: item["role"] for item in manifest["entries"]}

    assert roles["model_card.json"] == "model_card"
    assert roles["weights_manifest.json"] == "weights_manifest"


def test_validate_deploy_bundle_manifest_accepts_valid_weight_audit_files(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    ckpt = bundle_dir / "model.pt"
    data = b"bundle-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    (bundle_dir / "weights_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entries": [
                    {
                        "name": "bundle_model",
                        "path": "model.pt",
                        "sha256": sha,
                        "source": "test",
                        "license": "internal",
                        "runtime": "torch",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "model_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "bundle_model",
                "summary": {
                    "purpose": "Bundle validation",
                    "intended_inputs": "RGB",
                    "output_contract": "image-level",
                },
                "weights": {
                    "path": "model.pt",
                    "manifest_entry": "bundle_model",
                    "sha256": sha,
                    "source": "test",
                    "license": "internal",
                },
                "deployment": {"runtime": "torch"},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=True)

    assert errors == []


def test_validate_deploy_bundle_manifest_rejects_invalid_model_card_payload(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "broken_model",
                "weights": {
                    "path": "missing.pt",
                    "source": "test",
                    "license": "internal",
                },
                "deployment": {"runtime": "torch"},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("model_card.json" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_operator_contract_mismatch(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "operator_contract": {
                    "schema_version": 1,
                    "review_policy": {
                        "review_on": ["anomalous", "rejected_low_confidence"],
                        "confidence_gate_enabled": True,
                        "reject_confidence_below": 0.75,
                        "reject_label": -9,
                    },
                },
                "artifact_quality": {
                    "has_operator_contract": True,
                    "audit_refs": {"operator_contract": "operator_contract.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "operator_contract.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "review_policy": {
                    "review_on": ["anomalous"],
                    "confidence_gate_enabled": True,
                    "reject_confidence_below": 0.6,
                    "reject_label": -9,
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("operator_contract mismatch" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_missing_operator_contract_ref(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "operator_contract": {"schema_version": 1},
                "artifact_quality": {
                    "has_operator_contract": True,
                    "audit_refs": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "operator_contract.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("artifact_quality.audit_refs.operator_contract" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_missing_operator_contract_file(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "operator_contract": {"schema_version": 1},
                "artifact_quality": {
                    "has_operator_contract": True,
                    "audit_refs": {"operator_contract": "operator_contract.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("requires operator_contract.json" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_tampered_operator_contract_digests(tmp_path):
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    (run_dir / "artifacts" / "operator_contract.json").write_text(
        json.dumps(operator_contract),
        encoding="utf-8",
    )

    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "operator_contract": operator_contract,
                "artifact_quality": {
                    "has_operator_contract": True,
                    "audit_refs": {"operator_contract": "operator_contract.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "operator_contract.json").write_text(
        json.dumps(operator_contract),
        encoding="utf-8",
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    manifest["operator_contract_digests"]["bundle_operator_contract_sha256"] = "0" * 64

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any(
        "operator_contract_digests.bundle_operator_contract_sha256" in item for item in errors
    )
