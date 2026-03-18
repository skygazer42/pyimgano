from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_run_quality_detects_deployable_run(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {"threshold": 0.5, "split_fingerprint": {"sha256": "f" * 64}},
    )
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "f" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    bundle_dir = run_dir / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(bundle_dir / "report.json", {"dataset": "custom"})
    _write_json(bundle_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(bundle_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )
    _write_json(
        bundle_dir / "bundle_manifest.json",
        build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir),
    )

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "deployable"
    assert quality["score"] == pytest.approx(1.0)
    assert quality["missing_required"] == []
    assert quality["artifacts"]["infer_config"]["present"] is True
    assert quality["artifacts"]["calibration_card"]["present"] is True
    assert quality["artifacts"]["calibration_card"]["valid"] is True
    assert quality["artifacts"]["deploy_bundle_manifest"]["present"] is True
    assert quality["bundle_manifest"]["valid"] is True
    assert quality["trust_summary"]["status"] == "trust-signaled"


def test_evaluate_run_quality_reports_partial_run(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "partial"
    assert quality["score"] == pytest.approx(0.25)
    assert quality["missing_required"] == ["config.json", "environment.json"]
    assert quality["artifacts"]["report"]["present"] is True
    assert quality["artifacts"]["config"]["present"] is False
    assert quality["artifacts"]["environment"]["present"] is False
    assert quality["trust_summary"]["status"] == "partial"


def test_evaluate_run_quality_downgrades_invalid_calibration_card(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(run_dir / "artifacts" / "infer_config.json", {"threshold": 0.5})
    _write_json(run_dir / "artifacts" / "calibration_card.json", {"schema_version": 1})

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "reproducible"
    assert quality["audited_complete"] is False
    assert quality["artifacts"]["calibration_card"]["present"] is True
    assert quality["artifacts"]["calibration_card"]["valid"] is False
    assert any("image_threshold" in item for item in quality["artifacts"]["calibration_card"]["errors"])


def test_evaluate_run_quality_reports_calibration_audit_warnings(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {
            "threshold": 0.55,
            "split_fingerprint": {"sha256": "a" * 64},
            "prediction": {"reject_confidence_below": 0.75, "reject_label": -9},
        },
    )
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "reproducible"
    audit = quality["calibration_audit"]
    assert audit["present"] is True
    assert audit["matching_threshold"] is False
    assert audit["has_threshold_context"] is False
    assert audit["has_split_fingerprint"] is False
    assert audit["has_prediction_policy"] is False
    assert any("threshold mismatch" in item for item in audit["warnings"])
    assert any("split_fingerprint" in item for item in audit["warnings"])
    assert any("prediction_policy" in item for item in audit["warnings"])
    assert any("threshold mismatch" in item for item in quality["warnings"])
    trust = quality["trust_summary"]
    assert trust["status"] == "partial"
    assert trust["status_reasons"] == [
        "core_artifacts_present",
        "calibration_audit_incomplete",
        "warnings_present",
    ]
    assert "missing_threshold_context" in trust["degraded_by"]
    assert "missing_split_fingerprint" in trust["degraded_by"]
    assert "missing_prediction_policy" in trust["degraded_by"]
    assert trust["audit_refs"] == {
        "report_json": "report.json",
        "config_json": "config.json",
        "environment_json": "environment.json",
        "infer_config_json": "artifacts/infer_config.json",
        "calibration_card_json": "artifacts/calibration_card.json",
    }


def test_evaluate_run_quality_emits_trust_signals(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {"threshold": 0.5, "split_fingerprint": {"sha256": "f" * 64}},
    )
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "f" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    quality = evaluate_run_quality(run_dir)

    trust = quality["trust_summary"]
    assert trust["trust_signals"]["has_core_artifacts"] is True
    assert trust["trust_signals"]["has_infer_config"] is True
    assert trust["trust_signals"]["has_calibration_card"] is True
    assert trust["trust_signals"]["has_threshold_context"] is True
    assert trust["trust_signals"]["has_split_fingerprint"] is True
    assert trust["trust_signals"]["has_prediction_policy"] is False


def test_evaluate_run_quality_emits_operator_contract_trust_signals(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {
            "threshold": 0.5,
            "split_fingerprint": {"sha256": "f" * 64},
            "operator_contract": operator_contract,
        },
    )
    _write_json(run_dir / "artifacts" / "operator_contract.json", operator_contract)
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "f" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    quality = evaluate_run_quality(run_dir)

    trust = quality["trust_summary"]
    assert trust["status"] == "trust-signaled"
    assert trust["trust_signals"]["has_operator_contract"] is True
    assert trust["trust_signals"]["has_operator_contract_consistent"] is True
    assert trust["audit_refs"]["operator_contract_json"] == "artifacts/operator_contract.json"


def test_evaluate_run_quality_downgrades_mismatched_operator_contract(tmp_path: Path) -> None:
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir / "artifacts" / "infer_config.json",
        {
            "threshold": 0.5,
            "split_fingerprint": {"sha256": "f" * 64},
            "operator_contract": {
                "schema_version": 1,
                "review_policy": {
                    "review_on": ["anomalous", "rejected_low_confidence"],
                    "confidence_gate_enabled": True,
                    "reject_confidence_below": 0.75,
                    "reject_label": -9,
                },
            },
        },
    )
    _write_json(
        run_dir / "artifacts" / "operator_contract.json",
        {
            "schema_version": 1,
            "review_policy": {
                "review_on": ["anomalous"],
                "confidence_gate_enabled": True,
                "reject_confidence_below": 0.6,
                "reject_label": -9,
            },
        },
    )
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "f" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "reproducible"
    trust = quality["trust_summary"]
    assert trust["status"] == "partial"
    assert trust["trust_signals"]["has_operator_contract"] is True
    assert trust["trust_signals"]["has_operator_contract_consistent"] is False
    assert "operator_contract_incomplete" in trust["status_reasons"]
    assert "operator_contract_mismatch" in trust["degraded_by"]


def test_evaluate_run_quality_accepts_valid_bundle_weight_audit(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(run_dir / "artifacts" / "infer_config.json", {"threshold": 0.5})
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    bundle_dir = run_dir / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(bundle_dir / "report.json", {"dataset": "custom"})
    _write_json(bundle_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(bundle_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    ckpt = bundle_dir / "model.pt"
    data = b"bundle-weights"
    ckpt.write_bytes(data)

    import hashlib

    sha = hashlib.sha256(data).hexdigest()

    _write_json(
        bundle_dir / "weights_manifest.json",
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
        },
    )
    _write_json(
        bundle_dir / "model_card.json",
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
        },
    )
    _write_json(
        bundle_dir / "bundle_manifest.json",
        build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir),
    )

    quality = evaluate_run_quality(run_dir, check_bundle_hashes=True)

    assert quality["status"] == "deployable"
    assert quality["deployable_complete"] is True
    assert quality["weights_audit"]["present"] is True
    assert quality["weights_audit"]["valid"] is True
    assert quality["weights_audit"]["model_card"]["valid"] is True
    assert quality["weights_audit"]["weights_manifest"]["valid"] is True


def test_evaluate_run_quality_downgrades_invalid_bundle_model_card(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_quality import evaluate_run_quality

    run_dir = tmp_path / "run"
    _write_json(run_dir / "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(run_dir / "artifacts" / "infer_config.json", {"threshold": 0.5})
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    bundle_dir = run_dir / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(bundle_dir / "report.json", {"dataset": "custom"})
    _write_json(bundle_dir / "config.json", {"config": {"dataset": "custom"}})
    _write_json(bundle_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )
    _write_json(
        bundle_dir / "model_card.json",
        {
            "schema_version": 1,
            "model_name": "bundle_model",
            "summary": {
                "purpose": "Bundle validation",
                "intended_inputs": "RGB",
                "output_contract": "image-level",
            },
            "weights": {
                "path": "missing_model.pt",
                "source": "test",
                "license": "internal",
            },
            "deployment": {"runtime": "torch"},
        },
    )
    _write_json(
        bundle_dir / "bundle_manifest.json",
        build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir),
    )

    quality = evaluate_run_quality(run_dir)

    assert quality["status"] == "audited"
    assert quality["deployable_complete"] is False
    assert quality["weights_audit"]["present"] is True
    assert quality["weights_audit"]["valid"] is False
    assert quality["weights_audit"]["model_card"]["valid"] is False
    assert any(
        "Missing weights file" in item for item in quality["weights_audit"]["model_card"]["errors"]
    )
    trust = quality["trust_summary"]
    assert "deploy_bundle_incomplete" in trust["status_reasons"]
    assert "invalid_bundle_weights_audit" in trust["degraded_by"]
    assert trust["audit_refs"]["deploy_bundle_manifest_json"] == "deploy_bundle/bundle_manifest.json"
