from __future__ import annotations

import json
from pathlib import Path

import pytest

_ALLOWED_JSON_PATHS = {
    "artifacts/calibration_card.json",
    "artifacts/infer_config.json",
    "bundle_manifest.json",
    "calibration_card.json",
    "config.json",
    "environment.json",
    "infer_config.json",
    "report.json",
}


def _resolve_test_path(root: Path, rel_path: str) -> Path:
    if rel_path not in _ALLOWED_JSON_PATHS:
        raise ValueError(f"Unsupported test json path: {rel_path}")
    root_resolved = root.resolve()
    path = (root_resolved / rel_path).resolve()
    path.relative_to(root_resolved)
    return path


def _write_json(root: Path, rel_path: str, payload: dict) -> None:
    path = _resolve_test_path(root, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_json_rejects_unknown_relative_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported test json path"):
        _write_json(tmp_path, "unexpected.json", {"ok": True})


def _make_audited_run(run_dir: Path) -> None:
    _write_json(run_dir, "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir, "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir, "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        run_dir,
        "artifacts/infer_config.json",
        {
            "schema_version": 1,
            "model": {"name": "vision_ecod", "model_kwargs": {}},
            "threshold": 0.5,
            "split_fingerprint": {"sha256": "f" * 64},
        },
    )
    _write_json(
        run_dir,
        "artifacts/calibration_card.json",
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


def test_evaluate_run_acceptance_reports_audited_acceptance_state(tmp_path: Path) -> None:
    from pyimgano.reporting.run_acceptance import evaluate_run_acceptance

    run_dir = tmp_path / "run"
    _make_audited_run(run_dir)

    acceptance = evaluate_run_acceptance(run_dir, required_quality="audited")

    assert acceptance["acceptance_state"] == "audited"
    assert acceptance["reason_codes"] == []


def test_evaluate_run_acceptance_reports_deployable_acceptance_state(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_acceptance import evaluate_run_acceptance

    run_dir = tmp_path / "run"
    _make_audited_run(run_dir)
    bundle_dir = run_dir / "deploy_bundle"
    _write_json(
        bundle_dir,
        "infer_config.json",
        {
            "schema_version": 1,
            "model": {"name": "vision_ecod", "model_kwargs": {}},
            "threshold": 0.5,
            "artifact_quality": {
                "status": "deployable",
                "threshold_scope": "image",
                "has_threshold_provenance": True,
                "has_split_fingerprint": True,
                "has_prediction_policy": False,
                "has_deploy_bundle": True,
                "has_bundle_manifest": True,
                "required_bundle_artifacts_present": True,
                "bundle_artifact_roles": {"infer_config": ["infer_config.json"]},
                "audit_refs": {"calibration_card": "calibration_card.json"},
                "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
            },
        },
    )
    _write_json(bundle_dir, "report.json", {"dataset": "custom"})
    _write_json(bundle_dir, "config.json", {"config": {"dataset": "custom"}})
    _write_json(bundle_dir, "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(
        bundle_dir,
        "calibration_card.json",
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
    _write_json(
        bundle_dir,
        "bundle_manifest.json",
        build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir),
    )

    acceptance = evaluate_run_acceptance(run_dir, required_quality="deployable")

    assert acceptance["acceptance_state"] == "deployable"
    assert acceptance["reason_codes"] == []


def test_evaluate_run_acceptance_reports_blocked_reason_code_for_missing_infer_config(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.run_acceptance import evaluate_run_acceptance

    run_dir = tmp_path / "run"
    _write_json(run_dir, "report.json", {"dataset": "custom", "model": "vision_ecod"})
    _write_json(run_dir, "config.json", {"config": {"dataset": "custom"}})
    _write_json(run_dir, "environment.json", {"fingerprint_sha256": "f" * 64})

    acceptance = evaluate_run_acceptance(run_dir, required_quality="audited")

    assert acceptance["acceptance_state"] == "blocked"
    assert "BUNDLE_MISSING_INFER_CONFIG" in acceptance["reason_codes"]


def test_evaluate_run_acceptance_reports_quality_reason_code_when_gate_not_met(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.run_acceptance import evaluate_run_acceptance

    run_dir = tmp_path / "run"
    _make_audited_run(run_dir)

    acceptance = evaluate_run_acceptance(run_dir, required_quality="deployable")

    assert acceptance["acceptance_state"] == "blocked"
    assert "BUNDLE_REQUIRED_QUALITY_NOT_MET" in acceptance["reason_codes"]
