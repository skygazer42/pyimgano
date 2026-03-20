from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_deploy_bundle_manifest_emits_contract_v1_fields(tmp_path: Path) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest

    run_dir = tmp_path / "run"
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(run_dir / "artifacts" / "infer_config.json", {"threshold": 0.5})
    _write_json(
        run_dir / "artifacts" / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "a" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    bundle_dir = tmp_path / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "a" * 64},
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)

    assert manifest["bundle_type"] == "cpu-offline-qc"
    assert manifest["status"] == "draft"
    assert manifest["compatibility"]["schema_family"] == "deploy-bundle"
    assert manifest["input_contract"]["supported_sources"] == [
        "image_dir",
        "single_image",
        "input_manifest.jsonl",
    ]
    assert manifest["output_contract"]["primary_result_file"] == "results.jsonl"
    assert manifest["runtime_policy"] == {
        "batch_gates": {
            "max_anomaly_rate": None,
            "max_reject_rate": None,
            "max_error_rate": None,
            "min_processed": None,
        }
    }
    assert manifest["threshold_summary"]["scope"] == "image"
    assert manifest["threshold_summary"]["has_threshold_provenance"] is True
    assert manifest["threshold_summary"]["has_split_fingerprint"] is True
    assert manifest["threshold_summary"]["calibration_card_ref"] == "calibration_card.json"
    assert manifest["evaluation_summary"]["threshold_scope"] == "image"
    assert manifest["evaluation_summary"]["split_fingerprint_sha256"] == "a" * 64
    assert manifest["artifact_digests"]["infer_config.json"] == next(
        item["sha256"] for item in manifest["entries"] if item["path"] == "infer_config.json"
    )


def test_validate_deploy_bundle_manifest_rejects_tampered_threshold_summary(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    bundle_dir = tmp_path / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    manifest["threshold_summary"]["has_threshold_provenance"] = False

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("threshold_summary" in item for item in errors)


def test_validate_deploy_bundle_manifest_rejects_invalid_runtime_policy_batch_gates(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.deploy_bundle import (
        build_deploy_bundle_manifest,
        validate_deploy_bundle_manifest,
    )

    run_dir = tmp_path / "run"
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    bundle_dir = tmp_path / "deploy_bundle"
    _write_json(bundle_dir / "infer_config.json", {"threshold": 0.5})
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "threshold_context": {"scope": "image", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)
    manifest["runtime_policy"] = {
        "batch_gates": {
            "max_anomaly_rate": 1.2,
            "max_reject_rate": None,
            "max_error_rate": None,
            "min_processed": 0,
        }
    }

    errors = validate_deploy_bundle_manifest(manifest, bundle_dir=bundle_dir, check_hashes=False)

    assert any("runtime_policy.batch_gates.max_anomaly_rate" in item for item in errors)
    assert any("runtime_policy.batch_gates.min_processed" in item for item in errors)


def test_build_deploy_bundle_manifest_emits_optional_pixel_artifacts_when_supported(
    tmp_path: Path,
) -> None:
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest

    run_dir = tmp_path / "run"
    _write_json(run_dir / "environment.json", {"fingerprint_sha256": "f" * 64})
    _write_json(run_dir / "artifacts" / "infer_config.json", {"threshold": 0.5})
    bundle_dir = tmp_path / "deploy_bundle"
    _write_json(
        bundle_dir / "infer_config.json",
        {
            "threshold": 0.5,
            "artifact_quality": {
                "threshold_scope": "pixel",
            },
        },
    )
    _write_json(
        bundle_dir / "calibration_card.json",
        {
            "schema_version": 1,
            "split_fingerprint": {"sha256": "a" * 64},
            "threshold_context": {"scope": "pixel", "category_count": 1},
            "image_threshold": {
                "threshold": 0.5,
                "provenance": {"method": "fixed", "source": "test"},
            },
        },
    )

    manifest = build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)

    assert manifest["output_contract"]["supports_pixel_outputs"] is True
    assert manifest["output_contract"]["optional_artifacts"] == [
        "masks/",
        "overlays/",
        "defects_regions.jsonl",
    ]
