from __future__ import annotations

import json
from pathlib import Path


def test_validate_infer_config_cli_smoke(tmp_path: Path, capsys) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {
                    "name": "vision_patchcore",
                    "device": "cpu",
                    "pretrained": False,
                    "contamination": 0.1,
                    "model_kwargs": {},
                },
                "defects": {
                    "enabled": True,
                    "pixel_threshold_strategy": "normal_pixel_quantile",
                    "pixel_normal_quantile": 0.999,
                    "mask_format": "png",
                },
                "prediction": {
                    "reject_confidence_below": 0.75,
                    "reject_label": -9,
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "ok" in out


def test_validate_infer_config_cli_rejects_bad_mask_format(tmp_path: Path, capsys) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "defects": {"mask_format": "nope"},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "mask_format" in err


def test_validate_infer_config_cli_backfills_legacy_schema_version(tmp_path: Path, capsys) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "defects": {"mask_format": "png"},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["schema_version"] == 1


def test_validate_infer_config_cli_rejects_future_schema_version(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "schema_version" in err


def test_validate_infer_config_cli_rejects_bad_prediction_threshold(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "prediction": {"reject_confidence_below": 1.5},
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "reject_confidence_below" in err


def test_validate_infer_config_cli_rejects_bad_prediction_label(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "prediction": {
                    "reject_confidence_below": 0.75,
                    "reject_label": "not-an-int",
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "reject_label" in err


def test_validate_infer_config_cli_rejects_bad_artifact_quality_status(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "artifact_quality": {
                    "status": "unknown",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": False,
                    "has_prediction_policy": False,
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg), "--no-check-files"])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "artifact_quality.status" in err


def test_validate_infer_config_cli_rejects_missing_audit_ref_file(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    run_dir = tmp_path / "run"
    cfg = run_dir / "artifacts" / "infer_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "artifact_quality": {
                    "status": "audited",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "audit_refs": {"calibration_card": "artifacts/calibration_card.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "artifact_quality.audit_refs.calibration_card" in err


def test_validate_infer_config_cli_rejects_missing_deploy_ref_file(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    bundle_dir = tmp_path / "deploy_bundle"
    cfg = bundle_dir / "infer_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "artifact_quality": {
                    "status": "deployable",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "has_deploy_bundle": True,
                    "has_bundle_manifest": True,
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                    "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "calibration_card.json").write_text("{}", encoding="utf-8")

    rc = main([str(cfg)])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "artifact_quality.deploy_refs.bundle_manifest" in err


def test_validate_infer_config_cli_plain_output_shows_artifact_quality(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    cfg = tmp_path / "infer_config.json"
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "artifact_quality": {
                    "status": "audited",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": False,
                    "has_prediction_policy": False,
                    "has_deploy_bundle": False,
                    "has_bundle_manifest": False,
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                    "deploy_refs": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "calibration_card.json").write_text("{}", encoding="utf-8")

    rc = main([str(cfg)])

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "audit_status=audited" in out
    assert "threshold_scope=image" in out
    assert "deploy_bundle=false" in out
    assert "bundle_manifest=false" in out
    assert "trust_status=partial" in out
    assert "degraded_by=missing_split_fingerprint" in out
    assert "audit_ref.calibration_card=calibration_card.json" in out


def test_validate_infer_config_cli_reports_bundle_completeness_metadata(
    tmp_path: Path, capsys
) -> None:
    from pyimgano.validate_infer_config_cli import main

    bundle_dir = tmp_path / "deploy_bundle"
    cfg = bundle_dir / "infer_config.json"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        json.dumps(
            {
                "model": {"name": "vision_patchcore", "model_kwargs": {}},
                "artifact_quality": {
                    "status": "deployable",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "has_deploy_bundle": True,
                    "has_bundle_manifest": True,
                    "required_bundle_artifacts_present": True,
                    "bundle_artifact_roles": {
                        "infer_config": ["infer_config.json"],
                        "report": ["report.json"],
                        "config": ["config.json"],
                        "environment": ["environment.json"],
                    },
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                    "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "calibration_card.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    rc = main([str(cfg), "--json"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    trust = out["validation_trust"]
    assert trust["trust_signals"]["has_required_bundle_artifacts"] is True
    assert trust["trust_signals"]["has_bundle_artifact_roles"] is True

    rc = main([str(cfg)])

    assert rc == 0
    plain = capsys.readouterr().out.lower()
    assert "bundle_required=true" in plain
