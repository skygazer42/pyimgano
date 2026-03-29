from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def test_require_run_dir_returns_path_for_valid_report(tmp_path: Path) -> None:
    from pyimgano.services.train_export_helpers import require_run_dir

    run_dir = tmp_path / "run"

    resolved = require_run_dir({"run_dir": str(run_dir)}, deploy_bundle=False)

    assert resolved == run_dir


def test_validate_export_request_requires_save_run_and_pixel_threshold_when_needed() -> None:
    from pyimgano.services.train_export_helpers import validate_export_request

    cfg = SimpleNamespace(
        output=SimpleNamespace(save_run=False),
        defects=SimpleNamespace(enabled=True, pixel_threshold=None),
    )
    request = SimpleNamespace(
        export_infer_config=True,
        export_deploy_bundle=True,
    )

    with pytest.raises(ValueError, match="requires defects.pixel_threshold to be set"):
        validate_export_request(cfg, request)

    cfg = SimpleNamespace(
        output=SimpleNamespace(save_run=False),
        defects=SimpleNamespace(enabled=False, pixel_threshold=None),
    )

    with pytest.raises(ValueError, match="require output.save_run=true"):
        validate_export_request(cfg, request)


def test_build_optional_calibration_card_payload_preserves_prediction_defaults() -> None:
    from pyimgano.services.train_export_helpers import build_optional_calibration_card_payload

    payload = build_optional_calibration_card_payload(
        {
            "threshold": 0.5,
            "threshold_provenance": {
                "method": "fixed",
                "source": "test",
                "score_summary": {"count": 1},
            },
            "split_fingerprint": {"sha256": "a" * 64},
        },
        {
            "prediction": {"reject_confidence_below": 0.75, "reject_label": -9},
            "postprocess": {"threshold_scope": "image"},
        },
    )

    assert payload is not None
    assert payload["prediction_policy"]["reject_confidence_below"] == pytest.approx(0.75)
    assert payload["prediction_policy"]["reject_label"] == -9


def test_rewrite_bundle_paths_copies_relative_checkpoint_and_preserves_relative_path(
    tmp_path: Path,
) -> None:
    from pyimgano.services.train_export_helpers import rewrite_bundle_paths

    run_dir = tmp_path / "run"
    infer_src = run_dir / "artifacts" / "infer_config.json"
    infer_src.parent.mkdir(parents=True, exist_ok=True)
    infer_src.write_text("{}", encoding="utf-8")
    ckpt = run_dir / "checkpoints" / "custom" / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ckpt", encoding="utf-8")
    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    payload = {"checkpoint": {"path": "checkpoints/custom/model.pt"}}

    rewritten = rewrite_bundle_paths(
        payload,
        bundle_dir=bundle_dir,
        infer_src=infer_src,
    )

    assert rewritten["checkpoint"]["path"] == "checkpoints/custom/model.pt"
    assert (bundle_dir / "checkpoints" / "custom" / "model.pt").read_text(encoding="utf-8") == "ckpt"


def test_rewrite_bundle_paths_rewrites_absolute_model_checkpoint(tmp_path: Path) -> None:
    from pyimgano.services.train_export_helpers import rewrite_bundle_paths

    run_dir = tmp_path / "run"
    infer_src = run_dir / "artifacts" / "infer_config.json"
    infer_src.parent.mkdir(parents=True, exist_ok=True)
    infer_src.write_text("{}", encoding="utf-8")
    abs_artifact = tmp_path / "abs_backbone.onnx"
    abs_artifact.write_text("onnx_abs", encoding="utf-8")
    bundle_dir = tmp_path / "deploy_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    payload = {"model": {"checkpoint_path": str(abs_artifact)}}

    rewritten = rewrite_bundle_paths(
        payload,
        bundle_dir=bundle_dir,
        infer_src=infer_src,
    )

    assert rewritten["model"]["checkpoint_path"] == "artifacts_abs/abs_backbone.onnx"
    assert (bundle_dir / "artifacts_abs" / "abs_backbone.onnx").read_text(encoding="utf-8") == "onnx_abs"


def test_apply_bundle_manifest_metadata_updates_artifact_quality_fields() -> None:
    from pyimgano.services.train_export_helpers import apply_bundle_manifest_metadata

    payload = {
        "artifact_quality": {
            "required_bundle_artifacts_present": False,
            "bundle_artifact_roles": {},
        }
    }
    manifest = {
        "required_bundle_artifacts_present": True,
        "artifact_roles": {
            "infer_config": ["infer_config.json"],
            "operator_contract": ["operator_contract.json"],
        },
    }

    apply_bundle_manifest_metadata(payload, manifest)

    assert payload["artifact_quality"]["required_bundle_artifacts_present"] is True
    assert payload["artifact_quality"]["bundle_artifact_roles"] == {
        "infer_config": ["infer_config.json"],
        "operator_contract": ["operator_contract.json"],
    }
