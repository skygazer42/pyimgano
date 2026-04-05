from __future__ import annotations

import json
from pathlib import Path

from pyimgano.services.train_service import TrainRunRequest, build_train_dry_run_payload


def test_build_train_dry_run_payload_returns_config_payload(tmp_path) -> None:
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "recipe": "industrial-adapt",
                "dataset": {"name": "custom", "root": "/tmp/data"},
                "model": {"name": "vision_patchcore"},
                "output": {"save_run": False},
            }
        ),
        encoding="utf-8",
    )

    payload = build_train_dry_run_payload(TrainRunRequest(config_path=str(cfg_path)))

    assert payload["config"]["recipe"] == "industrial-adapt"


def test_export_deploy_bundle_orchestrates_helpers_before_manifest_write(
    tmp_path: Path, monkeypatch
) -> None:
    import pyimgano.services.train_service as train_service

    run_dir = tmp_path / "run"
    infer_src = run_dir / "artifacts" / "infer_config.json"
    infer_src.parent.mkdir(parents=True, exist_ok=True)
    infer_src.write_text("{}", encoding="utf-8")

    call_order: list[str] = []
    manifest_counter = {"count": 0}

    def _fake_copy_supporting_files(**kwargs):  # noqa: ANN003
        _ = kwargs
        call_order.append("copy_supporting_files")

    def _fake_prepare_payload(payload, **kwargs):  # noqa: ANN001, ANN003
        _ = kwargs
        call_order.append("prepare_payload")
        return {"artifact_quality": {}, **dict(payload)}

    def _fake_rewrite_paths(payload, **kwargs):  # noqa: ANN001, ANN003
        _ = kwargs
        call_order.append("rewrite_paths")
        return dict(payload)

    def _fake_build_handoff_report(**kwargs):  # noqa: ANN003
        _ = kwargs
        return {"handoff": True}

    def _fake_build_manifest(**kwargs):  # noqa: ANN003
        _ = kwargs
        manifest_counter["count"] += 1
        call_order.append(f"build_manifest_{manifest_counter['count']}")
        return {"build_index": manifest_counter["count"], "artifact_roles": {}}

    def _fake_apply_manifest_metadata(payload, manifest):  # noqa: ANN001
        _ = payload, manifest
        call_order.append("apply_manifest_metadata")

    def _fake_save_run_report(path, payload):  # noqa: ANN001
        _ = payload
        rel = Path(path).name
        if rel == "handoff_report.json":
            call_order.append("write_handoff_report")
        elif rel == "bundle_manifest.json":
            call_order.append("write_final_manifest")
        elif rel == "infer_config.json":
            call_order.append("write_infer_config")

    monkeypatch.setattr(
        train_service,
        "_copy_deploy_bundle_supporting_files_helper",
        _fake_copy_supporting_files,
    )
    monkeypatch.setattr(
        train_service,
        "_prepare_bundle_infer_config_payload_helper",
        _fake_prepare_payload,
    )
    monkeypatch.setattr(
        train_service,
        "_rewrite_bundle_paths_helper",
        _fake_rewrite_paths,
    )
    monkeypatch.setattr(
        train_service,
        "build_deploy_bundle_handoff_report",
        _fake_build_handoff_report,
    )
    monkeypatch.setattr(
        train_service,
        "build_deploy_bundle_manifest",
        _fake_build_manifest,
    )
    monkeypatch.setattr(
        train_service,
        "_apply_bundle_manifest_metadata_helper",
        _fake_apply_manifest_metadata,
    )
    monkeypatch.setattr(train_service, "save_run_report", _fake_save_run_report)

    bundle_dir = train_service._export_deploy_bundle(
        run_dir=run_dir,
        infer_config_payload={"artifact_quality": {}},
    )

    assert bundle_dir == run_dir / "deploy_bundle"
    expected = [
        "copy_supporting_files",
        "prepare_payload",
        "rewrite_paths",
        "write_infer_config",
        "write_handoff_report",
        "build_manifest_1",
        "apply_manifest_metadata",
        "write_infer_config",
        "build_manifest_2",
        "write_final_manifest",
    ]
    assert call_order == expected
