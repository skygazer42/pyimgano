from __future__ import annotations

import json
from pathlib import Path

import pytest


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def test_build_evaluation_matrix_blocks_checkpoint_wrappers_without_artifacts() -> None:
    from pyimgano.services.evaluation_harness_service import build_evaluation_matrix

    matrix = build_evaluation_matrix(
        model_inventory=[
            {
                "name": "vision_anomalib_checkpoint",
                "supports_pixel_map": True,
                "requires_checkpoint": True,
                "supports_save_load": False,
            },
            {
                "name": "vision_ecod",
                "supports_pixel_map": False,
                "requires_checkpoint": False,
                "supports_save_load": True,
            },
        ],
        dataset_inventory=[
            {
                "dataset": "mvtec",
                "root": "/datasets/mvtec",
                "manifest_path": None,
                "categories": ["bottle"],
            }
        ],
        artifact_audit_enabled=True,
        checkpoint_paths={},
    )

    by_model = {row["model"]: row for row in matrix}
    blocked = by_model["vision_anomalib_checkpoint"]
    runnable = by_model["vision_ecod"]

    assert blocked["status"] == "blocked"
    assert blocked["blocking_reasons"] == ["external_artifact_required"]
    assert blocked["planned_evaluations"]["benchmark"] is False
    assert blocked["planned_evaluations"]["artifact_audit"] is False
    assert blocked["supports_pixel_map"] is True

    assert runnable["status"] == "pending"
    assert runnable["blocking_reasons"] == []
    assert runnable["planned_evaluations"]["benchmark"] is True
    assert runnable["planned_evaluations"]["artifact_audit"] is True
    assert runnable["supports_save_load"] is True


def test_run_evaluation_harness_writes_inventory_matrix_and_artifact_audit_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.services.evaluation_harness_service as harness_service
    from pyimgano.services.evaluation_harness_service import (
        ArtifactAuditConfig,
        DatasetEvaluationTarget,
        EvaluationHarnessRequest,
        run_evaluation_harness,
    )

    monkeypatch.setattr(
        harness_service,
        "collect_model_inventory",
        lambda names=None, exclude_names=None: [
            {
                "name": "vision_ecod",
                "supports_pixel_map": False,
                "requires_checkpoint": False,
                "supports_save_load": True,
            }
        ],
    )
    monkeypatch.setattr(
        harness_service,
        "collect_dataset_inventory",
        lambda targets: [
            {
                "dataset": "mvtec",
                "root": "/datasets/mvtec",
                "manifest_path": None,
                "categories": ["bottle"],
            }
        ],
    )
    monkeypatch.setattr(
        harness_service,
        "_run_benchmark_for_entry",
        lambda entry, request: {
            "status": "ok",
            "run_dir": "/tmp/fake-benchmark-run",
            "metrics": {"auroc": 0.91},
        },
    )
    monkeypatch.setattr(
        harness_service,
        "_run_artifact_audit_for_entry",
        lambda entry, request: {
            "status": "ready",
            "run_dir": "/tmp/fake-train-run",
            "infer_config_valid": True,
        },
    )

    payload = run_evaluation_harness(
        EvaluationHarnessRequest(
            output_dir=str(tmp_path / "eval_out"),
            datasets=(
                DatasetEvaluationTarget(
                    name="mvtec",
                    root="/datasets/mvtec",
                ),
            ),
            model_names=("vision_ecod",),
            artifact_audit=ArtifactAuditConfig(
                enabled=True,
                train_config_path="examples/configs/industrial_adapt_audited.json",
            ),
        )
    )

    assert payload["entries_total"] == 1
    assert payload["entries_succeeded"] == 1
    assert payload["entries_failed"] == 0
    assert payload["entries_blocked"] == 0
    assert payload["artifact_audits_attempted"] == 1

    out_dir = Path(payload["output_dir"])
    model_rows = _read_jsonl(out_dir / "model_inventory.jsonl")
    matrix_rows = _read_jsonl(out_dir / "evaluation_matrix.jsonl")
    audit_rows = _read_jsonl(out_dir / "artifact_audit.jsonl")
    dataset_payload = json.loads((out_dir / "dataset_inventory.json").read_text(encoding="utf-8"))

    assert [row["name"] for row in model_rows] == ["vision_ecod"]
    assert dataset_payload["datasets"][0]["dataset"] == "mvtec"
    assert matrix_rows[0]["model"] == "vision_ecod"
    assert matrix_rows[0]["status"] == "ok"
    assert matrix_rows[0]["benchmark"]["metrics"]["auroc"] == pytest.approx(0.91)
    assert audit_rows[0]["model"] == "vision_ecod"
    assert audit_rows[0]["status"] == "ready"


def test_run_evaluation_harness_records_failures_and_continues_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    import pyimgano.services.evaluation_harness_service as harness_service
    from pyimgano.services.evaluation_harness_service import (
        DatasetEvaluationTarget,
        EvaluationHarnessRequest,
        run_evaluation_harness,
    )

    monkeypatch.setattr(
        harness_service,
        "collect_model_inventory",
        lambda names=None, exclude_names=None: [
            {
                "name": "vision_ecod",
                "supports_pixel_map": False,
                "requires_checkpoint": False,
                "supports_save_load": True,
            },
            {
                "name": "vision_patchcore",
                "supports_pixel_map": True,
                "requires_checkpoint": False,
                "supports_save_load": False,
            },
        ],
    )
    monkeypatch.setattr(
        harness_service,
        "collect_dataset_inventory",
        lambda targets: [
            {
                "dataset": "mvtec",
                "root": "/datasets/mvtec",
                "manifest_path": None,
                "categories": ["bottle"],
            }
        ],
    )

    def _fake_benchmark(entry, request):  # noqa: ANN001
        if entry["model"] == "vision_ecod":
            raise RuntimeError("benchmark failed")
        return {"status": "ok", "run_dir": "/tmp/fake-run", "metrics": {"auroc": 0.88}}

    monkeypatch.setattr(harness_service, "_run_benchmark_for_entry", _fake_benchmark)
    monkeypatch.setattr(
        harness_service,
        "_run_artifact_audit_for_entry",
        lambda entry, request: None,
    )

    payload = run_evaluation_harness(
        EvaluationHarnessRequest(
            output_dir=str(tmp_path / "eval_out"),
            datasets=(DatasetEvaluationTarget(name="mvtec", root="/datasets/mvtec"),),
            continue_on_error=True,
        )
    )

    rows = _read_jsonl(Path(payload["evaluation_matrix_path"]))
    by_model = {row["model"]: row for row in rows}

    assert payload["entries_total"] == 2
    assert payload["entries_failed"] == 1
    assert payload["entries_succeeded"] == 1
    assert by_model["vision_ecod"]["status"] == "error"
    assert "benchmark failed" in by_model["vision_ecod"]["error"]
    assert by_model["vision_patchcore"]["status"] == "ok"
