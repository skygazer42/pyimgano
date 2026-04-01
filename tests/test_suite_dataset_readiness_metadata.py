from __future__ import annotations

import json


def _write_run_artifacts(root):
    (root / "report.json").write_text(json.dumps({"suite": "industrial-v4"}), encoding="utf-8")
    (root / "config.json").write_text(json.dumps({"config": {"seed": 123}}), encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )


def test_export_suite_tables_carries_dataset_readiness_issue_codes(tmp_path) -> None:
    from pyimgano.reporting.suite_export import export_suite_tables

    _write_run_artifacts(tmp_path)
    payload = {
        "suite": "industrial-ci",
        "dataset": "custom",
        "category": "custom",
        "dataset_profile": {
            "total_records": 3,
            "train_count": 1,
            "test_count": 2,
            "test_normal_count": 1,
            "test_anomaly_count": 1,
            "categories": ["custom"],
            "category_count": 1,
            "has_masks": True,
            "pixel_metrics_available": True,
            "fewshot_risk": True,
            "multi_category": False,
        },
        "dataset_readiness": {
            "status": "warning",
            "issue_codes": ["FEWSHOT_TRAIN_SET"],
            "issue_details": [
                {
                    "code": "FEWSHOT_TRAIN_SET",
                    "message": "Train split has fewer than 16 normal samples; results may be unstable.",
                }
            ],
        },
        "rows": [{"name": "baseline_a", "auroc": 0.8, "run_dir": "runs/a"}],
    }

    export_suite_tables(payload, tmp_path, formats=["csv"])
    metadata = json.loads((tmp_path / "leaderboard_metadata.json").read_text(encoding="utf-8"))

    assert metadata["dataset_readiness"]["status"] == "warning"
    assert metadata["dataset_readiness"]["issue_codes"] == ["FEWSHOT_TRAIN_SET"]
    assert metadata["benchmark_context"]["dataset_readiness_status"] == "warning"
    assert metadata["benchmark_context"]["dataset_issue_codes"] == ["FEWSHOT_TRAIN_SET"]
