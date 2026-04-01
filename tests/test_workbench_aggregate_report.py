from __future__ import annotations

import math

from pyimgano.reporting.report import REPORT_SCHEMA_VERSION
from pyimgano.workbench.aggregate_report import build_workbench_aggregate_report
from pyimgano.workbench.config import WorkbenchConfig


def test_workbench_aggregate_report_builds_stamped_payload_and_metric_aggregates() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "seed": 123,
            "dataset": {
                "name": "manifest",
                "root": "/tmp/data",
                "manifest_path": "/tmp/data/manifest.jsonl",
                "category": "all",
                "resize": [16, 16],
                "input_mode": "paths",
            },
            "model": {
                "name": "vision_ecod",
                "device": "cpu",
                "pretrained": False,
                "contamination": 0.1,
            },
            "output": {"save_run": False},
        }
    )

    payload = build_workbench_aggregate_report(
        config=cfg,
        recipe_name="industrial-adapt",
        categories=["bottle", "cable", "capsule"],
        per_category={
            "bottle": {
                "results": {"auroc": 0.5, "average_precision": 0.4},
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
            },
            "cable": {
                "results": {"auroc": 1.0, "average_precision": 0.7},
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["PIXEL_METRICS_UNAVAILABLE"],
                    "issue_details": [
                        {
                            "code": "PIXEL_METRICS_UNAVAILABLE",
                            "message": "Pixel metrics are unavailable because anomaly masks are missing or no anomalous test samples carry masks.",
                        }
                    ],
                },
            },
            "capsule": {
                "results": {"auroc": float("nan"), "average_precision": None},
                "dataset_readiness": {
                    "status": "ok",
                    "issue_codes": [],
                    "issue_details": [],
                },
            },
        },
    )

    assert payload["dataset"] == "manifest"
    assert payload["category"] == "all"
    assert payload["categories"] == ["bottle", "cable", "capsule"]
    assert math.isclose(payload["per_category"]["bottle"]["results"]["auroc"], 0.5)
    assert math.isclose(payload["mean_metrics"]["auroc"], 0.75)
    assert math.isclose(payload["std_metrics"]["auroc"], 0.25)
    assert math.isclose(payload["mean_metrics"]["average_precision"], 0.55)
    assert math.isclose(payload["std_metrics"]["average_precision"], 0.15)
    assert payload["dataset_readiness"]["status"] == "warning"
    assert payload["dataset_readiness"]["issue_codes"] == [
        "FEWSHOT_TRAIN_SET",
        "PIXEL_METRICS_UNAVAILABLE",
    ]
    assert payload["schema_version"] == int(REPORT_SCHEMA_VERSION)
    assert isinstance(payload["timestamp_utc"], str)


def test_workbench_aggregate_report_ignores_missing_metrics() -> None:
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "industrial-adapt",
            "dataset": {
                "name": "custom",
                "root": "/tmp/data",
                "category": "all",
                "input_mode": "paths",
            },
            "model": {"name": "vision_ecod"},
            "output": {"save_run": False},
        }
    )

    payload = build_workbench_aggregate_report(
        config=cfg,
        recipe_name="industrial-adapt",
        categories=["custom"],
        per_category={
            "custom": {
                "results": {"threshold": 0.2},
                "dataset_readiness": {
                    "status": "error",
                    "issue_codes": ["CUSTOM_DATASET_INVALID_STRUCTURE"],
                    "issue_details": [
                        {
                            "code": "CUSTOM_DATASET_INVALID_STRUCTURE",
                            "message": "Custom dataset layout validation failed.",
                        }
                    ],
                },
            }
        },
    )

    assert payload["mean_metrics"] == {}
    assert payload["std_metrics"] == {}
    assert payload["dataset_readiness"]["status"] == "error"
    assert payload["dataset_readiness"]["issue_codes"] == ["CUSTOM_DATASET_INVALID_STRUCTURE"]
