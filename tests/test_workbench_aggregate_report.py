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
            "bottle": {"results": {"auroc": 0.5, "average_precision": 0.4}},
            "cable": {"results": {"auroc": 1.0, "average_precision": 0.7}},
            "capsule": {"results": {"auroc": float("nan"), "average_precision": None}},
        },
    )

    assert payload["dataset"] == "manifest"
    assert payload["category"] == "all"
    assert payload["categories"] == ["bottle", "cable", "capsule"]
    assert payload["per_category"]["bottle"]["results"]["auroc"] == 0.5
    assert math.isclose(payload["mean_metrics"]["auroc"], 0.75)
    assert math.isclose(payload["std_metrics"]["auroc"], 0.25)
    assert math.isclose(payload["mean_metrics"]["average_precision"], 0.55)
    assert math.isclose(payload["std_metrics"]["average_precision"], 0.15)
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
        per_category={"custom": {"results": {"threshold": 0.2}}},
    )

    assert payload["mean_metrics"] == {}
    assert payload["std_metrics"] == {}
