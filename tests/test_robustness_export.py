from __future__ import annotations

import csv
import json


def test_export_robustness_tables_writes_csv_and_summary(tmp_path) -> None:
    from pyimgano.reporting.robustness_export import export_robustness_tables

    payload = {
        "dataset": "mvtec",
        "category": "bottle",
        "model": "vision_ecod",
        "robustness_summary": {
            "clean_auroc": 0.95,
            "mean_corruption_auroc": 0.85,
            "worst_corruption_auroc": 0.8,
        },
        "robustness": {
            "clean": {
                "latency_ms_per_image": 1.0,
                "results": {"auroc": 0.95, "average_precision": 0.91},
            },
            "corruptions": {
                "lighting": {
                    "severity_1": {
                        "latency_ms_per_image": 1.2,
                        "results": {"auroc": 0.9, "average_precision": 0.86},
                    },
                    "severity_2": {
                        "latency_ms_per_image": 1.3,
                        "results": {"auroc": 0.8, "average_precision": 0.77},
                    },
                }
            },
        },
    }

    written = export_robustness_tables(payload, tmp_path)

    assert "conditions_csv" in written
    assert "summary_json" in written

    with (tmp_path / "robustness_conditions.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["condition"] == "clean"
    assert rows[0]["severity"] == ""
    assert rows[0]["drop_auroc"] == "0.0"
    assert rows[1]["condition"] == "lighting"
    assert rows[1]["severity"] == "1"
    assert rows[1]["drop_auroc"] == "0.05"
    assert rows[1]["drop_average_precision"] == "0.05"
    assert rows[2]["auroc"] == "0.8"
    assert rows[2]["drop_auroc"] == "0.15"

    summary = json.loads((tmp_path / "robustness_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset"] == "mvtec"
    assert summary["robustness_summary"]["worst_corruption_auroc"] == 0.8
    assert summary["robustness_protocol"]["condition_count"] == 3
    assert summary["robustness_protocol"]["corruption_count"] == 1
    assert summary["robustness_protocol"]["severities"] == [1, 2]
    assert summary["robustness_protocol"]["comparability_hints"] == {
        "recommends_same_environment": True,
        "requires_same_category": True,
        "requires_same_corruption_protocol": True,
        "requires_same_dataset": True,
        "requires_same_input_mode": True,
        "requires_same_resize": True,
        "requires_same_severities": True,
        "requires_same_split": True,
    }
    trust = summary["trust_summary"]
    assert trust["status"] == "trust-signaled"
    assert trust["trust_signals"]["has_clean_baseline"] is True
    assert trust["trust_signals"]["has_corruption_conditions"] is True
    assert trust["trust_signals"]["has_summary_metrics"] is True
    assert trust["trust_signals"]["has_latency_profile"] is True
    assert trust["audit_refs"]["robustness_conditions_csv"] == "robustness_conditions.csv"
    assert summary["condition_count"] == 3
