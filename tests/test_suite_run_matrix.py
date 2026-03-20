from __future__ import annotations

import importlib

import pytest


def test_run_baseline_suite_category_all_builds_matrix_and_uses_mean_metrics(
    monkeypatch,
) -> None:
    import pyimgano.pipelines.run_suite as run_suite

    run_benchmark_module = importlib.import_module("pyimgano.pipelines.run_benchmark")

    monkeypatch.setattr(run_suite, "_missing_extras_hint_for_baseline", lambda baseline: None)

    def _fake_run_benchmark(**kwargs):  # noqa: ANN003 - test seam
        return {
            "category": "all",
            "model": str(kwargs["model"]),
            "categories": ["bottle", "capsule"],
            "mean_metrics": {
                "auroc": 0.8,
                "average_precision": 0.7,
            },
            "std_metrics": {
                "auroc": 0.1,
                "average_precision": 0.05,
            },
            "per_category": {
                "bottle": {
                    "results": {
                        "auroc": 0.9,
                        "average_precision": 0.75,
                    }
                },
                "capsule": {
                    "results": {
                        "auroc": 0.7,
                        "average_precision": 0.65,
                    }
                },
            },
        }

    monkeypatch.setattr(run_benchmark_module, "run_benchmark", _fake_run_benchmark)

    payload = run_suite.run_baseline_suite(
        suite="industrial-ci",
        dataset="custom",
        root="/tmp/custom",
        manifest_path=None,
        category="all",
        save_run=False,
        continue_on_error=False,
    )

    rows = payload.get("rows")
    assert isinstance(rows, list)
    assert rows
    assert all(float(row["auroc"]) == pytest.approx(0.8) for row in rows)
    assert all(float(row["average_precision"]) == pytest.approx(0.7) for row in rows)

    matrix = payload.get("matrix")
    assert isinstance(matrix, dict)
    assert matrix["scope"] == "per_category"
    assert matrix["categories"] == ["bottle", "capsule"]
    assert matrix["metrics"] == ["auroc", "average_precision"]

    auroc_rows = matrix["by_metric"]["auroc"]
    assert isinstance(auroc_rows, list)
    assert len(auroc_rows) == len(rows)
    assert auroc_rows[0]["values"]["bottle"] == pytest.approx(0.9)
    assert auroc_rows[0]["values"]["capsule"] == pytest.approx(0.7)
    assert auroc_rows[0]["mean"] == pytest.approx(0.8)
    assert auroc_rows[0]["std"] == pytest.approx(0.1)
