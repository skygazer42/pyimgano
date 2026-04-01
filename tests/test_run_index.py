import hashlib
import json

import pytest


def test_list_run_summaries_sorts_newest_first(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    run_a = tmp_path / "20260317_100000_mvtec_suite_a"
    run_b = tmp_path / "20260317_110000_mvtec_suite_b"
    run_a.mkdir()
    run_b.mkdir()

    (run_a / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_patchcore",
                "timestamp_utc": "2026-03-17T10:00:00+00:00",
                "results": {"auroc": 0.91},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "suite": "industrial-v4",
                "timestamp_utc": "2026-03-17T11:00:00+00:00",
                "summary": {"by_auroc": [{"name": "x", "auroc": 0.93}]},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "suite": "industrial-v4",
                "timestamp_utc": "2026-03-17T11:00:00+00:00",
                "summary": {"by_auroc": [{"name": "x", "auroc": 0.93}]},
                "split_fingerprint": {"sha256": "a" * 64},
            }
        ),
        encoding="utf-8",
    )
    items = list_run_summaries(tmp_path)

    assert [item["run_dir_name"] for item in items] == [run_b.name, run_a.name]
    assert items[0]["environment_fingerprint_sha256"] == "f" * 64
    assert items[0]["split_fingerprint_sha256"] == "a" * 64
    assert items[0]["kind"] == "suite"
    assert items[1]["kind"] == "benchmark"


def test_list_run_summaries_can_filter_by_kind_and_dataset(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    suite_dir = tmp_path / "suite_run"
    bench_dir = tmp_path / "bench_run"
    suite_dir.mkdir()
    bench_dir.mkdir()

    (suite_dir / "report.json").write_text(
        json.dumps(
            {
                "suite": "industrial-v4",
                "dataset": "mvtec",
                "timestamp_utc": "2026-03-17T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (bench_dir / "report.json").write_text(
        json.dumps(
            {
                "model": "vision_patchcore",
                "dataset": "visa",
                "timestamp_utc": "2026-03-17T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path, kind="suite", dataset="mvtec")

    assert len(items) == 1
    assert items[0]["kind"] == "suite"
    assert items[0]["dataset"] == "mvtec"


def test_compare_run_summaries_collects_metric_table(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "a", "results": {"auroc": 0.9}}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "suite": "industrial-v4", "rows": [{"auroc": 0.92}]}),
        encoding="utf-8",
    )

    payload = compare_run_summaries([run_a, run_b])

    assert len(payload["runs"]) == 2
    assert payload["metrics"]["auroc"]["max"] == pytest.approx(0.92)
    assert payload["metrics"]["auroc"]["min"] == pytest.approx(0.9)
    assert payload["evaluation_contract"]["primary_metric"] == "auroc"
    assert payload["evaluation_contract"]["metric_directions"]["auroc"] == "higher_is_better"
    assert payload["evaluation_contract"]["comparability_hints"] == {
        "requires_same_dataset": True,
        "requires_same_category": True,
        "requires_same_split": True,
        "recommends_same_environment": True,
    }


def test_compare_run_summaries_can_report_deltas_vs_baseline(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "a", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "b", "results": {"auroc": 0.89}}),
        encoding="utf-8",
    )

    payload = compare_run_summaries([run_a, run_b], baseline_run_dir=run_a)

    metric = payload["metrics"]["auroc"]
    assert metric["direction"] == "higher_is_better"
    assert metric["baseline"] == pytest.approx(0.91)
    assert metric["comparisons"][0]["status"] == "baseline"
    assert metric["comparisons"][1]["status"] == "regressed"
    assert metric["comparisons"][1]["delta_vs_baseline"] == pytest.approx(-0.02)


def test_list_run_summaries_can_filter_by_same_split(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_c = tmp_path / "run_c"
    run_a.mkdir()
    run_b.mkdir()
    run_c.mkdir()
    (run_a / "report.json").write_text(
        json.dumps({"model": "a", "split_fingerprint": {"sha256": "f" * 64}}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps({"model": "b", "split_fingerprint": {"sha256": "f" * 64}}),
        encoding="utf-8",
    )
    (run_c / "report.json").write_text(
        json.dumps({"model": "c", "split_fingerprint": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path, same_split_as=run_a)

    assert [item["run_dir_name"] for item in items] == [run_b.name, run_a.name]


def test_list_run_summaries_can_filter_by_same_environment(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    mismatched = tmp_path / "mismatched"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    (baseline / "report.json").write_text(
        json.dumps({"model": "baseline"}),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps({"model": "matched"}),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps({"model": "mismatched"}),
        encoding="utf-8",
    )
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (matched / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (mismatched / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "1" * 64}),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path, same_environment_as=baseline)

    assert [item["run_dir_name"] for item in items] == [matched.name, baseline.name]


def test_compare_run_summaries_emits_trust_comparison_summary(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    for run_dir, model_name, metric_value in (
        (baseline, "baseline", 0.91),
        (candidate, "candidate", 0.92),
    ):
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "dataset": "mvtec",
                    "category": "bottle",
                    "model": model_name,
                    "results": {"auroc": metric_value},
                    "split_fingerprint": {"sha256": "f" * 64},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )
    (baseline / "artifacts").mkdir()
    (baseline / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "split_fingerprint": {"sha256": "f" * 64},
                "prediction": {"reject_confidence_below": 0.75, "reject_label": -9},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)

    trust = payload["trust_comparison"]
    assert trust["checked"] is True
    assert trust["quality_status"] == "audited"
    assert trust["gate"] == "limited"
    assert trust["status"] == "partial"
    assert trust["reason"] == "missing_split_fingerprint"
    assert trust["operator_contract_status"] == "missing"
    assert trust["operator_contract_consistent"] is False
    assert trust["status_reasons"] == [
        "core_artifacts_present",
        "calibration_audit_consistent",
        "warnings_present",
    ]
    assert trust["degraded_by"] == [
        "missing_split_fingerprint",
        "missing_prediction_policy",
        "missing_threshold_context",
    ]
    assert trust["audit_refs"]["infer_config_json"] == "artifacts/infer_config.json"
    assert trust["audit_refs"]["calibration_card_json"] == "artifacts/calibration_card.json"


def test_list_run_summaries_can_filter_by_same_robustness_protocol(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    mismatched = tmp_path / "mismatched"
    missing = tmp_path / "missing"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    missing.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "model": "baseline",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.95}},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.90}},
                            "severity_2": {"results": {"auroc": 0.85}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps(
            {
                "model": "matched",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.94}},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.89}},
                            "severity_2": {"results": {"auroc": 0.84}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps(
            {
                "model": "mismatched",
                "input_mode": "paths",
                "resize": [512, 512],
                "robustness": {
                    "corruption_mode": "clean_only",
                    "clean": {"results": {"auroc": 0.93}},
                    "corruptions": {
                        "jpeg": {
                            "severity_1": {"results": {"auroc": 0.88}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (missing / "report.json").write_text(
        json.dumps({"model": "missing", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path, same_robustness_protocol_as=baseline)

    assert [item["run_dir_name"] for item in items] == [matched.name, baseline.name]


def test_latest_run_summary_can_filter_by_same_target(tmp_path):
    from pyimgano.reporting.run_index import latest_run_summary

    baseline = tmp_path / "20260318_100000_baseline"
    matched_old = tmp_path / "20260318_101000_matched_old"
    matched_new = tmp_path / "20260318_102000_matched_new"
    mismatched = tmp_path / "20260318_103000_mismatched"
    baseline.mkdir()
    matched_old.mkdir()
    matched_new.mkdir()
    mismatched.mkdir()
    baseline_report = {"dataset": "mvtec", "category": "bottle", "model": "baseline"}
    matched_report = {"dataset": "mvtec", "category": "bottle", "model": "matched"}
    mismatched_report = {"dataset": "visa", "category": "capsule", "model": "mismatched"}
    (baseline / "report.json").write_text(json.dumps(baseline_report), encoding="utf-8")
    (matched_old / "report.json").write_text(
        json.dumps({**matched_report, "timestamp_utc": "2026-03-18T10:10:00+00:00"}),
        encoding="utf-8",
    )
    (matched_new / "report.json").write_text(
        json.dumps({**matched_report, "timestamp_utc": "2026-03-18T10:20:00+00:00"}),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps({**mismatched_report, "timestamp_utc": "2026-03-18T10:30:00+00:00"}),
        encoding="utf-8",
    )

    item = latest_run_summary(tmp_path, same_target_as=baseline)

    assert item is not None
    assert item["run_dir_name"] == matched_new.name


def test_latest_run_summary_can_filter_by_same_robustness_protocol(tmp_path):
    from pyimgano.reporting.run_index import latest_run_summary

    baseline = tmp_path / "20260318_100000_baseline"
    matched_old = tmp_path / "20260318_101000_matched_old"
    matched_new = tmp_path / "20260318_102000_matched_new"
    mismatched = tmp_path / "20260318_103000_mismatched"
    baseline.mkdir()
    matched_old.mkdir()
    matched_new.mkdir()
    mismatched.mkdir()
    baseline_report = {
        "input_mode": "numpy",
        "resize": [256, 256],
        "robustness": {
            "corruption_mode": "full",
            "clean": {"results": {"auroc": 0.95}},
            "corruptions": {
                "lighting": {
                    "severity_1": {"results": {"auroc": 0.90}},
                }
            },
        },
    }
    matched_report = {
        "input_mode": "numpy",
        "resize": [256, 256],
        "robustness": {
            "corruption_mode": "full",
            "clean": {"results": {"auroc": 0.94}},
            "corruptions": {
                "lighting": {
                    "severity_1": {"results": {"auroc": 0.89}},
                }
            },
        },
    }
    mismatched_report = {
        "input_mode": "paths",
        "resize": [512, 512],
        "robustness": {
            "corruption_mode": "clean_only",
            "clean": {"results": {"auroc": 0.93}},
            "corruptions": {},
        },
    }
    (baseline / "report.json").write_text(json.dumps(baseline_report), encoding="utf-8")
    (matched_old / "report.json").write_text(
        json.dumps({**matched_report, "timestamp_utc": "2026-03-18T10:10:00+00:00"}),
        encoding="utf-8",
    )
    (matched_new / "report.json").write_text(
        json.dumps({**matched_report, "timestamp_utc": "2026-03-18T10:20:00+00:00"}),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps({**mismatched_report, "timestamp_utc": "2026-03-18T10:30:00+00:00"}),
        encoding="utf-8",
    )

    item = latest_run_summary(tmp_path, same_robustness_protocol_as=baseline)

    assert item is not None
    assert item["run_dir_name"] == matched_new.name


def test_list_run_summaries_include_artifact_quality(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "model": "vision_ecod",
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path)

    assert items[0]["artifact_quality"]["status"] == "reproducible"
    assert items[0]["artifact_quality"]["missing_required"] == []
    assert items[0]["dataset_readiness_status"] == "warning"
    assert items[0]["dataset_issue_codes"] == ["FEWSHOT_TRAIN_SET"]
    assert items[0]["evaluation_contract"]["primary_metric"] == "auroc"
    assert items[0]["evaluation_contract"]["metric_directions"]["auroc"] == "higher_is_better"
    assert items[0]["evaluation_contract"]["comparability_hints"]["requires_same_split"] is True


def test_list_run_summaries_can_filter_by_min_quality(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    partial = tmp_path / "20260318_100000_partial"
    reproducible_old = tmp_path / "20260318_101000_repro_old"
    reproducible_new = tmp_path / "20260318_102000_repro_new"
    partial.mkdir()
    reproducible_old.mkdir()
    reproducible_new.mkdir()
    (partial / "report.json").write_text(
        json.dumps(
            {"dataset": "custom", "model": "partial", "timestamp_utc": "2026-03-18T10:00:00+00:00"}
        ),
        encoding="utf-8",
    )
    for run_dir, model_name, timestamp in (
        (reproducible_old, "repro_old", "2026-03-18T10:10:00+00:00"),
        (reproducible_new, "repro_new", "2026-03-18T10:20:00+00:00"),
    ):
        (run_dir / "report.json").write_text(
            json.dumps({"dataset": "custom", "model": model_name, "timestamp_utc": timestamp}),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "f" * 64}),
            encoding="utf-8",
        )

    items = list_run_summaries(tmp_path, min_quality="reproducible")

    assert [item["run_dir_name"] for item in items] == [
        reproducible_new.name,
        reproducible_old.name,
    ]


def test_list_run_summaries_classifies_robustness_and_extracts_metrics(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    run_dir = tmp_path / "robust_run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_ecod",
                "robustness": {
                    "clean": {"results": {"auroc": 0.95}, "latency_ms_per_image": 1.0},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.90}, "latency_ms_per_image": 1.2},
                            "severity_2": {"results": {"auroc": 0.80}, "latency_ms_per_image": 1.3},
                        },
                        "blur": {
                            "severity_1": {"results": {"auroc": 0.85}, "latency_ms_per_image": 1.1},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    items = list_run_summaries(tmp_path)

    assert items[0]["kind"] == "robustness"
    assert items[0]["robustness_protocol"]["condition_count"] == 4
    assert items[0]["robustness_protocol"]["corruption_count"] == 2
    assert items[0]["robustness_protocol"]["comparability_hints"] == {
        "recommends_same_environment": True,
        "requires_same_category": True,
        "requires_same_corruption_protocol": True,
        "requires_same_dataset": True,
        "requires_same_input_mode": True,
        "requires_same_resize": True,
        "requires_same_severities": True,
        "requires_same_split": True,
    }
    assert items[0]["robustness_trust"]["status"] == "trust-signaled"
    assert items[0]["robustness_trust"]["trust_signals"]["has_corruption_conditions"] is True
    assert items[0]["metrics"]["clean_auroc"] == pytest.approx(0.95)
    assert items[0]["metrics"]["mean_corruption_auroc"] == pytest.approx(0.85)
    assert items[0]["metrics"]["worst_corruption_auroc"] == pytest.approx(0.8)
    assert items[0]["metrics"]["mean_corruption_drop_auroc"] == pytest.approx(0.1)
    assert items[0]["metrics"]["worst_corruption_drop_auroc"] == pytest.approx(0.15)
    assert items[0]["metrics"]["clean_latency_ms_per_image"] == pytest.approx(1.0)
    assert items[0]["metrics"]["mean_corruption_latency_ms_per_image"] == pytest.approx(1.2)
    assert items[0]["metrics"]["worst_corruption_latency_ms_per_image"] == pytest.approx(1.3)
    assert items[0]["metrics"]["mean_corruption_latency_ratio"] == pytest.approx(1.2)
    assert items[0]["metrics"]["worst_corruption_latency_ratio"] == pytest.approx(1.3)
    assert items[0]["evaluation_contract"]["primary_metric"] == "worst_corruption_auroc"
    assert items[0]["evaluation_contract"]["ranking_metric"] == "worst_corruption_auroc"
    assert (
        items[0]["evaluation_contract"]["metric_directions"]["worst_corruption_drop_auroc"]
        == "lower_is_better"
    )
    assert (
        items[0]["evaluation_contract"]["comparability_hints"]["requires_same_corruption_protocol"]
        is True
    )


def test_list_run_summaries_revalidates_saved_robustness_artifact_digests(tmp_path):
    from pyimgano.reporting.run_index import list_run_summaries

    run_dir = tmp_path / "robust_run"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    conditions_csv = artifacts_dir / "robustness_conditions.csv"
    summary_json = artifacts_dir / "robustness_summary.json"
    conditions_csv.write_text("condition,severity,auroc\nclean,,0.95\n", encoding="utf-8")
    summary_json.write_text(
        json.dumps({"trust_summary": {"status": "trust-signaled"}}),
        encoding="utf-8",
    )
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_patchcore",
                "robustness_summary": {
                    "clean_auroc": 0.95,
                    "worst_corruption_auroc": 0.85,
                },
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.95}, "latency_ms_per_image": 1.0},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {
                                "results": {"auroc": 0.85},
                                "latency_ms_per_image": 1.2,
                            }
                        }
                    },
                },
                "robustness_trust": {
                    "status": "trust-signaled",
                    "audit_refs": {
                        "robustness_conditions_csv": "artifacts/robustness_conditions.csv",
                        "robustness_summary_json": "artifacts/robustness_summary.json",
                    },
                    "audit_digests": {
                        "robustness_conditions_csv": hashlib.sha256(
                            conditions_csv.read_bytes()
                        ).hexdigest(),
                        "robustness_summary_json": hashlib.sha256(
                            summary_json.read_bytes()
                        ).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    conditions_csv.write_text("condition,severity,auroc\nclean,,0.90\n", encoding="utf-8")

    items = list_run_summaries(tmp_path)

    trust = items[0]["robustness_trust"]
    assert trust["status"] == "partial"
    assert trust["trust_signals"]["has_audit_refs"] is True
    assert trust["trust_signals"]["has_audit_digests"] is False
    assert "audit_digest_mismatch.robustness_conditions_csv" in trust["degraded_by"]


def test_compare_run_summaries_uses_robustness_contract_when_baseline_is_robustness(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "baseline",
                "robustness": {
                    "clean": {"results": {"auroc": 0.95}, "latency_ms_per_image": 1.0},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.90}, "latency_ms_per_image": 1.1}
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "candidate",
                "robustness": {
                    "clean": {"results": {"auroc": 0.94}, "latency_ms_per_image": 1.0},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.88}, "latency_ms_per_image": 1.2}
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, candidate],
        baseline_run_dir=baseline,
    )

    assert payload["baseline_run"]["robustness_protocol"]["corruption_count"] == 1
    assert payload["baseline_run"]["robustness_trust"]["status"] == "trust-signaled"
    assert payload["evaluation_contract"]["primary_metric"] == "worst_corruption_auroc"
    assert payload["evaluation_contract"]["ranking_metric"] == "worst_corruption_auroc"
    assert (
        payload["evaluation_contract"]["comparability_hints"]["requires_same_corruption_protocol"]
        is True
    )
    assert payload["metrics"]["worst_corruption_auroc"]["direction"] == "higher_is_better"
    assert payload["metrics"]["worst_corruption_auroc"]["comparisons"][1]["status"] == "regressed"


def test_compare_run_summaries_reports_robustness_protocol_compatibility(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    mismatched = tmp_path / "mismatched"
    missing = tmp_path / "missing"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    missing.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "baseline",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.95}},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.90}},
                            "severity_2": {"results": {"auroc": 0.85}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "matched",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.94}},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.89}},
                            "severity_2": {"results": {"auroc": 0.84}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "mismatched",
                "input_mode": "paths",
                "resize": [512, 512],
                "robustness": {
                    "corruption_mode": "clean_only",
                    "clean": {"results": {"auroc": 0.93}},
                    "corruptions": {
                        "jpeg": {
                            "severity_1": {"results": {"auroc": 0.88}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (missing / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "missing",
                "results": {"auroc": 0.90},
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, matched, mismatched, missing],
        baseline_run_dir=baseline,
    )

    comparison = payload["robustness_protocol_comparison"]
    assert comparison["baseline"]["corruption_mode"] == "full"
    assert comparison["baseline"]["input_mode"] == "numpy"
    assert comparison["baseline"]["resize"] == [256, 256]
    assert [row["status"] for row in comparison["comparisons"]] == [
        "baseline",
        "matched",
        "mismatched",
        "missing",
    ]
    assert comparison["comparisons"][2]["mismatch_fields"] == [
        "conditions",
        "corruption_mode",
        "input_mode",
        "resize",
        "severities",
    ]
    assert comparison["summary"]["checked"] is True
    assert comparison["summary"]["matched_runs"] == 1
    assert comparison["summary"]["mismatched_runs"] == 1
    assert comparison["summary"]["missing_runs"] == 1
    assert comparison["summary"]["incompatible_runs"] == 2


def test_compare_run_summaries_treats_drop_metrics_as_lower_is_better(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "baseline",
                "robustness_summary": {"worst_corruption_drop_auroc": 0.10},
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "candidate",
                "robustness_summary": {"worst_corruption_drop_auroc": 0.15},
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, candidate],
        baseline_run_dir=baseline,
        metric="worst_corruption_drop_auroc",
    )

    metric = payload["metrics"]["worst_corruption_drop_auroc"]
    assert metric["direction"] == "lower_is_better"
    assert metric["comparisons"][0]["status"] == "baseline"
    assert metric["comparisons"][1]["status"] == "regressed"
    assert metric["comparisons"][1]["delta_vs_baseline"] == 0.05


def test_compare_run_summaries_treats_latency_ratio_as_lower_is_better(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "baseline",
                "robustness_summary": {"worst_corruption_latency_ratio": 1.1},
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "candidate",
                "robustness_summary": {"worst_corruption_latency_ratio": 1.4},
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, candidate],
        baseline_run_dir=baseline,
        metric="worst_corruption_latency_ratio",
    )

    metric = payload["metrics"]["worst_corruption_latency_ratio"]
    assert metric["direction"] == "lower_is_better"
    assert metric["comparisons"][0]["status"] == "baseline"
    assert metric["comparisons"][1]["status"] == "regressed"
    assert metric["comparisons"][1]["delta_vs_baseline"] == 0.3


def test_compare_run_summaries_reports_split_compatibility_vs_baseline(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    same = tmp_path / "same"
    different = tmp_path / "different"
    missing = tmp_path / "missing"
    baseline.mkdir()
    same.mkdir()
    different.mkdir()
    missing.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "baseline",
                "results": {"auroc": 0.91},
                "split_fingerprint": {"sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    (same / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "same",
                "results": {"auroc": 0.92},
                "split_fingerprint": {"sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    (different / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "different",
                "results": {"auroc": 0.90},
                "split_fingerprint": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    (missing / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "missing", "results": {"auroc": 0.89}}),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, same, different, missing],
        baseline_run_dir=baseline,
    )

    split = payload["split_comparison"]
    assert split["baseline_split_fingerprint_sha256"] == "f" * 64
    assert [row["status"] for row in split["comparisons"]] == [
        "baseline",
        "matched",
        "mismatched",
        "missing",
    ]
    assert split["summary"]["checked"] is True
    assert split["summary"]["matched_runs"] == 1
    assert split["summary"]["mismatched_runs"] == 1
    assert split["summary"]["missing_runs"] == 1
    assert split["summary"]["incompatible_runs"] == 2


def test_compare_run_summaries_reports_target_compatibility_vs_baseline(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    wrong_category = tmp_path / "wrong_category"
    missing_category = tmp_path / "missing_category"
    wrong_dataset = tmp_path / "wrong_dataset"
    baseline.mkdir()
    matched.mkdir()
    wrong_category.mkdir()
    missing_category.mkdir()
    wrong_dataset.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "baseline",
                "results": {"auroc": 0.91},
            }
        ),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "matched",
                "results": {"auroc": 0.92},
            }
        ),
        encoding="utf-8",
    )
    (wrong_category / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "capsule",
                "model": "wrong_category",
                "results": {"auroc": 0.90},
            }
        ),
        encoding="utf-8",
    )
    (missing_category / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "missing_category",
                "results": {"auroc": 0.89},
            }
        ),
        encoding="utf-8",
    )
    (wrong_dataset / "report.json").write_text(
        json.dumps(
            {
                "dataset": "visa",
                "category": "bottle",
                "model": "wrong_dataset",
                "results": {"auroc": 0.88},
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, matched, wrong_category, missing_category, wrong_dataset],
        baseline_run_dir=baseline,
    )

    target = payload["target_comparison"]
    assert target["baseline"]["dataset"] == "mvtec"
    assert target["baseline"]["category"] == "bottle"
    assert [row["status"] for row in target["comparisons"]] == [
        "baseline",
        "matched",
        "mismatched",
        "missing",
        "mismatched",
    ]
    assert target["comparisons"][1]["dataset_status"] == "matched"
    assert target["comparisons"][1]["category_status"] == "matched"
    assert target["comparisons"][2]["category_status"] == "mismatched"
    assert target["comparisons"][3]["category_status"] == "missing"
    assert target["comparisons"][4]["dataset_status"] == "mismatched"
    assert target["summary"]["checked"] is True
    assert target["summary"]["dataset_checked"] is True
    assert target["summary"]["category_checked"] is True
    assert target["summary"]["matched_runs"] == 1
    assert target["summary"]["mismatched_runs"] == 2
    assert target["summary"]["missing_runs"] == 1
    assert target["summary"]["incompatible_runs"] == 3


def test_compare_run_summaries_emits_machine_readable_verdict_summary(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "baseline",
                "results": {"auroc": 0.91},
                "split_fingerprint": {"sha256": "f" * 64},
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "visa",
                "category": "capsule",
                "model": "candidate",
                "results": {"auroc": 0.90},
                "split_fingerprint": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (candidate / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "1" * 64}),
        encoding="utf-8",
    )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)

    summary = payload["summary"]
    assert summary["total_regressions"] == 1
    assert summary["regression_gate"] == "regressed"
    assert summary["comparability_gates"] == {
        "split": "incompatible",
        "environment": "incompatible",
        "target": "incompatible",
        "robustness_protocol": "unchecked",
        "operator_contract": "unchecked",
        "bundle_operator_contract": "unchecked",
    }
    assert summary["blocking_flags"] == [
        "--fail-on-regression",
        "--require-same-split",
        "--require-same-environment",
        "--require-same-target",
    ]
    assert summary["verdict"] == "blocked"


def test_compare_run_summaries_emits_machine_readable_metric_and_trust_summary(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    for run_dir, model_name, metric_value in (
        (baseline, "baseline", 0.91),
        (candidate, "candidate", 0.92),
    ):
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    "dataset": "mvtec",
                    "category": "bottle",
                    "model": model_name,
                    "results": {"auroc": metric_value},
                    "split_fingerprint": {"sha256": "f" * 64},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )
    (baseline / "artifacts").mkdir()
    (baseline / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "split_fingerprint": {"sha256": "f" * 64},
                "prediction": {"reject_confidence_below": 0.75, "reject_label": -9},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)

    summary = payload["summary"]
    assert summary["primary_metric"] == "auroc"
    assert summary["primary_metric_direction"] == "higher_is_better"
    assert summary["primary_metric_baseline"] == 0.91
    assert summary["primary_metric_total_regressions"] == 0
    assert summary["primary_metric_statuses"] == {"candidate": "improved"}
    assert summary["primary_metric_deltas"] == {"candidate": 0.01}
    assert summary["trust_checked"] is True
    assert summary["trust_gate"] == "limited"
    assert summary["trust_status"] == "partial"
    assert summary["trust_reason"] == "missing_split_fingerprint"
    assert summary["operator_contract_status"] == "missing"
    assert summary["operator_contract_consistent"] is False


def test_compare_run_summaries_emits_informational_summary_without_baseline(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "a", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "b", "results": {"auroc": 0.93}}),
        encoding="utf-8",
    )

    payload = compare_run_summaries([run_a, run_b])

    summary = payload["summary"]
    assert summary["baseline_checked"] is False
    assert summary["regression_gate"] == "unchecked"
    assert summary["blocking_flags"] == []
    assert summary["verdict"] == "informational"
    assert summary["trust_checked"] is False
    assert summary["candidate_verdicts"] == {}
    assert summary["candidate_blocking_reasons"] == {}
    assert summary["candidate_comparability_gates"] == {}
    assert summary["candidate_bundle_operator_contract_digest_statuses"] == {}
    assert summary["candidate_incompatibility_digest"] == {}


def test_compare_run_summaries_emits_candidate_verdicts_and_blocking_reasons(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    clean = tmp_path / "clean"
    blocked = tmp_path / "blocked"
    baseline.mkdir()
    clean.mkdir()
    blocked.mkdir()
    shared_report = {
        "dataset": "mvtec",
        "category": "bottle",
        "split_fingerprint": {"sha256": "f" * 64},
    }
    (baseline / "report.json").write_text(
        json.dumps({**shared_report, "model": "baseline", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (clean / "report.json").write_text(
        json.dumps({**shared_report, "model": "clean", "results": {"auroc": 0.92}}),
        encoding="utf-8",
    )
    (blocked / "report.json").write_text(
        json.dumps(
            {
                "dataset": "visa",
                "category": "capsule",
                "model": "blocked",
                "results": {"auroc": 0.90},
                "split_fingerprint": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (clean / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (blocked / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "1" * 64}),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, clean, blocked],
        baseline_run_dir=baseline,
    )

    summary = payload["summary"]
    assert summary["candidate_verdicts"] == {
        "clean": "pass",
        "blocked": "blocked",
    }
    assert summary["candidate_blocking_reasons"]["clean"] == []
    assert summary["candidate_blocking_reasons"]["blocked"] == [
        "primary_metric:regressed",
        "split:mismatched",
        "environment:mismatched",
        "target.dataset:mismatched",
        "target.category:mismatched",
    ]
    assert summary["candidate_incompatibility_digest"] == {
        "blocked": {
            "verdict": "blocked",
            "incompatible_gates": [
                "split:mismatched",
                "environment:mismatched",
                "target:mismatched",
                "target_dataset:mismatched",
                "target_category:mismatched",
            ],
            "blocking_reasons": [
                "primary_metric:regressed",
                "split:mismatched",
                "environment:mismatched",
                "target.dataset:mismatched",
                "target.category:mismatched",
            ],
        },
        "clean": {
            "verdict": "pass",
            "incompatible_gates": [],
            "blocking_reasons": [],
        },
    }
    assert summary["candidate_comparability_gates"] == {
        "clean": {
            "split": "matched",
            "environment": "matched",
            "target": "matched",
            "target_dataset": "matched",
            "target_category": "matched",
            "robustness_protocol": "unchecked",
            "operator_contract": "unchecked",
            "bundle_operator_contract": "unchecked",
        },
        "blocked": {
            "split": "mismatched",
            "environment": "mismatched",
            "target": "mismatched",
            "target_dataset": "mismatched",
            "target_category": "mismatched",
            "robustness_protocol": "unchecked",
            "operator_contract": "unchecked",
            "bundle_operator_contract": "unchecked",
        },
    }


def test_compare_run_summaries_blocks_candidate_missing_operator_contract_when_baseline_consistent(
    tmp_path,
):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    shared_report = {
        "dataset": "mvtec",
        "category": "bottle",
        "split_fingerprint": {"sha256": "f" * 64},
    }
    (baseline / "report.json").write_text(
        json.dumps({**shared_report, "model": "baseline", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps({**shared_report, "model": "candidate", "results": {"auroc": 0.92}}),
        encoding="utf-8",
    )
    (baseline / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (candidate / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (candidate / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (baseline / "artifacts").mkdir()
    operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    (baseline / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "split_fingerprint": {"sha256": "f" * 64},
                "operator_contract": operator_contract,
            }
        ),
        encoding="utf-8",
    )
    (baseline / "artifacts" / "operator_contract.json").write_text(
        json.dumps(operator_contract),
        encoding="utf-8",
    )
    (baseline / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_fingerprint": {"sha256": "f" * 64},
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)
    summary = payload["summary"]

    assert summary["candidate_verdicts"]["candidate"] == "blocked"
    assert summary["candidate_blocking_reasons"]["candidate"] == ["operator_contract:missing"]
    assert len(str(summary["operator_contract_baseline_sha256"])) == 64
    assert summary["candidate_incompatibility_digest"]["candidate"] == {
        "verdict": "blocked",
        "incompatible_gates": ["operator_contract:missing"],
        "blocking_reasons": ["operator_contract:missing"],
    }
    assert summary["operator_contract_gate"] == "incompatible"
    assert summary["comparability_gates"]["operator_contract"] == "incompatible"
    assert "--require-same-operator-contract" in summary["blocking_flags"]
    operator_cmp = payload["operator_contract_comparison"]
    assert operator_cmp["summary"]["incompatible_runs"] == 1
    assert operator_cmp["comparisons"][1]["status"] == "missing"
    assert len(str(operator_cmp["baseline_contract_sha256"])) == 64
    assert len(str(operator_cmp["comparisons"][0]["contract_sha256"])) == 64
    assert operator_cmp["comparisons"][1]["contract_sha256"] is None


def test_compare_run_summaries_blocks_candidate_missing_bundle_operator_contract_when_baseline_consistent(
    tmp_path,
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    shared_report = {
        "dataset": "mvtec",
        "category": "bottle",
        "split_fingerprint": {"sha256": "f" * 64},
    }
    (baseline / "report.json").write_text(
        json.dumps({**shared_report, "model": "baseline", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps({**shared_report, "model": "candidate", "results": {"auroc": 0.92}}),
        encoding="utf-8",
    )
    (baseline / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (candidate / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (candidate / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (baseline / "artifacts").mkdir()
    (candidate / "artifacts").mkdir()
    operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    for run_dir in (baseline, candidate):
        (run_dir / "artifacts" / "infer_config.json").write_text(
            json.dumps(
                {
                    "threshold": 0.5,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "operator_contract": operator_contract,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "operator_contract.json").write_text(
            json.dumps(operator_contract),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "calibration_card.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "threshold_context": {"scope": "image", "category_count": 1},
                    "image_threshold": {
                        "threshold": 0.5,
                        "provenance": {"method": "fixed", "source": "test"},
                    },
                }
            ),
            encoding="utf-8",
        )

    baseline_bundle = baseline / "deploy_bundle"
    baseline_bundle.mkdir()
    (baseline_bundle / "infer_config.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "operator_contract": operator_contract,
                "artifact_quality": {
                    "has_operator_contract": True,
                    "audit_refs": {"operator_contract": "operator_contract.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (baseline_bundle / "operator_contract.json").write_text(
        json.dumps(operator_contract),
        encoding="utf-8",
    )
    (baseline_bundle / "report.json").write_text(
        json.dumps({"dataset": "mvtec"}),
        encoding="utf-8",
    )
    (baseline_bundle / "config.json").write_text(
        json.dumps({"config": {}}),
        encoding="utf-8",
    )
    (baseline_bundle / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (baseline_bundle / "calibration_card.json").write_text(
        json.dumps({"schema_version": 1}),
        encoding="utf-8",
    )
    (baseline_bundle / "bundle_manifest.json").write_text(
        json.dumps(
            build_deploy_bundle_manifest(
                bundle_dir=baseline_bundle,
                source_run_dir=baseline,
            )
        ),
        encoding="utf-8",
    )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)
    summary = payload["summary"]

    assert summary["candidate_verdicts"]["candidate"] == "blocked"
    assert summary["candidate_blocking_reasons"]["candidate"] == [
        "operator_contract_bundle:missing"
    ]
    assert len(str(summary["bundle_operator_contract_baseline_sha256"])) == 64
    assert summary["candidate_incompatibility_digest"]["candidate"] == {
        "verdict": "blocked",
        "incompatible_gates": ["bundle_operator_contract:missing"],
        "blocking_reasons": ["operator_contract_bundle:missing"],
    }
    assert summary["candidate_bundle_operator_contract_digest_statuses"]["candidate"] == "missing"
    assert summary["bundle_operator_contract_digests_valid"] is True
    assert summary["bundle_operator_contract_gate"] == "incompatible"
    assert summary["comparability_gates"]["bundle_operator_contract"] == "incompatible"
    assert "--require-same-bundle-operator-contract" in summary["blocking_flags"]
    assert payload["bundle_operator_contract_comparison"]["summary"]["incompatible_runs"] == 1
    assert payload["bundle_operator_contract_comparison"]["comparisons"][1]["status"] == "missing"


def test_compare_run_summaries_blocks_candidate_bundle_operator_contract_baseline_mismatch(
    tmp_path,
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    shared_report = {
        "dataset": "mvtec",
        "category": "bottle",
        "split_fingerprint": {"sha256": "f" * 64},
    }
    (baseline / "report.json").write_text(
        json.dumps({**shared_report, "model": "baseline", "results": {"auroc": 0.91}}),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps({**shared_report, "model": "candidate", "results": {"auroc": 0.92}}),
        encoding="utf-8",
    )
    (baseline / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (candidate / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (candidate / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "e" * 64}),
        encoding="utf-8",
    )
    (baseline / "artifacts").mkdir()
    (candidate / "artifacts").mkdir()
    run_operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    for run_dir in (baseline, candidate):
        (run_dir / "artifacts" / "infer_config.json").write_text(
            json.dumps(
                {
                    "threshold": 0.5,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "operator_contract": run_operator_contract,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "operator_contract.json").write_text(
            json.dumps(run_operator_contract),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "calibration_card.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "threshold_context": {"scope": "image", "category_count": 1},
                    "image_threshold": {
                        "threshold": 0.5,
                        "provenance": {"method": "fixed", "source": "test"},
                    },
                }
            ),
            encoding="utf-8",
        )

    baseline_bundle_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.8,
            "reject_label": -9,
        },
    }
    candidate_bundle_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.65,
            "reject_label": -9,
        },
    }
    for run_dir, bundle_contract in (
        (baseline, baseline_bundle_contract),
        (candidate, candidate_bundle_contract),
    ):
        bundle_dir = run_dir / "deploy_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "infer_config.json").write_text(
            json.dumps(
                {
                    "threshold": 0.5,
                    "operator_contract": bundle_contract,
                    "artifact_quality": {
                        "has_operator_contract": True,
                        "audit_refs": {"operator_contract": "operator_contract.json"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "operator_contract.json").write_text(
            json.dumps(bundle_contract),
            encoding="utf-8",
        )
        (bundle_dir / "report.json").write_text(
            json.dumps({"dataset": "mvtec"}),
            encoding="utf-8",
        )
        (bundle_dir / "config.json").write_text(
            json.dumps({"config": {}}),
            encoding="utf-8",
        )
        (bundle_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )
        (bundle_dir / "calibration_card.json").write_text(
            json.dumps({"schema_version": 1}),
            encoding="utf-8",
        )
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(
                build_deploy_bundle_manifest(
                    bundle_dir=bundle_dir,
                    source_run_dir=run_dir,
                )
            ),
            encoding="utf-8",
        )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)
    summary = payload["summary"]

    assert summary["candidate_verdicts"]["candidate"] == "blocked"
    assert summary["candidate_blocking_reasons"]["candidate"] == [
        "operator_contract_bundle:baseline_mismatch"
    ]
    assert summary["candidate_bundle_operator_contract_digest_statuses"]["candidate"] == "valid"
    assert summary["bundle_operator_contract_gate"] == "incompatible"
    bundle_cmp = payload["bundle_operator_contract_comparison"]
    assert bundle_cmp["summary"]["incompatible_runs"] == 1
    assert bundle_cmp["comparisons"][1]["status"] == "mismatched"
    assert len(str(bundle_cmp["baseline_contract_sha256"])) == 64
    assert len(str(bundle_cmp["comparisons"][1]["contract_sha256"])) == 64
    assert bundle_cmp["comparisons"][1]["contract_sha256"] != bundle_cmp["baseline_contract_sha256"]


def test_compare_run_summaries_flags_candidate_bundle_operator_contract_digest_mismatch(
    tmp_path,
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    shared_report = {
        "dataset": "mvtec",
        "category": "bottle",
        "split_fingerprint": {"sha256": "f" * 64},
    }
    for run_dir, model_name, metric_value in (
        (baseline, "baseline", 0.91),
        (candidate, "candidate", 0.92),
    ):
        (run_dir / "report.json").write_text(
            json.dumps(
                {
                    **shared_report,
                    "model": model_name,
                    "results": {"auroc": metric_value},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )

    run_operator_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    for run_dir in (baseline, candidate):
        (run_dir / "artifacts").mkdir()
        (run_dir / "artifacts" / "infer_config.json").write_text(
            json.dumps(
                {
                    "threshold": 0.5,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "operator_contract": run_operator_contract,
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "operator_contract.json").write_text(
            json.dumps(run_operator_contract),
            encoding="utf-8",
        )
        (run_dir / "artifacts" / "calibration_card.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "split_fingerprint": {"sha256": "f" * 64},
                    "threshold_context": {"scope": "image", "category_count": 1},
                    "image_threshold": {
                        "threshold": 0.5,
                        "provenance": {"method": "fixed", "source": "test"},
                    },
                }
            ),
            encoding="utf-8",
        )

    bundle_contract = {
        "schema_version": 1,
        "review_policy": {
            "review_on": ["anomalous", "rejected_low_confidence"],
            "confidence_gate_enabled": True,
            "reject_confidence_below": 0.75,
            "reject_label": -9,
        },
    }
    for run_dir in (baseline, candidate):
        bundle_dir = run_dir / "deploy_bundle"
        bundle_dir.mkdir()
        (bundle_dir / "infer_config.json").write_text(
            json.dumps(
                {
                    "threshold": 0.5,
                    "operator_contract": bundle_contract,
                    "artifact_quality": {
                        "has_operator_contract": True,
                        "audit_refs": {"operator_contract": "operator_contract.json"},
                    },
                }
            ),
            encoding="utf-8",
        )
        (bundle_dir / "operator_contract.json").write_text(
            json.dumps(bundle_contract),
            encoding="utf-8",
        )
        (bundle_dir / "report.json").write_text(
            json.dumps({"dataset": "mvtec"}),
            encoding="utf-8",
        )
        (bundle_dir / "config.json").write_text(
            json.dumps({"config": {}}),
            encoding="utf-8",
        )
        (bundle_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )
        (bundle_dir / "calibration_card.json").write_text(
            json.dumps({"schema_version": 1}),
            encoding="utf-8",
        )
        manifest = build_deploy_bundle_manifest(
            bundle_dir=bundle_dir,
            source_run_dir=run_dir,
        )
        if run_dir == candidate:
            manifest["operator_contract_digests"]["bundle_operator_contract_sha256"] = "0" * 64
        (bundle_dir / "bundle_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

    payload = compare_run_summaries([baseline, candidate], baseline_run_dir=baseline)
    summary = payload["summary"]

    assert summary["candidate_verdicts"]["candidate"] == "blocked"
    assert (
        "operator_contract_bundle:mismatched" in summary["candidate_blocking_reasons"]["candidate"]
    )
    assert (
        "operator_contract_bundle:digest_mismatch"
        in summary["candidate_blocking_reasons"]["candidate"]
    )
    assert summary["candidate_incompatibility_digest"]["candidate"]["incompatible_gates"] == [
        "bundle_operator_contract:mismatched"
    ]
    assert summary["candidate_bundle_operator_contract_digest_statuses"]["candidate"] == "invalid"
    assert summary["bundle_operator_contract_gate"] == "incompatible"
    assert payload["bundle_operator_contract_comparison"]["summary"]["incompatible_runs"] == 1
    assert (
        payload["bundle_operator_contract_comparison"]["comparisons"][1]["status"] == "mismatched"
    )


def test_compare_run_summaries_reports_environment_compatibility_vs_baseline(tmp_path):
    from pyimgano.reporting.run_index import compare_run_summaries

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    mismatched = tmp_path / "mismatched"
    missing = tmp_path / "missing"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    missing.mkdir()
    (baseline / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "baseline"}),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "matched"}),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "mismatched"}),
        encoding="utf-8",
    )
    (missing / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "missing"}),
        encoding="utf-8",
    )
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (matched / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (mismatched / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "0" * 64}),
        encoding="utf-8",
    )

    payload = compare_run_summaries(
        [baseline, matched, mismatched, missing],
        baseline_run_dir=baseline,
    )

    environment = payload["environment_comparison"]
    assert environment["baseline_environment_fingerprint_sha256"] == "f" * 64
    assert [row["status"] for row in environment["comparisons"]] == [
        "baseline",
        "matched",
        "mismatched",
        "missing",
    ]
    assert environment["summary"]["checked"] is True
    assert environment["summary"]["matched_runs"] == 1
    assert environment["summary"]["mismatched_runs"] == 1
    assert environment["summary"]["missing_runs"] == 1
    assert environment["summary"]["incompatible_runs"] == 2
