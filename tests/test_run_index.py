import json


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
        json.dumps({"suite": "industrial-v4", "dataset": "mvtec", "timestamp_utc": "2026-03-17T10:00:00+00:00"}),
        encoding="utf-8",
    )
    (bench_dir / "report.json").write_text(
        json.dumps({"model": "vision_patchcore", "dataset": "visa", "timestamp_utc": "2026-03-17T11:00:00+00:00"}),
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
    assert payload["metrics"]["auroc"]["max"] == 0.92
    assert payload["metrics"]["auroc"]["min"] == 0.9
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
    assert metric["baseline"] == 0.91
    assert metric["comparisons"][0]["status"] == "baseline"
    assert metric["comparisons"][1]["status"] == "regressed"
    assert metric["comparisons"][1]["delta_vs_baseline"] == -0.02


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
        json.dumps({"dataset": "custom", "model": "vision_ecod"}),
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
        json.dumps({"dataset": "custom", "model": "partial", "timestamp_utc": "2026-03-18T10:00:00+00:00"}),
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

    assert [item["run_dir_name"] for item in items] == [reproducible_new.name, reproducible_old.name]


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
    assert items[0]["metrics"]["clean_auroc"] == 0.95
    assert items[0]["metrics"]["mean_corruption_auroc"] == 0.85
    assert items[0]["metrics"]["worst_corruption_auroc"] == 0.8
    assert items[0]["metrics"]["mean_corruption_drop_auroc"] == 0.1
    assert items[0]["metrics"]["worst_corruption_drop_auroc"] == 0.15
    assert items[0]["metrics"]["clean_latency_ms_per_image"] == 1.0
    assert items[0]["metrics"]["mean_corruption_latency_ms_per_image"] == 1.2
    assert items[0]["metrics"]["worst_corruption_latency_ms_per_image"] == 1.3
    assert items[0]["metrics"]["mean_corruption_latency_ratio"] == 1.2
    assert items[0]["metrics"]["worst_corruption_latency_ratio"] == 1.3
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
    assert summary["candidate_comparability_gates"] == {
        "clean": {
            "split": "matched",
            "environment": "matched",
            "target": "matched",
            "target_dataset": "matched",
            "target_category": "matched",
            "robustness_protocol": "unchecked",
        },
        "blocked": {
            "split": "mismatched",
            "environment": "mismatched",
            "target": "mismatched",
            "target_dataset": "mismatched",
            "target_category": "mismatched",
            "robustness_protocol": "unchecked",
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
