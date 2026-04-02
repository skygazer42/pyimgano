import hashlib
import json
from types import MappingProxyType


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runs_cli_list_json(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "vision_patchcore",
                "results": {"auroc": 0.95},
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

    rc = main(["list", "--root", str(tmp_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["runs"][0]["model_or_suite"] == "vision_patchcore"
    assert out["runs"][0]["dataset_readiness"] == {
        "status": "warning",
        "issue_codes": ["FEWSHOT_TRAIN_SET"],
        "issue_details": [],
    }


def test_runs_cli_compare_json(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["compare", str(run_a), str(run_b), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metrics"]["auroc"]["max"] == 0.93


def test_runs_cli_compare_json_is_informational_without_baseline(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["compare", str(run_a), str(run_b), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["baseline_checked"] is False
    assert out["summary"]["regression_gate"] == "unchecked"
    assert out["summary"]["blocking_flags"] == []
    assert out["summary"]["verdict"] == "informational"
    assert out["summary"]["trust_checked"] is False


def test_runs_cli_compare_json_emits_candidate_verdicts_and_blocking_reasons(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(clean),
            str(blocked),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["candidate_verdicts"] == {
        "clean": "pass",
        "blocked": "blocked",
    }
    assert out["summary"]["candidate_blocking_reasons"]["clean"] == []
    assert out["summary"]["candidate_blocking_reasons"]["blocked"] == [
        "primary_metric:regressed",
        "split:mismatched",
        "environment:mismatched",
        "target.dataset:mismatched",
        "target.category:mismatched",
    ]
    assert out["summary"]["candidate_incompatibility_digest"] == {
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
    assert out["summary"]["candidate_comparability_gates"] == {
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


def test_runs_cli_list_supports_kind_and_dataset_filters(tmp_path, capsys):
    from pyimgano.runs_cli import main

    suite_dir = tmp_path / "suite_run"
    bench_dir = tmp_path / "bench_run"
    suite_dir.mkdir()
    bench_dir.mkdir()
    (suite_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4", "dataset": "mvtec"}),
        encoding="utf-8",
    )
    (bench_dir / "report.json").write_text(
        json.dumps({"model": "vision_patchcore", "dataset": "visa"}),
        encoding="utf-8",
    )

    rc = main(["list", "--root", str(tmp_path), "--kind", "suite", "--dataset", "mvtec", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["runs"]) == 1
    assert out["runs"][0]["kind"] == "suite"


def test_runs_cli_list_plain_output_prints_quality_and_trust(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_patchcore",
                "results": {"auroc": 0.95},
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

    rc = main(["list", "--root", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "quality=reproducible" in out
    assert "trust=partial" in out
    assert "operator_contract=missing" in out
    assert "bundle_operator_contract=missing" in out
    assert "primary_metric=auroc:0.95" in out
    assert "dataset_readiness=warning" in out
    assert "reason=core_artifacts_present" in out


def test_runs_cli_compare_can_fail_on_regression(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "a", "results": {"auroc": 0.93}}),
        encoding="utf-8",
    )
    (run_b / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "model": "b", "results": {"auroc": 0.90}}),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(run_a),
            str(run_b),
            "--baseline",
            str(run_a),
            "--metric",
            "auroc",
            "--fail-on-regression",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["summary"]["total_regressions"] == 1
    assert out["metrics"]["auroc"]["comparisons"][1]["status"] == "regressed"


def test_runs_cli_latest_json(tmp_path, capsys):
    from pyimgano.runs_cli import main

    older = tmp_path / "20260317_090000_old"
    newer = tmp_path / "20260317_100000_new"
    older.mkdir()
    newer.mkdir()
    (older / "report.json").write_text(
        json.dumps(
            {"dataset": "mvtec", "model": "old", "timestamp_utc": "2026-03-17T09:00:00+00:00"}
        ),
        encoding="utf-8",
    )
    (newer / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "new",
                "timestamp_utc": "2026-03-17T10:00:00+00:00",
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                },
            }
        ),
        encoding="utf-8",
    )
    (newer / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (newer / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    rc = main(["latest", "--root", str(tmp_path), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["run_dir_name"] == newer.name
    assert out["run"]["dataset_readiness"] == {
        "status": "warning",
        "issue_codes": ["FEWSHOT_TRAIN_SET"],
        "issue_details": [],
    }


def test_runs_cli_latest_plain_output_prints_quality_and_trust(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_dir = tmp_path / "20260317_100000_new"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_patchcore",
                "timestamp_utc": "2026-03-17T10:00:00+00:00",
                "results": {"auroc": 0.95},
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

    rc = main(["latest", "--root", str(tmp_path)])

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "quality=reproducible" in out
    assert "trust=partial" in out
    assert "operator_contract=missing" in out
    assert "bundle_operator_contract=missing" in out
    assert "primary_metric=auroc:0.95" in out
    assert "dataset_readiness=warning" in out
    assert "reason=core_artifacts_present" in out


def test_runs_cli_latest_can_filter_by_same_robustness_protocol(tmp_path, capsys):
    from pyimgano.runs_cli import main

    baseline = tmp_path / "20260318_100000_baseline"
    matched = tmp_path / "20260318_101000_matched"
    mismatched = tmp_path / "20260318_102000_mismatched"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.95}},
                    "corruptions": {"lighting": {"severity_1": {"results": {"auroc": 0.90}}}},
                },
            }
        ),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T10:10:00+00:00",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.94}},
                    "corruptions": {"lighting": {"severity_1": {"results": {"auroc": 0.89}}}},
                },
            }
        ),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T10:20:00+00:00",
                "input_mode": "paths",
                "resize": [512, 512],
                "robustness": {
                    "corruption_mode": "clean_only",
                    "clean": {"results": {"auroc": 0.93}},
                    "corruptions": {},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "latest",
            "--root",
            str(tmp_path),
            "--same-robustness-protocol-as",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["run_dir_name"] == matched.name


def test_runs_cli_list_can_filter_by_same_split(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["list", "--root", str(tmp_path), "--same-split-as", str(run_a), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert [item["run_dir_name"] for item in out["runs"]] == [run_b.name, run_a.name]


def test_runs_cli_list_can_filter_by_same_environment(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["list", "--root", str(tmp_path), "--same-environment-as", str(baseline), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert [item["run_dir_name"] for item in out["runs"]] == [matched.name, baseline.name]


def test_runs_cli_list_can_filter_by_same_robustness_protocol(tmp_path, capsys):
    from pyimgano.runs_cli import main

    baseline = tmp_path / "baseline"
    matched = tmp_path / "matched"
    mismatched = tmp_path / "mismatched"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    (baseline / "report.json").write_text(
        json.dumps(
            {
                "model": "baseline",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.95}},
                    "corruptions": {"lighting": {"severity_1": {"results": {"auroc": 0.90}}}},
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
                    "corruptions": {"lighting": {"severity_1": {"results": {"auroc": 0.89}}}},
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
                    "corruptions": {},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "list",
            "--root",
            str(tmp_path),
            "--same-robustness-protocol-as",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert [item["run_dir_name"] for item in out["runs"]] == [matched.name, baseline.name]


def test_runs_cli_latest_can_filter_by_same_target(tmp_path, capsys):
    from pyimgano.runs_cli import main

    baseline = tmp_path / "20260318_100000_baseline"
    matched = tmp_path / "20260318_101000_matched"
    mismatched = tmp_path / "20260318_102000_mismatched"
    baseline.mkdir()
    matched.mkdir()
    mismatched.mkdir()
    (baseline / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "baseline"}),
        encoding="utf-8",
    )
    (matched / "report.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T10:10:00+00:00",
                "dataset": "mvtec",
                "category": "bottle",
                "model": "matched",
            }
        ),
        encoding="utf-8",
    )
    (mismatched / "report.json").write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-18T10:20:00+00:00",
                "dataset": "visa",
                "category": "capsule",
                "model": "mismatched",
            }
        ),
        encoding="utf-8",
    )

    rc = main(["latest", "--root", str(tmp_path), "--same-target-as", str(baseline), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["run_dir_name"] == matched.name


def test_runs_cli_latest_can_filter_by_min_quality(tmp_path, capsys):
    from pyimgano.runs_cli import main

    partial = tmp_path / "20260318_100000_partial"
    reproducible = tmp_path / "20260318_101000_reproducible"
    partial.mkdir()
    reproducible.mkdir()
    (partial / "report.json").write_text(
        json.dumps(
            {"dataset": "custom", "model": "partial", "timestamp_utc": "2026-03-18T10:20:00+00:00"}
        ),
        encoding="utf-8",
    )
    (reproducible / "report.json").write_text(
        json.dumps(
            {
                "dataset": "custom",
                "model": "reproducible",
                "timestamp_utc": "2026-03-18T10:10:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (reproducible / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (reproducible / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    rc = main(["latest", "--root", str(tmp_path), "--min-quality", "reproducible", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run"]["run_dir_name"] == reproducible.name


def test_runs_cli_quality_json(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["quality", str(run_dir), "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["quality"]["status"] == "reproducible"
    assert out["quality"]["missing_required"] == []
    assert out["quality"]["handoff_report_status"] == "not_applicable"
    assert (
        out["quality"]["next_action"]
        == f"Export infer-config and deploy bundle artifacts, then rerun pyimgano runs quality {run_dir} --json"
    )


def test_runs_cli_quality_returns_nonzero_for_partial_run(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps({"dataset": "custom", "model": "vision_ecod"}),
        encoding="utf-8",
    )

    rc = main(["quality", str(run_dir), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["quality"]["status"] == "partial"
    assert out["quality"]["missing_required"] == ["config.json", "environment.json"]


def test_runs_cli_quality_can_require_audited_status(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps({"threshold": 0.5}),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
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

    rc = main(["quality", str(run_dir), "--require-status", "audited", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["quality"]["status"] == "audited"


def test_runs_cli_quality_fails_when_required_status_not_met(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(["quality", str(run_dir), "--require-status", "audited", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["quality"]["status"] == "reproducible"


def test_runs_cli_quality_fails_deployable_gate_when_bundle_weight_audit_is_invalid(
    tmp_path, capsys
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps({"threshold": 0.5}),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
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
    bundle_dir = run_dir / "deploy_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "infer_config.json").write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
    (bundle_dir / "report.json").write_text(json.dumps({"dataset": "custom"}), encoding="utf-8")
    (bundle_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (bundle_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (bundle_dir / "calibration_card.json").write_text(
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
    (bundle_dir / "model_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "broken_model",
                "summary": {
                    "purpose": "demo",
                    "intended_inputs": "RGB",
                    "output_contract": "image-level",
                },
                "weights": {
                    "path": "missing_model.pt",
                    "source": "unit-test",
                    "license": "internal",
                },
                "deployment": {"runtime": "torch"},
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)),
        encoding="utf-8",
    )

    rc = main(["quality", str(run_dir), "--require-status", "deployable", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["quality"]["status"] == "audited"
    assert out["quality"]["weights_audit"]["valid"] is False
    assert out["quality"]["weights_audit"]["model_card"]["valid"] is False


def test_runs_cli_quality_plain_output_prints_warnings(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "prediction": {"reject_confidence_below": 0.75, "reject_label": -9},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
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

    rc = main(["quality", str(run_dir)])

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "status=audited" in out
    assert "threshold_context" in out
    assert "prediction_policy" in out
    assert "reason=calibration_audit_consistent" in out
    assert "degraded_by=missing_threshold_context" in out


def test_runs_cli_quality_plain_output_prints_trust_signals(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps({"threshold": 0.5, "split_fingerprint": {"sha256": "f" * 64}}),
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

    rc = main(["quality", str(run_dir)])

    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert "trust_signal.has_threshold_context=true" in out
    assert "trust_signal.has_split_fingerprint=true" in out
    assert "handoff_report=not_applicable" in out
    assert "next_action=" in out
    assert "dataset_readiness_status=warning" in out
    assert "dataset_issue_codes=fewshot_train_set" in out
    assert "ref=calibration_card_json:artifacts/calibration_card.json" in out


def test_runs_cli_acceptance_json_ready_for_audited_run(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "threshold": 0.5,
                "artifact_quality": {
                    "status": "audited",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "has_operator_contract": False,
                    "audit_refs": {"calibration_card": "artifacts/calibration_card.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(run_dir), "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    acceptance = out["acceptance"]
    assert acceptance["status"] == "ready"
    assert acceptance["ready"] is True
    assert acceptance["required_quality"] == "audited"
    assert acceptance["quality"]["status"] == "audited"
    assert acceptance["infer_config"]["valid"] is True
    assert acceptance["infer_config"]["selected_source"] == "artifacts"
    assert acceptance["bundle_weights"]["applicable"] is False
    assert acceptance["blocking_reasons"] == []
    assert acceptance["handoff_report_status"] == "not_applicable"
    assert (
        acceptance["next_action"]
        == f"pyimgano-infer --from-run {run_dir} --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl"
    )
    assert acceptance["quality"]["dataset_readiness"] is None


def test_runs_cli_acceptance_plain_output_prints_run_dataset_readiness(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                    "issue_details": [],
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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "threshold": 0.5,
                "artifact_quality": {
                    "status": "audited",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "has_operator_contract": False,
                    "audit_refs": {"calibration_card": "artifacts/calibration_card.json"},
                },
            }
        ),
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

    rc = main(["acceptance", str(run_dir)])
    out = capsys.readouterr().out.lower()

    assert rc == 0
    assert "kind=run" in out
    assert "handoff_report=not_applicable" in out
    assert "next_action=" in out
    assert "dataset_readiness=warning" in out
    assert "dataset_readiness_status=warning" in out
    assert "dataset_issue_codes=fewshot_train_set" in out


def test_runs_cli_acceptance_json_includes_postprocess_contract_trust_signals(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "postprocess": {
                    "schema_version": 1,
                    "threshold_scope": "image",
                    "image_threshold": {
                        "threshold": 0.5,
                        "score_order": "higher_is_more_anomalous",
                    },
                    "review_policy": {
                        "review_on": ["anomalous", "rejected_low_confidence"],
                        "confidence_gate_enabled": True,
                        "reject_confidence_below": 0.75,
                        "reject_label": -9,
                    },
                    "label_encoding": {
                        "normal": 0,
                        "anomalous": 1,
                        "rejected": -9,
                    },
                },
                "artifact_quality": {
                    "status": "audited",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": True,
                    "has_prediction_policy": False,
                    "has_postprocess_contract": True,
                    "audit_refs": {"calibration_card": "artifacts/calibration_card.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(run_dir), "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    acceptance = out["acceptance"]
    trust = acceptance["infer_config"]["trust_summary"]
    assert acceptance["ready"] is True
    assert trust["trust_signals"]["has_postprocess_contract"] is True
    assert trust["trust_signals"]["has_postprocess_threshold"] is True
    assert trust["trust_signals"]["has_postprocess_review_policy"] is True
    assert trust["trust_signals"]["has_postprocess_label_encoding"] is True


def test_runs_cli_acceptance_uses_deploy_bundle_infer_config_when_present(tmp_path, capsys):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps({"schema_version": 1, "model": {"name": "vision_ecod", "model_kwargs": {}}}),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = run_dir / "deploy_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "threshold": 0.5,
                "artifact_quality": {
                    "status": "deployable",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": False,
                    "has_prediction_policy": False,
                    "has_operator_contract": False,
                    "has_deploy_bundle": True,
                    "has_bundle_manifest": True,
                    "required_bundle_artifacts_present": True,
                    "bundle_artifact_roles": {"infer_config": ["infer_config.json"]},
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                    "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "report.json").write_text(json.dumps({"dataset": "custom"}), encoding="utf-8")
    (bundle_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (bundle_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(run_dir), "--require-status", "deployable", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    acceptance = out["acceptance"]
    assert acceptance["ready"] is True
    assert acceptance["acceptance_state"] == "deployable"
    assert acceptance["reason_codes"] == []
    assert acceptance["infer_config"]["selected_source"] == "deploy_bundle"
    assert acceptance["infer_config"]["path"].endswith("deploy_bundle/infer_config.json")


def test_runs_cli_acceptance_blocks_invalid_bundle_weights_metadata(tmp_path, capsys):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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
    (run_dir / "artifacts").mkdir()
    (run_dir / "artifacts" / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "artifacts" / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )

    bundle_dir = run_dir / "deploy_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "infer_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model": {"name": "vision_ecod", "model_kwargs": {}},
                "threshold": 0.5,
                "artifact_quality": {
                    "status": "deployable",
                    "threshold_scope": "image",
                    "has_threshold_provenance": True,
                    "has_split_fingerprint": False,
                    "has_prediction_policy": False,
                    "has_operator_contract": False,
                    "has_deploy_bundle": True,
                    "has_bundle_manifest": True,
                    "required_bundle_artifacts_present": True,
                    "bundle_artifact_roles": {"infer_config": ["infer_config.json"]},
                    "audit_refs": {"calibration_card": "calibration_card.json"},
                    "deploy_refs": {"bundle_manifest": "bundle_manifest.json"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "calibration_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_context": {"scope": "image", "category_count": 1},
                "image_threshold": {
                    "threshold": 0.5,
                    "provenance": {"method": "fixed", "source": "test"},
                },
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "report.json").write_text(json.dumps({"dataset": "custom"}), encoding="utf-8")
    (bundle_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (bundle_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (bundle_dir / "model_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "broken_model",
                "summary": {
                    "purpose": "demo",
                    "intended_inputs": "RGB",
                    "output_contract": "image-level",
                },
                "weights": {
                    "path": "missing_model.pt",
                    "source": "unit-test",
                    "license": "internal",
                },
                "deployment": {"runtime": "torch"},
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(build_deploy_bundle_manifest(bundle_dir=bundle_dir, source_run_dir=run_dir)),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(run_dir), "--require-status", "audited", "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 1
    acceptance = out["acceptance"]
    assert acceptance["status"] == "partial"
    assert acceptance["ready"] is False
    assert acceptance["bundle_weights"]["applicable"] is True
    assert acceptance["bundle_weights"]["valid"] is False
    assert "bundle_weights_not_ready" in acceptance["blocking_reasons"]


def test_runs_cli_list_supports_robustness_kind(tmp_path, capsys):
    from pyimgano.runs_cli import main

    run_dir = tmp_path / "robust_run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "vision_ecod",
                "robustness": {"clean": {"results": {"auroc": 0.95}}, "corruptions": {}},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    rc = main(["list", "--root", str(tmp_path), "--kind", "robustness", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["runs"]) == 1
    assert out["runs"][0]["kind"] == "robustness"
    assert (
        out["runs"][0]["robustness_protocol"]["comparability_hints"][
            "requires_same_corruption_protocol"
        ]
        is True
    )
    assert out["runs"][0]["robustness_trust"]["status"] == "partial"


def test_runs_cli_list_revalidates_saved_robustness_artifact_digests(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                        "robustness_conditions_csv": _sha256_file(conditions_csv),
                        "robustness_summary_json": _sha256_file(summary_json),
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

    rc = main(["list", "--root", str(tmp_path), "--kind", "robustness", "--json"])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    trust = out["runs"][0]["robustness_trust"]
    assert trust["status"] == "partial"
    assert trust["trust_signals"]["has_audit_digests"] is False
    assert "audit_digest_mismatch.robustness_conditions_csv" in trust["degraded_by"]


def test_runs_cli_compare_json_uses_robustness_contract(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["evaluation_contract"]["primary_metric"] == "worst_corruption_auroc"
    assert (
        out["evaluation_contract"]["comparability_hints"]["requires_same_corruption_protocol"]
        is True
    )
    assert out["baseline_run"]["robustness_trust"]["status"] == "trust-signaled"
    assert out["metrics"]["worst_corruption_auroc"]["comparisons"][1]["status"] == "regressed"


def test_runs_cli_compare_json_emits_trust_comparison_summary(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["trust_comparison"]["checked"] is True
    assert out["trust_comparison"]["gate"] == "limited"
    assert out["trust_comparison"]["status"] == "partial"
    assert out["trust_comparison"]["reason"] == "missing_split_fingerprint"
    assert out["trust_comparison"]["operator_contract_status"] == "missing"
    assert out["trust_comparison"]["operator_contract_consistent"] is False
    assert out["trust_comparison"]["audit_refs"]["calibration_card_json"] == (
        "artifacts/calibration_card.json"
    )


def test_runs_cli_compare_can_fail_on_robustness_drop_regression(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--metric",
            "worst_corruption_drop_auroc",
            "--fail-on-regression",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["summary"]["total_regressions"] == 1
    assert out["metrics"]["worst_corruption_drop_auroc"]["direction"] == "lower_is_better"
    assert out["metrics"]["worst_corruption_drop_auroc"]["comparisons"][1]["status"] == "regressed"


def test_runs_cli_compare_json_emits_machine_readable_verdict_summary(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["regression_gate"] == "regressed"
    assert out["summary"]["comparability_gates"]["split"] == "incompatible"
    assert out["summary"]["comparability_gates"]["environment"] == "incompatible"
    assert out["summary"]["comparability_gates"]["target"] == "incompatible"
    assert out["summary"]["comparability_gates"]["robustness_protocol"] == "unchecked"
    assert out["summary"]["comparability_gates"]["operator_contract"] == "unchecked"
    assert out["summary"]["comparability_gates"]["bundle_operator_contract"] == "unchecked"
    assert out["summary"]["blocking_flags"] == [
        "--fail-on-regression",
        "--require-same-split",
        "--require-same-environment",
        "--require-same-target",
    ]
    assert out["summary"]["verdict"] == "blocked"


def test_runs_cli_compare_json_emits_machine_readable_metric_and_trust_summary(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["primary_metric"] == "auroc"
    assert out["summary"]["primary_metric_direction"] == "higher_is_better"
    assert out["summary"]["primary_metric_baseline"] == 0.91
    assert out["summary"]["primary_metric_total_regressions"] == 0
    assert out["summary"]["primary_metric_statuses"] == {"candidate": "improved"}
    assert out["summary"]["primary_metric_deltas"] == {"candidate": 0.01}
    assert out["summary"]["trust_checked"] is True
    assert out["summary"]["trust_gate"] == "limited"
    assert out["summary"]["trust_status"] == "partial"
    assert out["summary"]["trust_reason"] == "missing_split_fingerprint"
    assert out["summary"]["operator_contract_status"] == "missing"
    assert out["summary"]["operator_contract_consistent"] is False


def test_runs_cli_compare_can_fail_on_robustness_latency_regression(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--metric",
            "worst_corruption_latency_ratio",
            "--fail-on-regression",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["summary"]["total_regressions"] == 1
    assert out["metrics"]["worst_corruption_latency_ratio"]["direction"] == "lower_is_better"
    assert (
        out["metrics"]["worst_corruption_latency_ratio"]["comparisons"][1]["status"] == "regressed"
    )


def test_runs_cli_compare_can_fail_on_split_incompatibility(tmp_path, capsys):
    from pyimgano.runs_cli import main

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
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
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "model": "candidate",
                "results": {"auroc": 0.92},
                "split_fingerprint": {"sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-split",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["split_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["split_comparison"]["comparisons"][1]["status"] == "mismatched"


def test_runs_cli_compare_can_fail_on_target_incompatibility(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "visa",
                "category": "bottle",
                "model": "candidate",
                "results": {"auroc": 0.92},
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-target",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["target_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["target_comparison"]["comparisons"][1]["dataset_status"] == "mismatched"
    assert out["target_comparison"]["comparisons"][1]["status"] == "mismatched"


def test_runs_cli_compare_can_fail_on_environment_incompatibility(tmp_path, capsys):
    from pyimgano.runs_cli import main

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    (baseline / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "baseline"}),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps({"dataset": "mvtec", "category": "bottle", "model": "candidate"}),
        encoding="utf-8",
    )
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (candidate / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "0" * 64}),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-environment",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["environment_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["environment_comparison"]["comparisons"][1]["status"] == "mismatched"


def test_runs_cli_compare_can_fail_on_robustness_protocol_incompatibility(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "candidate",
                "input_mode": "paths",
                "resize": [512, 512],
                "robustness": {
                    "corruption_mode": "clean_only",
                    "clean": {"results": {"auroc": 0.94}},
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-robustness-protocol",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["robustness_protocol_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["robustness_protocol_comparison"]["comparisons"][1]["status"] == "mismatched"
    assert out["robustness_protocol_comparison"]["comparisons"][1]["mismatch_fields"] == [
        "conditions",
        "corruption_mode",
        "input_mode",
        "resize",
        "severities",
    ]


def test_runs_cli_compare_can_fail_on_operator_contract_incompatibility(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-operator-contract",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["operator_contract_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["operator_contract_comparison"]["comparisons"][1]["status"] == "missing"


def test_emit_contract_incompat_details_accepts_mapping_rows(capsys) -> None:
    from pyimgano.runs_cli import _emit_contract_incompat_details

    _emit_contract_incompat_details(
        label="operator_contract",
        comparison_payload={
            "baseline_contract_sha256": "a" * 64,
            "comparisons": [
                MappingProxyType(
                    {
                        "run_dir_name": "candidate",
                        "status": "mismatched",
                        "mismatch_reason": "baseline_mismatch",
                        "contract_sha256": "b" * 64,
                    }
                )
            ],
        },
        summary_payload={
            "checked": True,
            "matched_runs": 0,
            "mismatched_runs": 1,
            "missing_runs": 0,
        },
    )

    out = capsys.readouterr().out
    assert "operator_contract: checked=True matched=0 mismatched=1 missing=0" in out
    assert "operator_contract_baseline.sha256=" in out
    assert "operator_contract_incompat.candidate=mismatched:baseline_mismatch" in out
    assert "operator_contract_sha256.candidate=" in out


def test_runs_cli_compare_can_fail_on_bundle_operator_contract_incompatibility(tmp_path, capsys):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--require-same-bundle-operator-contract",
            "--json",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["bundle_operator_contract_comparison"]["summary"]["incompatible_runs"] == 1
    assert out["bundle_operator_contract_comparison"]["comparisons"][1]["status"] == "missing"


def test_runs_cli_compare_plain_output_prints_robustness_protocol_mismatch_fields(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "candidate",
                "input_mode": "paths",
                "resize": [512, 512],
                "robustness": {
                    "corruption_mode": "clean_only",
                    "clean": {"results": {"auroc": 0.94}},
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "robustness_protocol:" in out
    assert (
        "robustness_protocol_mismatch.candidate=conditions,corruption_mode,input_mode,resize,severities"
        in out
    )


def test_runs_cli_compare_plain_output_prints_robustness_protocol_baseline_summary(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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
    (candidate / "report.json").write_text(
        json.dumps(
            {
                "dataset": "mvtec",
                "category": "bottle",
                "model": "candidate",
                "input_mode": "numpy",
                "resize": [256, 256],
                "robustness": {
                    "corruption_mode": "full",
                    "clean": {"results": {"auroc": 0.94}},
                    "corruptions": {
                        "lighting": {
                            "severity_1": {"results": {"auroc": 0.88}},
                            "severity_2": {"results": {"auroc": 0.84}},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "robustness_protocol_baseline.corruption_mode=full" in out
    assert "robustness_protocol_baseline.conditions=clean,lighting" in out
    assert "robustness_protocol_baseline.severities=1,2" in out
    assert "robustness_protocol_baseline.input_mode=numpy" in out
    assert "robustness_protocol_baseline.resize=256,256" in out


def test_runs_cli_compare_plain_output_prints_split_environment_target_details(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                "results": {"auroc": 0.92},
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "split_baseline.sha256=" + ("f" * 64) in out
    assert "environment_baseline.fingerprint_sha256=" + ("e" * 64) in out
    assert "target_baseline.dataset=mvtec" in out
    assert "target_baseline.category=bottle" in out
    assert "split_incompat.candidate=mismatched" in out
    assert "environment_incompat.candidate=mismatched" in out
    assert "target_incompat.candidate=dataset:mismatched,category:mismatched" in out


def test_runs_cli_compare_plain_output_prints_primary_metric_and_baseline_trust_summary(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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
                "results": {"auroc": 0.92},
            }
        ),
        encoding="utf-8",
    )
    (baseline / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    (baseline / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_primary_metric=auroc" in out
    assert "comparison_primary_metric_direction=higher_is_better" in out
    assert "comparison_primary_metric_baseline=0.91" in out
    assert "comparison_primary_metric_total_regressions=0" in out
    assert "baseline_quality=reproducible" in out
    assert "baseline_trust=partial" in out
    assert "comparison_operator_contract_status=missing" in out
    assert "comparison_operator_contract_consistent=false" in out
    assert "comparison_bundle_operator_contract_status=missing" in out
    assert "comparison_bundle_operator_contract_consistent=false" in out
    assert "baseline_reason=core_artifacts_present" in out
    assert "primary_metric_status.candidate=improved" in out
    assert "primary_metric_delta.candidate=0.01" in out


def test_runs_cli_compare_plain_output_prints_operator_contract_incompatibility_details(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert (
        "comparability_gates: split=compatible environment=compatible target=compatible "
        "robustness_protocol=unchecked operator_contract=incompatible "
        "bundle_operator_contract=unchecked"
    ) in out
    operator_baseline_hash_lines = [
        line for line in out.splitlines() if line.startswith("operator_contract_baseline.sha256=")
    ]
    assert len(operator_baseline_hash_lines) == 1
    assert len(operator_baseline_hash_lines[0].split("=", 1)[1]) == 64
    assert "operator_contract: checked=true matched=0 mismatched=0 missing=1" in out
    assert "operator_contract_incompat.candidate=missing" in out
    assert "--require-same-operator-contract" in out


def test_runs_cli_compare_plain_output_prints_bundle_operator_contract_incompatibility_details(
    tmp_path, capsys
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert (
        "comparability_gates: split=compatible environment=compatible target=compatible "
        "robustness_protocol=unchecked operator_contract=compatible "
        "bundle_operator_contract=incompatible"
    ) in out
    assert "comparison_bundle_operator_contract_digests_valid=true" in out
    bundle_baseline_hash_lines = [
        line
        for line in out.splitlines()
        if line.startswith("bundle_operator_contract_baseline.sha256=")
    ]
    assert len(bundle_baseline_hash_lines) == 1
    assert len(bundle_baseline_hash_lines[0].split("=", 1)[1]) == 64
    assert "candidate_bundle_operator_contract_digest_status.candidate=missing" in out
    assert "bundle_operator_contract: checked=true matched=0 mismatched=0 missing=1" in out
    assert "bundle_operator_contract_incompat.candidate=missing" in out
    assert "--require-same-bundle-operator-contract" in out


def test_runs_cli_compare_plain_output_prints_structured_run_briefs(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                "results": {"auroc": 0.92},
            }
        ),
        encoding="utf-8",
    )
    for run_dir in (baseline, candidate):
        (run_dir / "config.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "f" * 64}),
            encoding="utf-8",
        )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert (
        "baseline: baseline quality=reproducible trust=partial operator_contract=missing "
        "primary_metric=auroc:0.91 primary_metric_status=baseline"
    ) in out
    assert (
        "candidate: candidate quality=reproducible trust=partial operator_contract=missing "
        "primary_metric=auroc:0.92 primary_metric_status=improved primary_metric_delta=0.01"
    ) in out
    assert out.count("bundle_operator_contract=missing") >= 2


def test_runs_cli_compare_plain_output_prints_dataset_readiness_summary(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
        json.dumps(
            {
                **shared_report,
                "model": "baseline",
                "results": {"auroc": 0.91},
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                },
            }
        ),
        encoding="utf-8",
    )
    (candidate / "report.json").write_text(
        json.dumps(
            {
                **shared_report,
                "model": "candidate",
                "results": {"auroc": 0.92},
                "dataset_readiness": {
                    "status": "error",
                    "issue_codes": ["MISSING_TEST_ANOMALY"],
                },
            }
        ),
        encoding="utf-8",
    )
    for run_dir in (baseline, candidate):
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "baseline_dataset_readiness_status=warning" in out
    assert "baseline_dataset_issue_codes=fewshot_train_set" in out
    assert "candidate_dataset_readiness_status.candidate=error" in out
    assert "candidate_dataset_issue_codes.candidate=missing_test_anomaly" in out
    assert "baseline: baseline quality=partial trust=partial operator_contract=missing" in out
    assert "dataset_readiness=warning" in out
    assert "candidate: candidate quality=partial trust=partial operator_contract=missing" in out
    assert "dataset_readiness=error" in out
    assert (
        "candidate_incompatibility_digest.candidate="
        "verdict:pass|incompatible_gates:none|blocking_reasons:none|"
        "dataset_readiness_status:error|dataset_issue_codes:missing_test_anomaly"
    ) in out


def test_runs_cli_compare_plain_output_prints_comparability_gate_summary(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                "results": {"auroc": 0.92},
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert (
        "comparability_gates: split=incompatible environment=incompatible "
        "target=incompatible robustness_protocol=unchecked "
        "operator_contract=unchecked bundle_operator_contract=unchecked"
    ) in out


def test_runs_cli_compare_plain_output_prints_blocked_verdict_with_required_flags(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
                "results": {"auroc": 0.92},
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_verdict=blocked" in out
    assert "comparison_regression_gate=clean" in out
    assert (
        "comparison_blocking_flags=--require-same-split,--require-same-environment,"
        "--require-same-target"
    ) in out


def test_runs_cli_compare_plain_output_prints_pass_verdict_when_all_gates_clear(tmp_path, capsys):
    from pyimgano.runs_cli import main

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
    for run_dir in (baseline, candidate):
        (run_dir / "environment.json").write_text(
            json.dumps({"fingerprint_sha256": "e" * 64}),
            encoding="utf-8",
        )

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_verdict=pass" in out
    assert "comparison_regression_gate=clean" in out
    assert "comparison_blocking_flags=none" in out


def test_runs_cli_compare_plain_output_prints_candidate_verdicts_and_gates(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(clean),
            str(blocked),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "candidate_verdict.clean=pass" in out
    assert "candidate_blocking_reasons.clean=none" in out
    assert (
        "candidate_comparability_gates.clean="
        "split:matched,environment:matched,target:matched,target_dataset:matched,"
        "target_category:matched,robustness_protocol:unchecked,operator_contract:unchecked,"
        "bundle_operator_contract:unchecked"
    ) in out
    assert (
        "candidate_incompatibility_digest.clean="
        "verdict:pass|incompatible_gates:none|blocking_reasons:none"
    ) in out
    assert "candidate_verdict.blocked=blocked" in out
    assert (
        "candidate_blocking_reasons.blocked="
        "primary_metric:regressed,split:mismatched,environment:mismatched,"
        "target.dataset:mismatched,target.category:mismatched"
    ) in out
    assert (
        "candidate_comparability_gates.blocked="
        "split:mismatched,environment:mismatched,target:mismatched,"
        "target_dataset:mismatched,target_category:mismatched,"
        "robustness_protocol:unchecked,operator_contract:unchecked,"
        "bundle_operator_contract:unchecked"
    ) in out
    assert (
        "candidate_incompatibility_digest.blocked="
        "verdict:blocked|incompatible_gates:split:mismatched,environment:mismatched,"
        "target:mismatched,target_dataset:mismatched,target_category:mismatched|"
        "blocking_reasons:primary_metric:regressed,split:mismatched,environment:mismatched,"
        "target.dataset:mismatched,target.category:mismatched"
    ) in out


def test_runs_cli_compare_json_blocks_candidate_missing_operator_contract_when_baseline_consistent(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["candidate_verdicts"]["candidate"] == "blocked"
    assert out["summary"]["candidate_blocking_reasons"]["candidate"] == [
        "operator_contract:missing"
    ]
    assert len(str(out["summary"]["operator_contract_baseline_sha256"])) == 64
    assert out["summary"]["candidate_incompatibility_digest"]["candidate"] == {
        "verdict": "blocked",
        "incompatible_gates": ["operator_contract:missing"],
        "blocking_reasons": ["operator_contract:missing"],
    }


def test_runs_cli_compare_json_blocks_candidate_missing_bundle_operator_contract_when_baseline_consistent(
    tmp_path, capsys
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["candidate_verdicts"]["candidate"] == "blocked"
    assert out["summary"]["candidate_blocking_reasons"]["candidate"] == [
        "operator_contract_bundle:missing"
    ]
    assert len(str(out["summary"]["bundle_operator_contract_baseline_sha256"])) == 64
    assert out["summary"]["candidate_incompatibility_digest"]["candidate"] == {
        "verdict": "blocked",
        "incompatible_gates": ["bundle_operator_contract:missing"],
        "blocking_reasons": ["operator_contract_bundle:missing"],
    }
    assert (
        out["summary"]["candidate_bundle_operator_contract_digest_statuses"]["candidate"]
        == "missing"
    )


def test_runs_cli_compare_json_blocks_candidate_bundle_operator_contract_baseline_mismatch(
    tmp_path, capsys
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["candidate_verdicts"]["candidate"] == "blocked"
    assert out["summary"]["candidate_blocking_reasons"]["candidate"] == [
        "operator_contract_bundle:baseline_mismatch"
    ]
    assert (
        out["summary"]["candidate_bundle_operator_contract_digest_statuses"]["candidate"] == "valid"
    )


def test_runs_cli_compare_json_flags_candidate_bundle_operator_contract_digest_mismatch(
    tmp_path, capsys
):
    from pyimgano.reporting.deploy_bundle import build_deploy_bundle_manifest
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["candidate_verdicts"]["candidate"] == "blocked"
    assert (
        "operator_contract_bundle:mismatched"
        in out["summary"]["candidate_blocking_reasons"]["candidate"]
    )
    assert (
        "operator_contract_bundle:digest_mismatch"
        in out["summary"]["candidate_blocking_reasons"]["candidate"]
    )
    assert out["summary"]["candidate_incompatibility_digest"]["candidate"][
        "incompatible_gates"
    ] == ["bundle_operator_contract:mismatched"]
    assert (
        out["summary"]["candidate_bundle_operator_contract_digest_statuses"]["candidate"]
        == "invalid"
    )

    rc_plain = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    plain_out = capsys.readouterr().out.lower()
    assert rc_plain == 0
    assert "candidate_bundle_operator_contract_digest_status.candidate=invalid" in plain_out
    assert "bundle_operator_contract_integrity.candidate=digest_mismatch" in plain_out
    bundle_candidate_hash_lines = [
        line
        for line in plain_out.splitlines()
        if line.startswith("bundle_operator_contract_sha256.candidate=")
    ]
    assert len(bundle_candidate_hash_lines) == 1
    assert len(bundle_candidate_hash_lines[0].split("=", 1)[1]) == 64


def test_runs_cli_compare_plain_output_prints_limited_trust_gate_for_partial_baseline(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_trust_gate=limited" in out
    assert "comparison_trust_status=partial" in out
    assert "comparison_trust_reason=calibration_audit_incomplete" in out


def test_runs_cli_compare_plain_output_prints_trusted_trust_gate_for_audited_baseline(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

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
        json.dumps({"threshold": 0.5, "split_fingerprint": {"sha256": "f" * 64}}),
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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_trust_gate=trusted" in out
    assert "comparison_trust_status=trust-signaled" in out
    assert "comparison_trust_reason=calibration_audit_consistent" in out


def test_runs_cli_compare_plain_output_prints_trust_degradations_and_refs(tmp_path, capsys):
    from pyimgano.runs_cli import main

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

    rc = main(
        [
            "compare",
            str(baseline),
            str(candidate),
            "--baseline",
            str(baseline),
        ]
    )
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "comparison_trust_gate=limited" in out
    assert "comparison_trust_reason=missing_split_fingerprint" in out
    assert (
        "comparison_trust_degraded_by="
        "missing_split_fingerprint,missing_prediction_policy,missing_threshold_context"
    ) in out
    assert "comparison_trust_ref.infer_config_json=artifacts/infer_config.json" in out
    assert "comparison_trust_ref.calibration_card_json=artifacts/calibration_card.json" in out


def test_runs_cli_publication_json(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4"}),
        encoding="utf-8",
    )
    (export_dir / "config.json").write_text(
        json.dumps({"config": {"seed": 123}}),
        encoding="utf-8",
    )
    (export_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                    "issue_details": [],
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(export_dir / "report.json"),
                    "config_json": _sha256_file(export_dir / "config.json"),
                    "environment_json": _sha256_file(export_dir / "environment.json"),
                },
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(export_dir / "leaderboard.csv"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["publication", str(export_dir), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["publication"]["status"] == "ready"
    assert out["publication"]["publication_ready"] is True
    assert out["publication"]["dataset_readiness"]["status"] == "warning"
    assert out["publication"]["trust_signals"]["has_evaluation_contract"] is True
    assert out["publication"]["trust_signals"]["has_benchmark_citation"] is True
    assert out["publication"]["trust_signals"]["has_benchmark_provenance"] is True
    assert out["publication"]["audit_refs"]["benchmark_config_source"].endswith(".json")


def test_runs_cli_acceptance_routes_suite_export_to_publication_gate(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4"}),
        encoding="utf-8",
    )
    (export_dir / "config.json").write_text(
        json.dumps({"config": {"seed": 123}}),
        encoding="utf-8",
    )
    (export_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                    "issue_details": [],
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(export_dir / "report.json"),
                    "config_json": _sha256_file(export_dir / "config.json"),
                    "environment_json": _sha256_file(export_dir / "environment.json"),
                },
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(export_dir / "leaderboard.csv"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(export_dir), "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 0
    acceptance = out["acceptance"]
    assert acceptance["kind"] == "publication"
    assert acceptance["status"] == "ready"
    assert acceptance["ready"] is True
    assert acceptance["blocking_reasons"] == []
    assert acceptance["publication"]["status"] == "ready"
    assert acceptance["publication"]["publication_ready"] is True
    assert acceptance["publication"]["dataset_readiness"]["status"] == "warning"


def test_runs_cli_acceptance_plain_output_prints_publication_dataset_readiness(
    tmp_path, capsys
):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4"}),
        encoding="utf-8",
    )
    (export_dir / "config.json").write_text(
        json.dumps({"config": {"seed": 123}}),
        encoding="utf-8",
    )
    (export_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                    "issue_details": [],
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(export_dir / "report.json"),
                    "config_json": _sha256_file(export_dir / "config.json"),
                    "environment_json": _sha256_file(export_dir / "environment.json"),
                },
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(export_dir / "leaderboard.csv"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(export_dir)])
    out = capsys.readouterr().out.lower()

    assert rc == 0
    assert "kind=publication" in out
    assert "publication_ready=true" in out
    assert "dataset_readiness_status=warning" in out
    assert "dataset_issue_codes=fewshot_train_set" in out


def test_runs_cli_acceptance_returns_nonzero_for_partial_suite_export(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    metadata_path = export_dir / "leaderboard_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "artifact_quality": {
                    "required_files_present": False,
                    "missing_required": ["environment_fingerprint_sha256"],
                    "has_official_benchmark_config": False,
                    "has_environment_fingerprint": False,
                    "has_split_fingerprint": True,
                },
                "publication_ready": False,
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(metadata_path),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["acceptance", str(metadata_path), "--json"])
    out = json.loads(capsys.readouterr().out)

    assert rc == 1
    acceptance = out["acceptance"]
    assert acceptance["kind"] == "publication"
    assert acceptance["status"] == "partial"
    assert acceptance["ready"] is False
    assert "publication_not_ready" in acceptance["blocking_reasons"]
    assert "missing_required_exports" in acceptance["blocking_reasons"]
    assert acceptance["publication"]["status"] == "partial"
    assert "leaderboard_csv" in acceptance["publication"]["missing_required"]


def test_runs_cli_publication_plain_output_prints_trust_signals(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4"}),
        encoding="utf-8",
    )
    (export_dir / "config.json").write_text(
        json.dumps({"config": {"seed": 123}}),
        encoding="utf-8",
    )
    (export_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "dataset_readiness": {
                    "status": "warning",
                    "issue_codes": ["FEWSHOT_TRAIN_SET"],
                    "issue_details": [],
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(export_dir / "report.json"),
                    "config_json": _sha256_file(export_dir / "config.json"),
                    "environment_json": _sha256_file(export_dir / "environment.json"),
                },
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(export_dir / "leaderboard.csv"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["publication", str(export_dir)])
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "status=ready" in out
    assert "publication_ready=true" in out
    assert "dataset_readiness=warning" in out
    assert "dataset_readiness_status=warning" in out
    assert "dataset_issue_codes=fewshot_train_set" in out
    assert "trust_signal.has_benchmark_provenance=true" in out
    assert (
        "audit_ref.benchmark_config_source=benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json"
        in out
    )


def test_runs_cli_publication_returns_nonzero_for_partial_export(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "artifact_quality": {
                    "required_files_present": False,
                    "missing_required": ["environment_fingerprint_sha256"],
                    "has_official_benchmark_config": False,
                    "has_environment_fingerprint": False,
                    "has_split_fingerprint": True,
                },
                "publication_ready": False,
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["publication", str(export_dir), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["publication"]["status"] == "partial"
    assert "leaderboard_csv" in out["publication"]["missing_required"]


def test_runs_cli_publication_returns_nonzero_for_invalid_declared_model_card(tmp_path, capsys):
    from pyimgano.runs_cli import main

    export_dir = tmp_path / "suite_export"
    export_dir.mkdir()
    (export_dir / "report.json").write_text(
        json.dumps({"suite": "industrial-v4"}),
        encoding="utf-8",
    )
    (export_dir / "config.json").write_text(
        json.dumps({"config": {"seed": 123}}),
        encoding="utf-8",
    )
    (export_dir / "environment.json").write_text(
        json.dumps({"fingerprint_sha256": "f" * 64}),
        encoding="utf-8",
    )
    (export_dir / "leaderboard.csv").write_text("name,auroc\nx,0.9\n", encoding="utf-8")
    (export_dir / "model_card.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "broken_model",
                "weights": {
                    "path": "checkpoints/demo.pt",
                    "source": "unit-test",
                    "license": "internal",
                },
                "deployment": {"runtime": "torch"},
            }
        ),
        encoding="utf-8",
    )
    (export_dir / "leaderboard_metadata.json").write_text(
        json.dumps(
            {
                "benchmark_config": {
                    "source": "benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json",
                    "official": True,
                    "sha256": "a" * 64,
                },
                "artifact_quality": {
                    "required_files_present": True,
                    "missing_required": [],
                    "has_official_benchmark_config": True,
                    "has_environment_fingerprint": True,
                    "has_split_fingerprint": True,
                },
                "environment_fingerprint_sha256": "f" * 64,
                "split_fingerprint": {"sha256": "b" * 64},
                "evaluation_contract": {"primary_metric": "auroc"},
                "citation": {"project": "pyimgano"},
                "publication_ready": True,
                "audit_refs": {
                    "report_json": "report.json",
                    "config_json": "config.json",
                    "environment_json": "environment.json",
                },
                "audit_digests": {
                    "report_json": _sha256_file(export_dir / "report.json"),
                    "config_json": _sha256_file(export_dir / "config.json"),
                    "environment_json": _sha256_file(export_dir / "environment.json"),
                },
                "exported_files": {
                    "leaderboard_csv": str(export_dir / "leaderboard.csv"),
                    "leaderboard_metadata_json": str(export_dir / "leaderboard_metadata.json"),
                    "model_card_json": str(export_dir / "model_card.json"),
                },
                "exported_file_digests": {
                    "leaderboard_csv": _sha256_file(export_dir / "leaderboard.csv"),
                    "model_card_json": _sha256_file(export_dir / "model_card.json"),
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(["publication", str(export_dir), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["publication"]["status"] == "partial"
    assert out["publication"]["invalid_declared"] == ["model_card_json"]
