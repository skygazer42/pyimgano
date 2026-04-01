from __future__ import annotations


def test_format_run_brief_line_renders_quality_and_trust() -> None:
    from pyimgano.runs_cli_rendering import format_run_brief_line

    line = format_run_brief_line(
        {
            "run_dir_name": "run_a",
            "kind": "benchmark",
            "dataset": "mvtec",
            "category": "bottle",
            "model_or_suite": "vision_patchcore",
            "artifact_quality": {
                "status": "audited",
                "trust_summary": {"status": "trust-signaled", "status_reasons": ["ok"]},
            },
            "operator_contract_status": "consistent",
            "bundle_operator_contract_status": "missing",
            "evaluation_contract": {"primary_metric": "auroc"},
            "metrics": {"auroc": 0.95},
        }
    )

    assert line == (
        "run_a: benchmark mvtec/bottle vision_patchcore "
        "quality=audited trust=trust-signaled operator_contract=consistent "
        "primary_metric=auroc:0.95 reason=ok bundle_operator_contract=missing"
    )


def test_format_compare_run_brief_line_includes_primary_metric_status() -> None:
    from pyimgano.runs_cli_rendering import format_compare_run_brief_line

    line = format_compare_run_brief_line(
        {
            "run_dir_name": "run_b",
            "model_or_suite": "vision_patchcore",
            "artifact_quality": {
                "status": "deployable",
                "trust_summary": {"status": "trust-signaled"},
            },
            "operator_contract_status": "consistent",
            "bundle_operator_contract_status": "consistent",
            "metrics": {"auroc": 0.97},
        },
        primary_metric_name="auroc",
        primary_metric_row={"status": "improved", "delta_vs_baseline": 0.02},
    )

    assert line == (
        "run_b: vision_patchcore quality=deployable trust=trust-signaled "
        "operator_contract=consistent primary_metric=auroc:0.97 "
        "primary_metric_status=improved primary_metric_delta=0.02 "
        "bundle_operator_contract=consistent"
    )


def test_format_quality_summary_line_includes_status_score_and_trust() -> None:
    from pyimgano.runs_cli_rendering import format_quality_summary_line

    line = format_quality_summary_line(
        run_name="run_a",
        quality={
            "status": "audited",
            "score": 0.75,
            "trust_summary": {"status": "limited"},
            "dataset_readiness": {"status": "warning", "issue_codes": ["FEWSHOT_TRAIN_SET"]},
        },
    )

    assert line == (
        "run_a: status=audited score=0.75 trust=limited "
        "dataset_readiness=warning"
    )


def test_format_acceptance_run_summary_line_includes_required_quality_and_bundle_status() -> None:
    from pyimgano.runs_cli_rendering import format_acceptance_run_summary_line

    line = format_acceptance_run_summary_line(
        run_name="run_a",
        acceptance={
            "status": "ready",
            "required_quality": "audited",
            "quality": {"status": "deployable"},
            "infer_config": {"selected_source": "deploy_bundle"},
            "bundle_weights": {"applicable": True, "status": "ready"},
        },
    )

    assert (
        line
        == "run_a: kind=run status=ready required_quality=audited quality=deployable infer_config=deploy_bundle bundle_weights=ready"
    )


def test_format_publication_summary_line_includes_status_and_ready_flag() -> None:
    from pyimgano.runs_cli_rendering import format_publication_summary_line

    line = format_publication_summary_line(
        path_name="suite_export",
        publication={"status": "ready", "publication_ready": True},
    )

    assert line == "suite_export: status=ready publication_ready=True"
