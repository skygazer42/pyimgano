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
