from __future__ import annotations


def test_build_trust_comparison_exposes_operator_contract_status() -> None:
    from pyimgano.reporting.run_index_helpers import build_trust_comparison

    comparison = build_trust_comparison(
        {
            "artifact_quality": {
                "status": "audited",
                "trust_summary": {
                    "status": "trust-signaled",
                    "status_reasons": ["calibration_audit_consistent"],
                    "degraded_by": [],
                    "audit_refs": {"calibration_card": "artifacts/calibration_card.json"},
                    "trust_signals": {
                        "has_operator_contract": True,
                        "has_operator_contract_consistent": True,
                        "has_bundle_operator_contract": True,
                        "has_bundle_operator_contract_consistent": False,
                        "has_bundle_operator_contract_digests_valid": True,
                    },
                },
            }
        }
    )

    assert comparison["checked"] is True
    assert comparison["gate"] == "trusted"
    assert comparison["reason"] == "calibration_audit_consistent"
    assert comparison["operator_contract_status"] == "consistent"
    assert comparison["operator_contract_consistent"] is True
    assert comparison["bundle_operator_contract_status"] == "mismatched"
    assert comparison["bundle_operator_contract_consistent"] is False
    assert comparison["bundle_operator_contract_digests_valid"] is True
    assert comparison["audit_refs"] == {"calibration_card": "artifacts/calibration_card.json"}


def test_operator_contract_status_from_trust_summary_handles_missing_and_consistent() -> None:
    from pyimgano.reporting.run_index_helpers import operator_contract_status_from_trust_summary

    assert operator_contract_status_from_trust_summary({}) == ("missing", False)
    assert operator_contract_status_from_trust_summary(
        {
            "trust_signals": {
                "has_operator_contract": True,
                "has_operator_contract_consistent": True,
            }
        }
    ) == ("consistent", True)


def test_comparability_gate_status_reports_unchecked_and_incompatible() -> None:
    from pyimgano.reporting.run_index_helpers import comparability_gate_status

    assert comparability_gate_status({"checked": False, "incompatible_runs": 0}) == "unchecked"
    assert comparability_gate_status({"checked": True, "incompatible_runs": 2}) == "incompatible"
    assert comparability_gate_status({"checked": True, "incompatible_runs": 0}) == "compatible"


def test_compare_blocking_flags_uses_incompatible_gate_states() -> None:
    from pyimgano.reporting.run_index_helpers import compare_blocking_flags

    flags = compare_blocking_flags(
        total_regressions=1,
        split_summary={"checked": True, "incompatible_runs": 1},
        environment_summary={"checked": True, "incompatible_runs": 0},
        target_summary={"checked": True, "incompatible_runs": 1},
        robustness_protocol_summary={"checked": False, "incompatible_runs": 0},
        operator_contract_summary={"checked": True, "incompatible_runs": 1},
        bundle_operator_contract_summary={"checked": True, "incompatible_runs": 0},
    )

    assert flags == [
        "--fail-on-regression",
        "--require-same-split",
        "--require-same-target",
        "--require-same-operator-contract",
    ]


def test_format_candidate_incompatibility_digest_is_stable() -> None:
    from pyimgano.reporting.run_index_helpers import format_candidate_incompatibility_digest

    digest = format_candidate_incompatibility_digest(
        {
            "verdict": "blocked",
            "incompatible_gates": ["split:mismatched", "target:mismatched"],
            "blocking_reasons": ["primary_metric:regressed"],
        }
    )

    assert (
        digest
        == "verdict:blocked|incompatible_gates:split:mismatched,target:mismatched|blocking_reasons:primary_metric:regressed"
    )


def test_format_metric_value_returns_none_for_non_numeric_values() -> None:
    from pyimgano.reporting.run_index_helpers import format_metric_value

    assert format_metric_value("x") is None
    assert format_metric_value(0.91234) == "0.91234"
