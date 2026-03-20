from __future__ import annotations


def test_build_reason_codes_uses_existing_reason_map() -> None:
    from pyimgano.bundle_cli_helpers import build_reason_codes

    codes = build_reason_codes(
        ["missing_manifest", "invalid_manifest", "missing_manifest"],
        mapping={
            "missing_manifest": "BUNDLE_MISSING_MANIFEST",
            "invalid_manifest": "BUNDLE_INVALID_MANIFEST",
        },
    )

    assert codes == [
        "BUNDLE_MISSING_MANIFEST",
        "BUNDLE_INVALID_MANIFEST",
    ]


def test_validate_exit_code_returns_zero_only_when_ready() -> None:
    from pyimgano.bundle_cli_helpers import validate_exit_code

    assert validate_exit_code({"ready": True}) == 0
    assert validate_exit_code({"ready": False}) == 1


def test_run_exit_code_returns_zero_only_for_completed_status() -> None:
    from pyimgano.bundle_cli_helpers import run_exit_code

    assert run_exit_code("completed") == 0
    assert run_exit_code("blocked") == 1


def test_build_input_source_summary_reports_kind_and_count() -> None:
    from pyimgano.bundle_cli_helpers import build_input_source_summary

    assert build_input_source_summary(kind="image_dir", count=2) == {
        "kind": "image_dir",
        "count": 2,
    }


def test_build_batch_gate_summary_preserves_existing_contract_shape() -> None:
    from pyimgano.bundle_cli_helpers import build_batch_gate_summary

    summary = build_batch_gate_summary(
        requested=True,
        evaluated=True,
        processed=3,
        counts={"normal": 1, "anomalous": 0, "rejected": 1, "error": 1},
        rates={"anomaly_rate": 0.0, "reject_rate": 1 / 3, "error_rate": 1 / 3},
        thresholds={
            "max_anomaly_rate": None,
            "max_reject_rate": 0.2,
            "max_error_rate": 0.2,
            "min_processed": 4,
        },
        failed_gates=["min_processed", "max_reject_rate", "max_error_rate"],
    )

    assert summary == {
        "requested": True,
        "evaluated": True,
        "processed": 3,
        "counts": {
            "normal": 1,
            "anomalous": 0,
            "rejected": 1,
            "error": 1,
        },
        "rates": {
            "anomaly_rate": 0.0,
            "reject_rate": 1 / 3,
            "error_rate": 1 / 3,
        },
        "thresholds": {
            "max_anomaly_rate": None,
            "max_reject_rate": 0.2,
            "max_error_rate": 0.2,
            "min_processed": 4,
        },
        "failed_gates": ["min_processed", "max_reject_rate", "max_error_rate"],
    }


def test_build_batch_gate_summary_includes_sources_when_present() -> None:
    from pyimgano.bundle_cli_helpers import build_batch_gate_summary

    summary = build_batch_gate_summary(
        requested=True,
        evaluated=True,
        processed=2,
        counts={"normal": 1, "anomalous": 1, "rejected": 0, "error": 0},
        rates={"anomaly_rate": 0.5, "reject_rate": 0.0, "error_rate": 0.0},
        thresholds={
            "max_anomaly_rate": 0.5,
            "max_reject_rate": None,
            "max_error_rate": None,
            "min_processed": 2,
        },
        sources={
            "max_anomaly_rate": "cli",
            "max_reject_rate": "unset",
            "max_error_rate": "unset",
            "min_processed": "bundle_manifest",
        },
        failed_gates=[],
    )

    assert summary["sources"] == {
        "max_anomaly_rate": "cli",
        "max_reject_rate": "unset",
        "max_error_rate": "unset",
        "min_processed": "bundle_manifest",
    }
