from __future__ import annotations


def test_format_bundle_validate_lines_preserves_text_contract() -> None:
    from pyimgano.bundle_rendering import format_bundle_validate_lines

    lines = format_bundle_validate_lines(
        {
            "bundle_dir": "/tmp/deploy_bundle",
            "status": "ready",
            "ready": True,
            "reason_codes": ["BUNDLE_OK"],
            "contract": {"bundle_type": "cpu-offline-qc"},
        }
    )

    assert lines == [
        "bundle_dir=/tmp/deploy_bundle",
        "status=ready",
        "ready=true",
        "reason_code=BUNDLE_OK",
        "bundle_type=cpu-offline-qc",
    ]


def test_format_bundle_run_lines_preserves_text_contract() -> None:
    from pyimgano.bundle_rendering import format_bundle_run_lines

    lines = format_bundle_run_lines(
        {
            "bundle_dir": "/tmp/deploy_bundle",
            "output_dir": "/tmp/output",
            "status": "completed",
            "processed": 10,
            "batch_verdict": "pass",
            "reason_codes": ["RUN_OK"],
            "artifacts": {"results_jsonl": "/tmp/output/results.jsonl"},
        }
    )

    assert lines == [
        "bundle_dir=/tmp/deploy_bundle",
        "output_dir=/tmp/output",
        "status=completed",
        "processed=10",
        "batch_verdict=pass",
        "reason_code=RUN_OK",
        "results_jsonl=/tmp/output/results.jsonl",
    ]
