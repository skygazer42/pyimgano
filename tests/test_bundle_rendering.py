from __future__ import annotations


def test_format_bundle_validate_lines_preserves_text_contract() -> None:
    from pyimgano.bundle_rendering import format_bundle_validate_lines

    lines = format_bundle_validate_lines(
        {
            "bundle_dir": "/tmp/deploy_bundle",
            "status": "ready",
            "ready": True,
            "reason_codes": ["BUNDLE_OK"],
            "handoff_report_status": "valid",
            "next_action": "pyimgano bundle run /tmp/deploy_bundle --image-dir /tmp/in --output-dir /tmp/out --json",
            "watch_command": "pyimgano bundle watch /tmp/deploy_bundle --watch-dir /tmp/inbox --output-dir /tmp/out --once --json",
            "contract": {"bundle_type": "cpu-offline-qc"},
        }
    )

    assert lines == [
        "bundle_dir=/tmp/deploy_bundle",
        "status=ready",
        "ready=true",
        "reason_code=BUNDLE_OK",
        "bundle_type=cpu-offline-qc",
        "handoff_report_status=valid",
        "next_action=pyimgano bundle run /tmp/deploy_bundle --image-dir /tmp/in --output-dir /tmp/out --json",
        "watch_command=pyimgano bundle watch /tmp/deploy_bundle --watch-dir /tmp/inbox --output-dir /tmp/out --once --json",
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


def test_format_bundle_watch_lines_preserves_text_contract() -> None:
    from pyimgano.bundle_rendering import format_bundle_watch_lines

    lines = format_bundle_watch_lines(
        {
            "bundle_dir": "/tmp/deploy_bundle",
            "watch_dir": "/tmp/inbox",
            "output_dir": "/tmp/output",
            "status": "completed",
            "processed": 3,
            "pending": 1,
            "error": 0,
            "delivery_summary": {"pending_retry": 1},
            "next_delivery_attempt_after_min": 130.0,
            "last_delivery_error_path": "sample.png",
            "last_delivery_error": "RuntimeError: temporary webhook outage",
            "artifacts": {"results_jsonl": "/tmp/output/results.jsonl"},
        }
    )

    assert lines == [
        "bundle_dir=/tmp/deploy_bundle",
        "watch_dir=/tmp/inbox",
        "output_dir=/tmp/output",
        "status=completed",
        "processed=3",
        "pending=1",
        "error=0",
        "pending_retry=1",
        "next_retry_after=130.0",
        "last_delivery_error_path=sample.png",
        "last_delivery_error=RuntimeError: temporary webhook outage",
        "results_jsonl=/tmp/output/results.jsonl",
    ]
