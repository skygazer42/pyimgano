from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def format_bundle_validate_lines(payload: dict[str, Any]) -> list[str]:
    lines = [
        f"bundle_dir={payload.get('bundle_dir')}",
        f"status={payload.get('status')}",
        f"ready={str(bool(payload.get('ready'))).lower()}",
    ]
    for code in payload.get("reason_codes", []):
        lines.append(f"reason_code={code}")
    contract = payload.get("contract", {})
    if isinstance(contract, Mapping):
        bundle_type = contract.get("bundle_type")
        if bundle_type is not None:
            lines.append(f"bundle_type={bundle_type}")
    handoff_status = payload.get("handoff_report_status")
    if handoff_status is not None:
        lines.append(f"handoff_report_status={handoff_status}")
    next_action = payload.get("next_action")
    if isinstance(next_action, str) and next_action:
        lines.append(f"next_action={next_action}")
    watch_command = payload.get("watch_command")
    if isinstance(watch_command, str) and watch_command:
        lines.append(f"watch_command={watch_command}")
    return lines


def format_bundle_run_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        f"bundle_dir={report.get('bundle_dir')}",
        f"output_dir={report.get('output_dir')}",
        f"status={report.get('status')}",
        f"processed={report.get('processed')}",
    ]
    if report.get("batch_verdict") is not None:
        lines.append(f"batch_verdict={report.get('batch_verdict')}")
    for code in report.get("reason_codes", []):
        lines.append(f"reason_code={code}")
    artifacts = report.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts.get("results_jsonl") is not None:
        lines.append(f"results_jsonl={artifacts.get('results_jsonl')}")
    return lines


def format_bundle_watch_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        f"bundle_dir={report.get('bundle_dir')}",
        f"watch_dir={report.get('watch_dir')}",
        f"output_dir={report.get('output_dir')}",
        f"status={report.get('status')}",
        f"processed={report.get('processed')}",
        f"pending={report.get('pending')}",
        f"error={report.get('error')}",
    ]
    delivery_summary = report.get("delivery_summary", {})
    if isinstance(delivery_summary, Mapping) and delivery_summary.get("pending_retry") is not None:
        lines.append(f"pending_retry={delivery_summary.get('pending_retry')}")
    if report.get("next_delivery_attempt_after_min") is not None:
        lines.append(f"next_retry_after={report.get('next_delivery_attempt_after_min')}")
    if report.get("last_delivery_error_path") is not None:
        lines.append(f"last_delivery_error_path={report.get('last_delivery_error_path')}")
    if report.get("last_delivery_error") is not None:
        lines.append(f"last_delivery_error={report.get('last_delivery_error')}")
    for code in report.get("reason_codes", []):
        lines.append(f"reason_code={code}")
    artifacts = report.get("artifacts", {})
    if isinstance(artifacts, Mapping) and artifacts.get("results_jsonl") is not None:
        lines.append(f"results_jsonl={artifacts.get('results_jsonl')}")
    return lines


__all__ = [
    "format_bundle_run_lines",
    "format_bundle_watch_lines",
    "format_bundle_validate_lines",
]
