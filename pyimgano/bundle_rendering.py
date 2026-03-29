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


__all__ = [
    "format_bundle_run_lines",
    "format_bundle_validate_lines",
]
