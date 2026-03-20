from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_reason_codes(blocking_reasons: list[str], *, mapping: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    for reason in blocking_reasons:
        code = mapping.get(str(reason))
        if code is not None and code not in out:
            out.append(code)
    return out


def validate_exit_code(payload: Mapping[str, Any]) -> int:
    return 0 if bool(payload.get("ready")) else 1


def run_exit_code(status: str) -> int:
    return 0 if str(status) == "completed" else 1


def build_input_source_summary(*, kind: str, count: int) -> dict[str, object]:
    return {
        "kind": str(kind),
        "count": int(count),
    }


def build_batch_gate_summary(
    *,
    requested: bool,
    evaluated: bool,
    processed: int,
    counts: Mapping[str, object],
    rates: Mapping[str, object],
    thresholds: Mapping[str, object],
    sources: Mapping[str, object] | None = None,
    failed_gates: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "requested": bool(requested),
        "evaluated": bool(evaluated),
        "processed": int(processed),
        "counts": {
            "normal": int(counts.get("normal", 0)),
            "anomalous": int(counts.get("anomalous", 0)),
            "rejected": int(counts.get("rejected", 0)),
            "error": int(counts.get("error", 0)),
        },
        "rates": {
            "anomaly_rate": float(rates.get("anomaly_rate", 0.0)),
            "reject_rate": float(rates.get("reject_rate", 0.0)),
            "error_rate": float(rates.get("error_rate", 0.0)),
        },
        "thresholds": {
            "max_anomaly_rate": thresholds.get("max_anomaly_rate"),
            "max_reject_rate": thresholds.get("max_reject_rate"),
            "max_error_rate": thresholds.get("max_error_rate"),
            "min_processed": thresholds.get("min_processed"),
        },
        "failed_gates": list(failed_gates),
    }
    if sources is not None:
        payload["sources"] = {
            "max_anomaly_rate": sources.get("max_anomaly_rate"),
            "max_reject_rate": sources.get("max_reject_rate"),
            "max_error_rate": sources.get("max_error_rate"),
            "min_processed": sources.get("min_processed"),
        }
    return payload


__all__ = [
    "build_batch_gate_summary",
    "build_input_source_summary",
    "build_reason_codes",
    "run_exit_code",
    "validate_exit_code",
]
