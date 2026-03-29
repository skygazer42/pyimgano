from __future__ import annotations

from typing import Any, Mapping


def format_metric_value(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return f"{float(value):.6g}"


def comparison_trust_gate(trust_status: object) -> str | None:
    status_text = str(trust_status) if trust_status is not None else ""
    if not status_text:
        return None
    return "trusted" if status_text == "trust-signaled" else "limited"


def operator_contract_status_from_trust_summary(
    trust_summary: Mapping[str, Any]
) -> tuple[str, bool]:
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    has_contract = bool(signal_map.get("has_operator_contract"))
    is_consistent = bool(signal_map.get("has_operator_contract_consistent"))
    if not has_contract:
        return "missing", False
    return ("consistent" if is_consistent else "mismatched"), bool(is_consistent)


def bundle_operator_contract_status_from_trust_summary(
    trust_summary: Mapping[str, Any],
) -> tuple[str, bool]:
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    has_contract = bool(signal_map.get("has_bundle_operator_contract"))
    is_consistent = bool(signal_map.get("has_bundle_operator_contract_consistent"))
    if not has_contract:
        return "missing", False
    return ("consistent" if is_consistent else "mismatched"), bool(is_consistent)


def comparison_trust_reason(
    *,
    trust_status: object,
    quality_status: object,
    status_reasons: list[object],
    degraded_by: list[object],
) -> str | None:
    reasons = [str(item) for item in status_reasons if str(item)]
    degradations = [str(item) for item in degraded_by if str(item)]
    if str(trust_status) == "trust-signaled":
        if "calibration_audit_consistent" in reasons:
            return "calibration_audit_consistent"
        if reasons:
            return reasons[0]
    if degradations:
        return degradations[0]
    if str(quality_status) in {"reproducible", "partial"}:
        return "calibration_audit_incomplete"
    if reasons:
        return reasons[0]
    status_text = str(trust_status)
    return status_text if status_text else None


def build_trust_comparison(
    baseline_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(baseline_summary, Mapping):
        return {
            "checked": False,
            "quality_status": None,
            "gate": None,
            "status": None,
            "reason": None,
            "status_reasons": [],
            "degraded_by": [],
            "audit_refs": {},
        }

    artifact_quality = dict(baseline_summary.get("artifact_quality", {}))
    trust_summary = dict(artifact_quality.get("trust_summary", {}))
    trust_signals = trust_summary.get("trust_signals", None)
    signal_map = dict(trust_signals) if isinstance(trust_signals, Mapping) else {}
    operator_contract_status, operator_contract_consistent = (
        operator_contract_status_from_trust_summary(trust_summary)
    )
    bundle_operator_contract_status, bundle_operator_contract_consistent = (
        bundle_operator_contract_status_from_trust_summary(trust_summary)
    )
    baseline_operator_contract_status = baseline_summary.get("operator_contract_status", None)
    if isinstance(baseline_operator_contract_status, str) and baseline_operator_contract_status:
        operator_contract_status = str(baseline_operator_contract_status)
        operator_contract_consistent = operator_contract_status == "consistent"
    baseline_bundle_operator_contract_status = baseline_summary.get(
        "bundle_operator_contract_status",
        None,
    )
    if (
        isinstance(baseline_bundle_operator_contract_status, str)
        and baseline_bundle_operator_contract_status
    ):
        bundle_operator_contract_status = str(baseline_bundle_operator_contract_status)
        bundle_operator_contract_consistent = bundle_operator_contract_status == "consistent"
    quality_status = artifact_quality.get("status", None)
    trust_status = trust_summary.get("status", None)
    status_reasons = list(trust_summary.get("status_reasons", []))
    degraded_by = list(trust_summary.get("degraded_by", []))
    audit_refs = dict(trust_summary.get("audit_refs", {}))
    return {
        "checked": True,
        "quality_status": (
            str(quality_status) if isinstance(quality_status, str) and quality_status else None
        ),
        "gate": comparison_trust_gate(trust_status),
        "status": (str(trust_status) if isinstance(trust_status, str) and trust_status else None),
        "reason": comparison_trust_reason(
            trust_status=trust_status,
            quality_status=quality_status,
            status_reasons=status_reasons,
            degraded_by=degraded_by,
        ),
        "status_reasons": [str(item) for item in status_reasons if str(item)],
        "degraded_by": [str(item) for item in degraded_by if str(item)],
        "operator_contract_status": str(operator_contract_status),
        "operator_contract_consistent": bool(operator_contract_consistent),
        "bundle_operator_contract_status": str(bundle_operator_contract_status),
        "bundle_operator_contract_consistent": bool(bundle_operator_contract_consistent),
        "bundle_operator_contract_digests_valid": bool(
            signal_map.get("has_bundle_operator_contract_digests_valid", False)
        ),
        "audit_refs": {
            str(key): str(value) for key, value in audit_refs.items() if str(key) and str(value)
        },
    }


def comparability_gate_status(summary: Mapping[str, Any]) -> str:
    if not bool(summary.get("checked")):
        return "unchecked"
    if int(summary.get("incompatible_runs", 0) or 0) > 0:
        return "incompatible"
    return "compatible"


def compare_blocking_flags(
    *,
    total_regressions: int,
    split_summary: Mapping[str, Any],
    environment_summary: Mapping[str, Any],
    target_summary: Mapping[str, Any],
    robustness_protocol_summary: Mapping[str, Any],
    operator_contract_summary: Mapping[str, Any],
    bundle_operator_contract_summary: Mapping[str, Any],
) -> list[str]:
    flags: list[str] = []
    if int(total_regressions) > 0:
        flags.append("--fail-on-regression")
    if comparability_gate_status(split_summary) == "incompatible":
        flags.append("--require-same-split")
    if comparability_gate_status(environment_summary) == "incompatible":
        flags.append("--require-same-environment")
    if comparability_gate_status(target_summary) == "incompatible":
        flags.append("--require-same-target")
    if comparability_gate_status(robustness_protocol_summary) == "incompatible":
        flags.append("--require-same-robustness-protocol")
    if comparability_gate_status(operator_contract_summary) == "incompatible":
        flags.append("--require-same-operator-contract")
    if comparability_gate_status(bundle_operator_contract_summary) == "incompatible":
        flags.append("--require-same-bundle-operator-contract")
    return flags


def format_candidate_incompatibility_digest(entry: dict[str, object]) -> str:
    verdict = entry.get("verdict", None)
    verdict_text = str(verdict) if isinstance(verdict, str) and verdict else "pass"
    incompatible = entry.get("incompatible_gates", [])
    blocking = entry.get("blocking_reasons", [])
    incompatible_items = (
        [str(item) for item in incompatible if str(item)] if isinstance(incompatible, list) else []
    )
    blocking_items = (
        [str(item) for item in blocking if str(item)] if isinstance(blocking, list) else []
    )
    incompatible_text = ",".join(incompatible_items) if incompatible_items else "none"
    blocking_text = ",".join(blocking_items) if blocking_items else "none"
    return (
        f"verdict:{verdict_text}|"
        f"incompatible_gates:{incompatible_text}|"
        f"blocking_reasons:{blocking_text}"
    )


__all__ = [
    "build_trust_comparison",
    "bundle_operator_contract_status_from_trust_summary",
    "comparability_gate_status",
    "compare_blocking_flags",
    "comparison_trust_gate",
    "comparison_trust_reason",
    "format_candidate_incompatibility_digest",
    "format_metric_value",
    "operator_contract_status_from_trust_summary",
]
