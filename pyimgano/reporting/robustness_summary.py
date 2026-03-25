from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_ROBUSTNESS_COMPARABILITY_HINTS = {
    "requires_same_dataset": True,
    "requires_same_category": True,
    "requires_same_split": True,
    "requires_same_input_mode": True,
    "requires_same_resize": True,
    "requires_same_corruption_protocol": True,
    "requires_same_severities": True,
    "recommends_same_environment": True,
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_string_mapping(payload: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in (payload.items() if isinstance(payload, Mapping) else ())
        if isinstance(value, str) and value.strip()
    }


def _resolve_audit_artifact_path(ref: str, audit_root: Path) -> Path:
    artifact_path = Path(ref)
    if artifact_path.is_absolute():
        return artifact_path
    return audit_root / artifact_path


def _audit_material_status(
    key: str,
    *,
    refs: Mapping[str, str],
    digests: Mapping[str, str],
    audit_root: Path,
) -> tuple[bool, bool, list[str]]:
    ref = refs.get(key)
    if ref is None:
        return False, False, [f"missing_audit_ref.{key}"]

    artifact_path = _resolve_audit_artifact_path(ref, audit_root)
    if not artifact_path.is_file():
        return False, False, [f"missing_audit_artifact.{key}"]

    digest = digests.get(key)
    if digest is None:
        return True, False, [f"missing_audit_digest.{key}"]
    if _file_sha256(artifact_path) != digest:
        return True, False, [f"audit_digest_mismatch.{key}"]
    return True, True, []


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _iter_corruption_payloads(report: Mapping[str, Any]):
    corruptions = report.get("corruptions", None)
    if not isinstance(corruptions, Mapping):
        return
    for condition_name, by_severity in corruptions.items():
        if not isinstance(by_severity, Mapping):
            continue
        for severity_name, sev_payload in by_severity.items():
            if not isinstance(sev_payload, Mapping):
                continue
            yield str(condition_name), str(severity_name), sev_payload


def _iter_corruption_results(report: Mapping[str, Any]):
    for _condition_name, _severity_name, sev_payload in _iter_corruption_payloads(report):
        results = sev_payload.get("results", None)
        if isinstance(results, Mapping):
            yield results


def _extract_robustness_metric_values(
    report: dict[str, Any],
    *,
    image_key: str,
    pixel_key: str | None = None,
) -> tuple[float | None, list[float]]:
    clean = report.get("clean", None)
    clean_value = None
    if isinstance(clean, dict):
        clean_results = clean.get("results", None)
        if isinstance(clean_results, dict):
            clean_value = _safe_float(clean_results.get(image_key, None))
            if clean_value is None and pixel_key is not None:
                pixel_metrics = clean_results.get("pixel_metrics", None)
                if isinstance(pixel_metrics, dict):
                    clean_value = _safe_float(pixel_metrics.get(pixel_key, None))

    corruption_values: list[float] = []
    for results in _iter_corruption_results(report):
        value = _safe_float(results.get(image_key, None))
        if value is None and pixel_key is not None:
            pixel_metrics = results.get("pixel_metrics", None)
            if isinstance(pixel_metrics, Mapping):
                value = _safe_float(pixel_metrics.get(pixel_key, None))
        if value is not None:
            corruption_values.append(float(value))
    return clean_value, corruption_values


def _extract_latency_values(report: dict[str, Any]) -> tuple[float | None, list[float]]:
    clean = report.get("clean", None)
    clean_latency = None
    if isinstance(clean, dict):
        clean_latency = _safe_float(clean.get("latency_ms_per_image", None))

    corruption_latencies: list[float] = []
    for _condition_name, _severity_name, sev_payload in _iter_corruption_payloads(report):
        value = _safe_float(sev_payload.get("latency_ms_per_image", None))
        if value is not None:
            corruption_latencies.append(float(value))
    return clean_latency, corruption_latencies


def _has_report_metrics(report: Mapping[str, Any]) -> bool:
    clean = report.get("clean", None)
    if isinstance(clean, Mapping):
        results = clean.get("results", None)
        if isinstance(results, Mapping) and results:
            return True

    for results in _iter_corruption_results(report):
        if results:
            return True
    return False


def summarize_robustness_protocol(report: dict[str, Any]) -> dict[str, Any]:
    conditions: list[str] = []
    severities: set[int] = set()
    condition_count = 0

    clean = report.get("clean", None)
    has_clean_baseline = isinstance(clean, dict)
    if has_clean_baseline:
        conditions.append("clean")
        condition_count += 1

    corruption_names: list[str] = []
    seen_conditions: set[str] = set()
    corruption_names: list[str] = []
    for condition_name, severity_name, _sev_payload in _iter_corruption_payloads(report):
        if condition_name not in seen_conditions:
            seen_conditions.add(condition_name)
            corruption_names.append(condition_name)
        condition_count += 1
        text = severity_name
        if text.startswith("severity_"):
            text = text.split("_", 1)[1]
        try:
            severities.add(int(text))
        except Exception:
            continue

    conditions.extend(sorted(corruption_names))

    return {
        "corruption_mode": report.get("corruption_mode", None),
        "has_clean_baseline": bool(has_clean_baseline),
        "condition_count": int(condition_count),
        "corruption_count": int(len(corruption_names)),
        "severity_count": int(len(severities)),
        "conditions": conditions,
        "severities": sorted(severities),
        "comparability_hints": dict(_ROBUSTNESS_COMPARABILITY_HINTS),
    }


def summarize_robustness_report(report: dict[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    metric_specs = (
        ("auroc", "auroc"),
        ("average_precision", "average_precision"),
        ("pixel_auroc", "pixel_auroc"),
        ("pixel_average_precision", "pixel_average_precision"),
        ("aupro", "aupro"),
        ("pixel_segf1", "pixel_segf1"),
    )

    for metric_name, lookup_name in metric_specs:
        clean_value, corruption_values = _extract_robustness_metric_values(
            report,
            image_key=lookup_name,
            pixel_key=lookup_name,
        )
        if clean_value is not None:
            summary[f"clean_{metric_name}"] = float(clean_value)
        if not corruption_values:
            continue

        arr = np.asarray(corruption_values, dtype=np.float64)
        mean_value = round(float(np.mean(arr)), 12)
        worst_value = round(float(np.min(arr)), 12)
        summary[f"mean_corruption_{metric_name}"] = mean_value
        summary[f"worst_corruption_{metric_name}"] = worst_value

        if clean_value is not None:
            summary[f"mean_corruption_drop_{metric_name}"] = round(
                float(clean_value) - mean_value,
                12,
            )
            summary[f"worst_corruption_drop_{metric_name}"] = round(
                float(clean_value) - worst_value,
                12,
            )

    clean_latency, corruption_latencies = _extract_latency_values(report)
    if clean_latency is not None:
        summary["clean_latency_ms_per_image"] = float(clean_latency)
    if corruption_latencies:
        arr = np.asarray(corruption_latencies, dtype=np.float64)
        mean_latency = round(float(np.mean(arr)), 12)
        worst_latency = round(float(np.max(arr)), 12)
        summary["mean_corruption_latency_ms_per_image"] = mean_latency
        summary["worst_corruption_latency_ms_per_image"] = worst_latency
        if clean_latency is not None and float(clean_latency) > 0.0:
            summary["mean_corruption_latency_ratio"] = round(
                mean_latency / float(clean_latency),
                12,
            )
            summary["worst_corruption_latency_ratio"] = round(
                worst_latency / float(clean_latency),
                12,
            )

    return summary


def build_robustness_trust_summary(
    *,
    report: Mapping[str, Any],
    robustness_summary: Mapping[str, Any] | None = None,
    robustness_protocol: Mapping[str, Any] | None = None,
    audit_refs: Mapping[str, Any] | None = None,
    audit_digests: Mapping[str, Any] | None = None,
    audit_root: Path | None = None,
) -> dict[str, Any]:
    protocol = _normalized_robustness_protocol(report, robustness_protocol)
    summary = _normalized_robustness_summary(report, robustness_summary)
    trust_signals = _robustness_trust_signals(report, protocol, summary)
    refs, digests, audit_degraded_by = _normalized_robustness_audit_materials(
        audit_refs=audit_refs,
        audit_digests=audit_digests,
        audit_root=audit_root,
    )
    trust_signals["has_audit_refs"] = bool(refs)
    trust_signals["has_audit_digests"] = bool(digests)

    status_reasons = _robustness_status_reasons(trust_signals)
    degraded_by = _robustness_degraded_by(trust_signals, audit_degraded_by)
    status = _robustness_trust_status(trust_signals, degraded_by)

    return {
        "status": status,
        "status_reasons": list(dict.fromkeys(str(item) for item in status_reasons)),
        "trust_signals": trust_signals,
        "degraded_by": list(dict.fromkeys(str(item) for item in degraded_by)),
        "audit_refs": refs,
        "audit_digests": digests,
    }


def _normalized_robustness_protocol(
    report: Mapping[str, Any],
    robustness_protocol: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(robustness_protocol, Mapping):
        return dict(robustness_protocol)
    return summarize_robustness_protocol(dict(report))


def _normalized_robustness_summary(
    report: Mapping[str, Any],
    robustness_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(robustness_summary, Mapping):
        return dict(robustness_summary)
    return summarize_robustness_report(dict(report))


def _robustness_trust_signals(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> dict[str, bool]:
    corruption_count = int(protocol.get("corruption_count", 0) or 0)
    severity_count = int(protocol.get("severity_count", 0) or 0)
    raw_corruption_mode = str(protocol.get("corruption_mode", "") or "").strip().lower()
    full_corruption_mode = raw_corruption_mode == "full" or (
        raw_corruption_mode == "" and corruption_count > 0
    )
    has_summary_metrics = bool(summary) or _has_report_metrics(report)
    summary_has_latency = any(
        _safe_float(summary.get(key))
        is not None
        for key in (
            "clean_latency_ms_per_image",
            "mean_corruption_latency_ms_per_image",
            "worst_corruption_latency_ms_per_image",
        )
    )
    clean_latency, corruption_latencies = _extract_latency_values(dict(report))
    has_latency_profile = bool(
        summary_has_latency or clean_latency is not None or bool(corruption_latencies)
    )
    return {
        "has_clean_baseline": bool(protocol.get("has_clean_baseline")),
        "has_corruption_conditions": bool(corruption_count > 0),
        "has_summary_metrics": bool(has_summary_metrics),
        "has_latency_profile": bool(has_latency_profile),
        "has_severity_schedule": bool(severity_count > 0),
        "full_corruption_mode": bool(full_corruption_mode),
        "has_comparability_hints": isinstance(protocol.get("comparability_hints"), Mapping),
        "has_audit_refs": False,
        "has_audit_digests": False,
    }


def _normalized_robustness_audit_materials(
    *,
    audit_refs: Mapping[str, Any] | None,
    audit_digests: Mapping[str, Any] | None,
    audit_root: Path | None,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    refs = _normalized_string_mapping(audit_refs)
    digests = _normalized_string_mapping(audit_digests)
    if audit_root is None or not (refs or digests):
        return refs, digests, []

    has_audit_refs = True
    has_audit_digests = True
    degraded_by: list[str] = []
    audit_keys = list(dict.fromkeys([*refs.keys(), *digests.keys()]))
    for key in audit_keys:
        refs_ok, digests_ok, issues = _audit_material_status(
            key,
            refs=refs,
            digests=digests,
            audit_root=audit_root,
        )
        has_audit_refs = has_audit_refs and refs_ok
        has_audit_digests = has_audit_digests and digests_ok
        degraded_by.extend(issues)

    if not has_audit_refs:
        refs = {}
    if not has_audit_digests:
        digests = {}
    return refs, digests, degraded_by


def _robustness_status_reasons(trust_signals: Mapping[str, bool]) -> list[str]:
    status_reasons: list[str] = []
    if trust_signals["has_clean_baseline"]:
        status_reasons.append("clean_baseline_present")
    if trust_signals["has_corruption_conditions"]:
        status_reasons.append("corruption_conditions_present")
    if trust_signals["has_summary_metrics"]:
        status_reasons.append("summary_metrics_present")
    if trust_signals["has_latency_profile"]:
        status_reasons.append("latency_profile_present")
    if trust_signals["has_severity_schedule"]:
        status_reasons.append("severity_schedule_present")
    if trust_signals["has_comparability_hints"]:
        status_reasons.append("comparability_hints_present")
    if trust_signals["full_corruption_mode"]:
        status_reasons.append("full_corruption_mode")
    return status_reasons


def _robustness_degraded_by(
    trust_signals: Mapping[str, bool],
    audit_degraded_by: Sequence[str],
) -> list[str]:
    degraded_by: list[str] = []
    if not trust_signals["has_clean_baseline"]:
        degraded_by.append("missing_clean_baseline")
    if not trust_signals["has_corruption_conditions"]:
        degraded_by.append("missing_corruption_conditions")
    if not trust_signals["has_summary_metrics"]:
        degraded_by.append("missing_summary_metrics")
    if not trust_signals["has_latency_profile"]:
        degraded_by.append("missing_latency_profile")
    if not trust_signals["has_severity_schedule"] and trust_signals["has_corruption_conditions"]:
        degraded_by.append("missing_severity_schedule")
    if not trust_signals["has_comparability_hints"]:
        degraded_by.append("missing_comparability_hints")
    if not trust_signals["full_corruption_mode"]:
        degraded_by.append("clean_only_mode")
    degraded_by.extend(str(item) for item in audit_degraded_by)
    return degraded_by


def _robustness_trust_status(
    trust_signals: Mapping[str, bool],
    degraded_by: Sequence[str],
) -> str:
    if not trust_signals["has_clean_baseline"]:
        return "broken"
    if degraded_by:
        return "partial"
    return "trust-signaled"


__all__ = [
    "build_robustness_trust_summary",
    "summarize_robustness_protocol",
    "summarize_robustness_report",
]
