from __future__ import annotations

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


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


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
    corruptions = report.get("corruptions", None)
    if isinstance(corruptions, dict):
        for by_severity in corruptions.values():
            if not isinstance(by_severity, dict):
                continue
            for sev_payload in by_severity.values():
                if not isinstance(sev_payload, dict):
                    continue
                results = sev_payload.get("results", None)
                if not isinstance(results, dict):
                    continue
                value = _safe_float(results.get(image_key, None))
                if value is None and pixel_key is not None:
                    pixel_metrics = results.get("pixel_metrics", None)
                    if isinstance(pixel_metrics, dict):
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
    corruptions = report.get("corruptions", None)
    if isinstance(corruptions, dict):
        for by_severity in corruptions.values():
            if not isinstance(by_severity, dict):
                continue
            for sev_payload in by_severity.values():
                if not isinstance(sev_payload, dict):
                    continue
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

    corruptions = report.get("corruptions", None)
    if not isinstance(corruptions, Mapping):
        return False
    for by_severity in corruptions.values():
        if not isinstance(by_severity, Mapping):
            continue
        for sev_payload in by_severity.values():
            if not isinstance(sev_payload, Mapping):
                continue
            results = sev_payload.get("results", None)
            if isinstance(results, Mapping) and results:
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

    corruptions = report.get("corruptions", None)
    corruption_names: list[str] = []
    if isinstance(corruptions, dict):
        for condition_name, by_severity in sorted(corruptions.items()):
            if not isinstance(by_severity, dict):
                continue
            corruption_names.append(str(condition_name))
            conditions.append(str(condition_name))
            for severity_name, sev_payload in by_severity.items():
                if not isinstance(sev_payload, dict):
                    continue
                condition_count += 1
                text = str(severity_name)
                if text.startswith("severity_"):
                    text = text.split("_", 1)[1]
                try:
                    severities.add(int(text))
                except Exception:
                    continue

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
) -> dict[str, Any]:
    protocol = (
        dict(robustness_protocol)
        if isinstance(robustness_protocol, Mapping)
        else summarize_robustness_protocol(dict(report))
    )
    summary = (
        dict(robustness_summary)
        if isinstance(robustness_summary, Mapping)
        else summarize_robustness_report(dict(report))
    )

    corruption_count = int(protocol.get("corruption_count", 0) or 0)
    severity_count = int(protocol.get("severity_count", 0) or 0)
    has_clean_baseline = bool(protocol.get("has_clean_baseline"))
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
        summary_has_latency
        or clean_latency is not None
        or bool(corruption_latencies)
    )

    trust_signals = {
        "has_clean_baseline": bool(has_clean_baseline),
        "has_corruption_conditions": bool(corruption_count > 0),
        "has_summary_metrics": bool(has_summary_metrics),
        "has_latency_profile": bool(has_latency_profile),
        "has_severity_schedule": bool(severity_count > 0),
        "full_corruption_mode": bool(full_corruption_mode),
        "has_comparability_hints": isinstance(protocol.get("comparability_hints"), Mapping),
    }

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

    if not trust_signals["has_clean_baseline"]:
        status = "broken"
    elif degraded_by:
        status = "partial"
    else:
        status = "trust-signaled"

    refs: dict[str, str] = {}
    if isinstance(audit_refs, Mapping):
        for key, value in audit_refs.items():
            if isinstance(value, str) and value.strip():
                refs[str(key)] = str(value)

    return {
        "status": status,
        "status_reasons": list(dict.fromkeys(str(item) for item in status_reasons)),
        "trust_signals": trust_signals,
        "degraded_by": list(dict.fromkeys(str(item) for item in degraded_by)),
        "audit_refs": refs,
    }


__all__ = [
    "build_robustness_trust_summary",
    "summarize_robustness_protocol",
    "summarize_robustness_report",
]
