from __future__ import annotations

from pyimgano.reporting.run_index_helpers import format_metric_value


def _resolve_operator_contract_status(
    *,
    run: dict[str, object],
    trust_summary: dict[str, object],
) -> str:
    run_level = run.get("operator_contract_status", None)
    if isinstance(run_level, str) and run_level:
        return str(run_level)

    trust_signals = trust_summary.get("trust_signals", {})
    signal_map = dict(trust_signals) if isinstance(trust_signals, dict) else {}
    has_contract = bool(signal_map.get("has_operator_contract"))
    if not has_contract:
        return "missing"
    return (
        "consistent" if bool(signal_map.get("has_operator_contract_consistent")) else "mismatched"
    )


def _resolve_bundle_operator_contract_status(
    *,
    run: dict[str, object],
    trust_summary: dict[str, object],
) -> str:
    run_level = run.get("bundle_operator_contract_status", None)
    if isinstance(run_level, str) and run_level:
        return str(run_level)

    trust_signals = trust_summary.get("trust_signals", {})
    signal_map = dict(trust_signals) if isinstance(trust_signals, dict) else {}
    has_contract = bool(signal_map.get("has_bundle_operator_contract"))
    if not has_contract:
        return "missing"
    return (
        "consistent"
        if bool(signal_map.get("has_bundle_operator_contract_consistent"))
        else "mismatched"
    )


def format_run_brief_line(run: dict[str, object]) -> str:
    artifact_quality = dict(run.get("artifact_quality", {}))
    trust_summary = dict(artifact_quality.get("trust_summary", {}))
    operator_contract_status = _resolve_operator_contract_status(
        run=run,
        trust_summary=trust_summary,
    )
    bundle_operator_contract_status = _resolve_bundle_operator_contract_status(
        run=run,
        trust_summary=trust_summary,
    )
    parts = [
        f"{run['run_dir_name']}: {run.get('kind')} "
        f"{run.get('dataset')}/{run.get('category')} "
        f"{run.get('model_or_suite')}",
        f"quality={artifact_quality.get('status')}",
        f"trust={trust_summary.get('status')}",
        f"operator_contract={operator_contract_status}",
    ]

    evaluation_contract = dict(run.get("evaluation_contract", {}))
    metrics = dict(run.get("metrics", {}))
    primary_metric = evaluation_contract.get("primary_metric", None)
    if isinstance(primary_metric, str) and primary_metric:
        metric_value = format_metric_value(metrics.get(primary_metric, None))
        if metric_value is not None:
            parts.append(f"primary_metric={primary_metric}:{metric_value}")
        else:
            parts.append(f"primary_metric={primary_metric}")

    status_reasons = list(trust_summary.get("status_reasons", []))
    if status_reasons:
        parts.append(f"reason={status_reasons[0]}")
    parts.append(f"bundle_operator_contract={bundle_operator_contract_status}")

    return " ".join(parts)


def format_compare_run_brief_line(
    run: dict[str, object],
    *,
    primary_metric_name: str | None = None,
    primary_metric_row: dict[str, object] | None = None,
) -> str:
    artifact_quality = dict(run.get("artifact_quality", {}))
    trust_summary = dict(artifact_quality.get("trust_summary", {}))
    operator_contract_status = _resolve_operator_contract_status(
        run=run,
        trust_summary=trust_summary,
    )
    bundle_operator_contract_status = _resolve_bundle_operator_contract_status(
        run=run,
        trust_summary=trust_summary,
    )
    parts = [
        f"{run['run_dir_name']}: {run.get('model_or_suite')}",
        f"quality={artifact_quality.get('status')}",
        f"trust={trust_summary.get('status')}",
        f"operator_contract={operator_contract_status}",
    ]

    metrics = dict(run.get("metrics", {}))
    if isinstance(primary_metric_name, str) and primary_metric_name:
        metric_value = format_metric_value(metrics.get(primary_metric_name, None))
        if metric_value is not None:
            parts.append(f"primary_metric={primary_metric_name}:{metric_value}")
        else:
            parts.append(f"primary_metric={primary_metric_name}")

    if isinstance(primary_metric_row, dict):
        status = primary_metric_row.get("status", None)
        if isinstance(status, str) and status:
            parts.append(f"primary_metric_status={status}")
        delta = format_metric_value(primary_metric_row.get("delta_vs_baseline", None))
        if delta is not None and str(status) != "baseline":
            parts.append(f"primary_metric_delta={delta}")
    parts.append(f"bundle_operator_contract={bundle_operator_contract_status}")

    return " ".join(parts)


__all__ = [
    "format_compare_run_brief_line",
    "format_run_brief_line",
]
