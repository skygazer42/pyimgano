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
    dataset_readiness_status = run.get("dataset_readiness_status", None)
    if isinstance(dataset_readiness_status, str) and dataset_readiness_status:
        parts.append(f"dataset_readiness={dataset_readiness_status}")
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
    dataset_readiness_status = run.get("dataset_readiness_status", None)
    if isinstance(dataset_readiness_status, str) and dataset_readiness_status:
        parts.append(f"dataset_readiness={dataset_readiness_status}")
    parts.append(f"bundle_operator_contract={bundle_operator_contract_status}")

    return " ".join(parts)


def format_quality_summary_line(*, run_name: str, quality: dict[str, object]) -> str:
    trust_status = dict(quality.get("trust_summary", {})).get("status")
    line = (
        f"{run_name}: status={quality.get('status')} "
        f"score={quality.get('score')} "
        f"trust={trust_status}"
    )
    dataset_readiness = quality.get("dataset_readiness", None)
    if isinstance(dataset_readiness, dict):
        status = dataset_readiness.get("status", None)
        if status is not None:
            line += f" dataset_readiness={status}"
    return line


def format_acceptance_run_summary_line(
    *,
    run_name: str,
    acceptance: dict[str, object],
) -> str:
    infer_cfg = dict(acceptance.get("infer_config", {}))
    bundle_weights = dict(acceptance.get("bundle_weights", {}))
    quality = dict(acceptance.get("quality", {}))
    bundle_status = (
        str(bundle_weights.get("status"))
        if bool(bundle_weights.get("applicable"))
        else "not_applicable"
    )
    line = (
        f"{run_name}: kind=run status={acceptance.get('status')} "
        f"required_quality={acceptance.get('required_quality')} "
        f"quality={quality.get('status')} "
        f"infer_config={infer_cfg.get('selected_source')} "
        f"bundle_weights={bundle_status}"
    )
    dataset_readiness = quality.get("dataset_readiness", None)
    if isinstance(dataset_readiness, dict):
        status = dataset_readiness.get("status", None)
        if status is not None:
            line += f" dataset_readiness={status}"
    return line


def format_publication_summary_line(
    *,
    path_name: str,
    publication: dict[str, object],
) -> str:
    line = (
        f"{path_name}: status={publication.get('status')} "
        f"publication_ready={publication.get('publication_ready')}"
    )
    dataset_readiness = publication.get("dataset_readiness", None)
    if isinstance(dataset_readiness, dict):
        status = dataset_readiness.get("status", None)
        if status is not None:
            line += f" dataset_readiness={status}"
    return line


__all__ = [
    "format_acceptance_run_summary_line",
    "format_compare_run_brief_line",
    "format_publication_summary_line",
    "format_quality_summary_line",
    "format_run_brief_line",
]
