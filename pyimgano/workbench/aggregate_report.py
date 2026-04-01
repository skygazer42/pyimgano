from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.reporting.report import stamp_report_payload
from pyimgano.workbench.config import WorkbenchConfig


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _aggregate_dataset_readiness(
    *,
    categories: Sequence[str],
    per_category: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    issue_codes: list[str] = []
    issue_details: list[dict[str, str]] = []
    statuses: list[str] = []

    for category in categories:
        payload = per_category.get(str(category), {})
        if not isinstance(payload, Mapping):
            continue
        readiness = payload.get("dataset_readiness", None)
        if not isinstance(readiness, Mapping):
            continue
        status = str(readiness.get("status", "")).strip()
        if status:
            statuses.append(status)
        for item in readiness.get("issue_details", []) or []:
            if not isinstance(item, Mapping):
                continue
            code = str(item.get("code", "")).strip()
            message = str(item.get("message", "")).strip()
            if not code or code in issue_codes:
                continue
            issue_codes.append(code)
            issue_details.append({"code": code, "message": message})
        for code in readiness.get("issue_codes", []) or []:
            text = str(code).strip()
            if text and text not in issue_codes:
                issue_codes.append(text)
                issue_details.append({"code": text, "message": ""})

    if not statuses and not issue_codes:
        return None

    if "error" in statuses:
        status = "error"
    elif "warning" in statuses:
        status = "warning"
    else:
        status = "ok"

    return {
        "status": status,
        "issue_codes": issue_codes,
        "issue_details": issue_details,
    }


def build_workbench_aggregate_report(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
    categories: Sequence[str],
    per_category: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metrics_to_average = ["auroc", "average_precision"]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for key in metrics_to_average:
        values: list[float] = []
        for category in categories:
            result_payload = per_category[str(category)].get("results", {})
            value = _safe_float(result_payload.get(key, None))
            if value is not None:
                values.append(value)
        if values:
            arr = np.asarray(values, dtype=np.float64)
            means[key] = float(np.mean(arr))
            stds[key] = float(np.std(arr))

    dataset_readiness = _aggregate_dataset_readiness(
        categories=categories,
        per_category=per_category,
    )

    payload = {
        "dataset": str(config.dataset.name),
        "category": "all",
        "model": str(config.model.name),
        "recipe": str(recipe_name),
        "seed": (int(config.seed) if config.seed is not None else None),
        "input_mode": str(config.dataset.input_mode),
        "device": str(config.model.device),
        "preset": config.model.preset,
        "resize": [int(config.dataset.resize[0]), int(config.dataset.resize[1])],
        "categories": list(categories),
        "mean_metrics": means,
        "std_metrics": stds,
        "per_category": dict(per_category),
    }
    if dataset_readiness is not None:
        payload["dataset_readiness"] = dataset_readiness
    return stamp_report_payload(payload)


__all__ = ["build_workbench_aggregate_report"]
