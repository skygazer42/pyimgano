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
    return stamp_report_payload(payload)


__all__ = ["build_workbench_aggregate_report"]
