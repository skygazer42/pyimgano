from __future__ import annotations

from typing import Any, Iterable, Mapping

_HIGHER_IS_BETTER_METRICS = {
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
    "clean_auroc",
    "clean_average_precision",
    "clean_pixel_auroc",
    "clean_pixel_average_precision",
    "clean_aupro",
    "clean_pixel_segf1",
    "mean_corruption_auroc",
    "mean_corruption_average_precision",
    "mean_corruption_pixel_auroc",
    "mean_corruption_pixel_average_precision",
    "mean_corruption_aupro",
    "mean_corruption_pixel_segf1",
    "worst_corruption_auroc",
    "worst_corruption_average_precision",
    "worst_corruption_pixel_auroc",
    "worst_corruption_pixel_average_precision",
    "worst_corruption_aupro",
    "worst_corruption_pixel_segf1",
}

_LOWER_IS_BETTER_METRICS = {
    "clean_latency_ms_per_image",
    "mean_corruption_drop_auroc",
    "mean_corruption_drop_average_precision",
    "mean_corruption_drop_pixel_auroc",
    "mean_corruption_drop_pixel_average_precision",
    "mean_corruption_drop_aupro",
    "mean_corruption_drop_pixel_segf1",
    "mean_corruption_latency_ms_per_image",
    "mean_corruption_latency_ratio",
    "worst_corruption_drop_auroc",
    "worst_corruption_drop_average_precision",
    "worst_corruption_drop_pixel_auroc",
    "worst_corruption_drop_pixel_average_precision",
    "worst_corruption_drop_aupro",
    "worst_corruption_drop_pixel_segf1",
    "worst_corruption_latency_ms_per_image",
    "worst_corruption_latency_ratio",
}

_DEFAULT_METRICS = [
    "auroc",
    "average_precision",
    "pixel_auroc",
    "pixel_average_precision",
    "aupro",
    "pixel_segf1",
]

_DEFAULT_COMPARABILITY_HINTS = {
    "requires_same_dataset": True,
    "requires_same_category": True,
    "requires_same_split": True,
    "recommends_same_environment": True,
}


def metric_direction(name: str) -> str:
    metric_name = str(name)
    if metric_name in _LOWER_IS_BETTER_METRICS:
        return "lower_is_better"
    if metric_name in _HIGHER_IS_BETTER_METRICS:
        return "higher_is_better"
    return "higher_is_better"


def _normalize_metric_names(metric_names: Iterable[Any]) -> list[str]:
    names = {str(name).strip() for name in metric_names if str(name).strip()}
    return sorted(names)


def build_evaluation_contract(
    *,
    metric_names: Iterable[Any],
    primary_metric: str | None = None,
    ranking_metric: str | None = None,
    pixel_metrics_enabled: bool | None = None,
    comparability_hints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    names = _normalize_metric_names(metric_names)
    if not names:
        names = list(_DEFAULT_METRICS)

    primary = str(primary_metric).strip() if primary_metric is not None else ""
    if not primary:
        primary = "auroc" if "auroc" in names else str(names[0])
    if primary not in names:
        names.append(primary)
        names.sort()

    ranking = str(ranking_metric).strip() if ranking_metric is not None else ""
    if not ranking:
        ranking = str(primary)
    if ranking not in names:
        names.append(ranking)
        names.sort()

    hints = dict(_DEFAULT_COMPARABILITY_HINTS)
    if comparability_hints is not None:
        for key, value in dict(comparability_hints).items():
            hints[str(key)] = bool(value)

    return {
        "schema_version": 1,
        "primary_metric": str(primary),
        "ranking_metric": str(ranking),
        "metric_directions": {str(name): metric_direction(str(name)) for name in names},
        "pixel_metrics_enabled": (
            bool(pixel_metrics_enabled) if pixel_metrics_enabled is not None else None
        ),
        "comparability_hints": hints,
    }


__all__ = [
    "build_evaluation_contract",
    "metric_direction",
]
