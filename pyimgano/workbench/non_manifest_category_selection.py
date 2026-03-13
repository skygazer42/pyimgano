from __future__ import annotations

from typing import Any, Callable, Iterable


def select_non_manifest_preflight_categories(
    *,
    requested_category: str,
    available_categories: Iterable[str],
    dataset: str,
    root: str,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    available_sorted = sorted(str(category) for category in available_categories)
    requested_all = requested_category.lower() == "all"

    if requested_all:
        return {
            "requested_all": True,
            "categories": available_sorted,
        }

    if requested_category not in set(available_sorted):
        issues.append(
            issue_builder(
                "DATASET_CATEGORY_EMPTY",
                "error",
                "Requested category not found in dataset.",
                context={
                    "dataset": str(dataset),
                    "root": str(root),
                    "category": str(requested_category),
                    "available_categories": available_sorted,
                },
            )
        )

    return {
        "requested_all": False,
        "categories": [str(requested_category)],
    }


__all__ = ["select_non_manifest_preflight_categories"]
