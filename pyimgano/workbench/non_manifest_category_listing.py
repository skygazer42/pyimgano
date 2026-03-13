from __future__ import annotations

from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import list_workbench_categories


def load_non_manifest_preflight_categories(
    *,
    config: WorkbenchConfig,
    dataset: str,
    root: str,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    try:
        categories = list_workbench_categories(config=config)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        issues.append(
            issue_builder(
                "DATASET_CATEGORY_LIST_FAILED",
                "error",
                "Unable to list dataset categories.",
                context={"dataset": str(dataset), "root": str(root), "error": str(exc)},
            )
        )
        return {
            "categories": None,
            "summary": {"dataset_root": str(root), "ok": False},
        }

    return {
        "categories": categories,
        "summary": None,
    }


__all__ = ["load_non_manifest_preflight_categories"]
