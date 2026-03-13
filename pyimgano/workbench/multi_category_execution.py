from __future__ import annotations

from pathlib import Path
from typing import Any

from pyimgano.workbench.aggregate_report import build_workbench_aggregate_report
from pyimgano.workbench.category_execution import run_workbench_category
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.dataset_loader import list_workbench_categories


def run_all_workbench_categories(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
    run_dir: Path | None,
) -> dict[str, Any]:
    categories = list_workbench_categories(config=config)

    per_category: dict[str, Any] = {}
    for cat in categories:
        per_category[str(cat)] = run_workbench_category(
            config=config,
            recipe_name=recipe_name,
            category=str(cat),
            run_dir=run_dir,
        )

    return build_workbench_aggregate_report(
        config=config,
        recipe_name=recipe_name,
        categories=categories,
        per_category=per_category,
    )


__all__ = ["run_all_workbench_categories"]
