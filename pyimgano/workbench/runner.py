from __future__ import annotations

from typing import Any, Mapping

from pyimgano.train_progress import get_active_train_progress_reporter
from pyimgano.workbench.category_execution import run_workbench_category
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.infer_config_payload import build_workbench_infer_config_payload
from pyimgano.workbench.multi_category_execution import run_all_workbench_categories
from pyimgano.workbench.runtime_guardrails import validate_workbench_runtime_guardrails
from pyimgano.workbench.run_context import initialize_workbench_run_context
from pyimgano.workbench.run_report import persist_workbench_run_report


def run_workbench(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
) -> dict[str, Any]:
    validate_workbench_runtime_guardrails(config=config)

    reporter = get_active_train_progress_reporter()
    run_context = initialize_workbench_run_context(config=config, recipe_name=recipe_name)
    run_dir = None if run_context is None else run_context.run_dir
    paths = None if run_context is None else run_context.paths
    reporter.on_run_context(run_dir=(str(run_dir) if run_dir is not None else None))

    category = str(config.dataset.category)

    if category.lower() != "all":
        reporter.on_category_start(category=category, index=1, total=1)
        payload = run_workbench_category(
            config=config,
            recipe_name=recipe_name,
            category=category,
            run_dir=run_dir,
        )
        return persist_workbench_run_report(payload=payload, paths=paths)

    payload = run_all_workbench_categories(
        config=config,
        recipe_name=recipe_name,
        run_dir=run_dir,
    )

    return persist_workbench_run_report(payload=payload, paths=paths)


def build_infer_config_payload(
    *, config: WorkbenchConfig, report: Mapping[str, Any]
) -> dict[str, Any]:
    return build_workbench_infer_config_payload(config=config, report=report)
