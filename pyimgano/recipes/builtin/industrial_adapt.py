from __future__ import annotations

from typing import Any

import pyimgano.services.workbench_service as workbench_service
from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import WorkbenchConfig


@register_recipe(
    "industrial-adapt",
    tags=("builtin", "adaptation"),
    metadata={
        "description": "Adaptation-first industrial workbench recipe (tiling/postprocess/maps optional).",
    },
)
def industrial_adapt(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "industrial-adapt"
    return workbench_service.run_workbench(config=config, recipe_name=recipe_name)
