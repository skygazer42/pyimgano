from __future__ import annotations

from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.runner import run_workbench
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
    return run_workbench(config=config, recipe_name=recipe_name)
