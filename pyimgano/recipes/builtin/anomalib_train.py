from __future__ import annotations

from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import WorkbenchConfig


@register_recipe(
    "anomalib-train",
    tags=("builtin", "optional", "anomalib"),
    metadata={
        "description": "Skeleton recipe for anomalib training (requires `pyimgano[anomalib]`).",
        "requires_extra": "anomalib",
    },
)
def anomalib_train(config: WorkbenchConfig) -> dict[str, Any]:
    _ = config
    try:
        import anomalib  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - dependency boundary
        raise ImportError(
            "The `anomalib-train` recipe requires anomalib.\n"
            "Install it via:\n"
            "  pip install 'pyimgano[anomalib]'"
        ) from exc

    raise NotImplementedError(
        "The anomalib training recipe is a placeholder in this milestone.\n"
        "For now, train models with anomalib directly and use pyimgano for evaluation/inference.\n"
        "Follow-up milestones will integrate anomalib training end-to-end."
    )

