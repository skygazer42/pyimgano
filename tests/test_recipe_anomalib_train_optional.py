import pytest

from pyimgano.recipes.registry import RECIPE_REGISTRY
from pyimgano.workbench.config import WorkbenchConfig


def test_recipe_anomalib_train_optional():
    pytest.importorskip("anomalib")

    # Ensure builtin recipes are registered.
    import pyimgano.recipes  # noqa: F401

    recipe = RECIPE_REGISTRY.get("anomalib-train")
    cfg = WorkbenchConfig.from_dict(
        {
            "recipe": "anomalib-train",
            "dataset": {"name": "custom", "root": "/tmp/data"},
            "model": {"name": "vision_patchcore"},
            "output": {"save_run": False},
        }
    )
    with pytest.raises(NotImplementedError):
        recipe(cfg)

