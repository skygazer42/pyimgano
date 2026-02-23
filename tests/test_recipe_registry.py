import pytest

from pyimgano.recipes.registry import RECIPE_REGISTRY, list_recipes, recipe_info
from pyimgano.workbench.config import WorkbenchConfig


def test_recipe_registry_register_and_discover():
    def _recipe(config: WorkbenchConfig):  # noqa: ANN001
        return {"recipe": "ok", "dataset": config.dataset.name}

    RECIPE_REGISTRY.register(
        "test_recipe_registry_ok",
        _recipe,
        tags=["unit", "test"],
        metadata={"description": "test"},
        overwrite=True,
    )

    assert "test_recipe_registry_ok" in list_recipes()
    assert "test_recipe_registry_ok" in list_recipes(tags=["unit"])
    assert "test_recipe_registry_ok" not in list_recipes(tags=["missing"])

    info = recipe_info("test_recipe_registry_ok")
    assert info["name"] == "test_recipe_registry_ok"
    assert "unit" in info["tags"]
    assert info["metadata"]["description"] == "test"


def test_recipe_registry_duplicate_requires_overwrite():
    def _recipe(config: WorkbenchConfig):  # noqa: ANN001
        return {"ok": True}

    RECIPE_REGISTRY.register(
        "test_recipe_registry_no_overwrite",
        _recipe,
        overwrite=True,
    )

    with pytest.raises(KeyError):
        RECIPE_REGISTRY.register(
            "test_recipe_registry_no_overwrite",
            _recipe,
            overwrite=False,
        )

