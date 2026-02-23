from __future__ import annotations

from .protocol import Recipe
from .registry import RECIPE_REGISTRY, list_recipes, recipe_info, register_recipe

__all__ = ["RECIPE_REGISTRY", "Recipe", "list_recipes", "recipe_info", "register_recipe"]

# Builtin recipes (side-effect registration).
from pyimgano.recipes.builtin import industrial_adapt  # noqa: E402,F401
