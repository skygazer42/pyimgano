from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from pyimgano.recipes.protocol import Recipe


@dataclass
class RecipeEntry:
    name: str
    recipe: Recipe
    tags: tuple[str, ...]
    metadata: Dict[str, Any]


class RecipeRegistry:
    """Registry for storing recipes with metadata."""

    def __init__(self) -> None:
        self._registry: Dict[str, RecipeEntry] = {}

    def register(
        self,
        name: str,
        recipe: Recipe,
        *,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._registry:
            raise KeyError(f"Recipe {name!r} already exists. Set overwrite=True to replace it.")
        entry = RecipeEntry(
            name=str(name),
            recipe=recipe,
            tags=tuple(str(t) for t in (tags or ())),
            metadata=dict(metadata or {}),
        )
        self._registry[str(name)] = entry

    def get(self, name: str) -> Recipe:
        try:
            return self._registry[str(name)].recipe
        except KeyError as exc:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(
                f"Recipe {name!r} not found. Available recipes: {available}"
            ) from exc

    def available(self, *, tags: Optional[Iterable[str]] = None) -> List[str]:
        if tags is None:
            return sorted(self._registry)
        tag_set = {str(t) for t in tags}
        return sorted(
            entry.name for entry in self._registry.values() if tag_set.issubset(entry.tags)
        )

    def info(self, name: str) -> RecipeEntry:
        if str(name) not in self._registry:
            raise KeyError(f"Recipe {name!r} not found in registry")
        return self._registry[str(name)]

    def recipe_info(self, name: str) -> Dict[str, Any]:
        entry = self.info(name)
        recipe_obj = entry.recipe
        callable_path = None
        module = getattr(recipe_obj, "__module__", None)
        qualname = getattr(recipe_obj, "__qualname__", None)
        if module and qualname:
            callable_path = f"{module}.{qualname}"
        else:
            callable_path = f"{type(recipe_obj).__module__}.{type(recipe_obj).__qualname__}"
        return {
            "name": entry.name,
            "tags": list(entry.tags),
            "metadata": dict(entry.metadata),
            "callable": callable_path,
        }


RECIPE_REGISTRY = RecipeRegistry()


def register_recipe(
    name: str,
    *,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable[[Recipe], Recipe]:
    def decorator(recipe: Recipe) -> Recipe:
        RECIPE_REGISTRY.register(
            name,
            recipe,
            tags=tags,
            metadata=metadata,
            overwrite=overwrite,
        )
        return recipe

    return decorator


def list_recipes(*, tags: Optional[Iterable[str]] = None) -> List[str]:
    return RECIPE_REGISTRY.available(tags=tags)


def recipe_info(name: str) -> Dict[str, Any]:
    return RECIPE_REGISTRY.recipe_info(name)
