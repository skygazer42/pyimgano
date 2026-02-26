"""
Model registry system inspired by torchvision and timm.

This module provides a centralized registry for model constructors,
enabling dynamic model creation and discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class ModelEntry:
    name: str
    constructor: Callable[..., Any]
    tags: tuple[str, ...]
    metadata: Dict[str, Any]


class ModelRegistry:
    """Registry for storing model constructors with metadata."""

    def __init__(self) -> None:
        self._registry: Dict[str, ModelEntry] = {}

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        constructor: Callable[..., Any],
        *,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._registry:
            raise KeyError(
                f"Model {name!r} already exists. Set overwrite=True to replace it."
            )
        entry = ModelEntry(
            name=name,
            constructor=constructor,
            tags=tuple(tags or ()),
            metadata=metadata or {},
        )
        self._registry[name] = entry

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name].constructor
        except KeyError as exc:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(f"Model {name!r} not found. Available models: {available}") from exc

    def available(self, *, tags: Optional[Iterable[str]] = None) -> List[str]:
        if tags is None:
            return sorted(self._registry)
        tag_set = set(tags)
        return sorted(
            entry.name for entry in self._registry.values() if tag_set.issubset(entry.tags)
        )

    def info(self, name: str) -> ModelEntry:
        if name not in self._registry:
            raise KeyError(f"Model {name!r} not found in registry")
        return self._registry[name]

    def model_info(self, name: str) -> Dict[str, Any]:
        from pyimgano.models.introspection import model_entry_info

        entry = self.info(name)
        return model_entry_info(entry)


MODEL_REGISTRY = ModelRegistry()


def register_model(
    name: str,
    *,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for automatically registering models at import time.

    Parameters
    ----------
    name : str
        Unique name for the model
    tags : Iterable[str], optional
        Tags for categorizing the model (e.g., 'vision', 'classical', 'deep')
    metadata : Dict[str, Any], optional
        Additional metadata (e.g., description, paper, year)
    overwrite : bool, default=False
        Whether to overwrite existing registration

    Returns
    -------
    Callable
        Decorator function

    Examples
    --------
    >>> @register_model("my_model", tags=["vision", "ml"])
    ... class MyModel:
    ...     pass
    """

    def decorator(constructor: Callable[..., Any]) -> Callable[..., Any]:
        MODEL_REGISTRY.register(
            name,
            constructor,
            tags=tags,
            metadata=metadata,
            overwrite=overwrite,
        )
        return constructor

    return decorator


def create_model(name: str, *args, **kwargs):
    """
    Create a model instance from its registered name.

    Parameters
    ----------
    name : str
        Registered model name
    *args
        Positional arguments to pass to the model constructor
    **kwargs
        Keyword arguments to pass to the model constructor

    Returns
    -------
    model
        Instantiated model object

    Examples
    --------
    >>> model = create_model("vision_ecod", contamination=0.1)
    """
    constructor = MODEL_REGISTRY.get(name)
    return constructor(*args, **kwargs)


def list_models(*, tags: Optional[Iterable[str]] = None) -> List[str]:
    """
    List available model names, optionally filtered by tags.

    Parameters
    ----------
    tags : Iterable[str], optional
        Filter models by tags (only models with all specified tags are returned)

    Returns
    -------
    List[str]
        Sorted list of model names

    Examples
    --------
    >>> all_models = list_models()
    >>> classical_models = list_models(tags=["classical"])
    >>> vision_ml_models = list_models(tags=["vision", "ml"])
    """
    return MODEL_REGISTRY.available(tags=tags)


def model_info(name: str) -> Dict[str, Any]:
    """Return a stable, JSON-friendly model info payload.

    Notes
    -----
    `ModelRegistry.model_info()` is intentionally a minimal, generic
    introspection payload. This module-level helper enriches the payload with
    computed capabilities for convenience in CLI/workbench discovery.
    """

    from pyimgano.models.capabilities import compute_model_capabilities
    from pyimgano.models.introspection import model_entry_info

    entry = MODEL_REGISTRY.info(name)
    payload = model_entry_info(entry)

    caps = compute_model_capabilities(entry)
    caps_payload = {
        "input_modes": list(caps.input_modes),
        "supports_pixel_map": bool(caps.supports_pixel_map),
        "supports_checkpoint": bool(caps.supports_checkpoint),
        "requires_checkpoint": bool(caps.requires_checkpoint),
        "supports_save_load": bool(caps.supports_save_load),
    }
    payload["capabilities"] = caps_payload
    # Convenience top-level aliases (stable for CLI output).
    payload["input_modes"] = list(caps.input_modes)
    payload["supports_pixel_map"] = bool(caps.supports_pixel_map)
    payload["supports_checkpoint"] = bool(caps.supports_checkpoint)
    payload["requires_checkpoint"] = bool(caps.requires_checkpoint)
    payload["supports_save_load"] = bool(caps.supports_save_load)
    return payload
