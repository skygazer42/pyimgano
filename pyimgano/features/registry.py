"""Feature extractor registry.

The model registry in `pyimgano.models` is the primary public entrypoint for detectors.
For classical workflows it is also useful to discover and construct reusable
feature extractors (HOG/LBP/ColorHist/...), so we provide a small registry with a
similar API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from collections.abc import Mapping


@dataclass
class FeatureEntry:
    name: str
    constructor: Callable[..., Any]
    tags: tuple[str, ...]
    metadata: Dict[str, Any]


class FeatureRegistry:
    """Registry for feature extractor constructors."""

    def __init__(self) -> None:
        self._registry: Dict[str, FeatureEntry] = {}

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
                f"Feature extractor {name!r} already exists. Set overwrite=True to replace it."
            )
        self._registry[name] = FeatureEntry(
            name=str(name),
            constructor=constructor,
            tags=tuple(tags or ()),
            metadata=metadata or {},
        )

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name].constructor
        except KeyError as exc:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(
                f"Feature extractor {name!r} not found. Available extractors: {available}"
            ) from exc

    def available(self, *, tags: Optional[Iterable[str]] = None) -> List[str]:
        if tags is None:
            return sorted(self._registry)
        tag_set = set(str(t) for t in tags)
        return sorted(
            entry.name for entry in self._registry.values() if tag_set.issubset(entry.tags)
        )

    def info(self, name: str) -> FeatureEntry:
        if name not in self._registry:
            raise KeyError(f"Feature extractor {name!r} not found in registry")
        return self._registry[name]

    def feature_info(self, name: str) -> Dict[str, Any]:
        from pyimgano.models.introspection import get_constructor_signature_info

        entry = self.info(name)
        signature, accepted, accepts_var_kwargs = get_constructor_signature_info(entry.constructor)
        return {
            "name": entry.name,
            "tags": list(entry.tags),
            "metadata": dict(entry.metadata),
            "signature": str(signature),
            "accepted_kwargs": sorted(accepted),
            "accepts_var_kwargs": bool(accepts_var_kwargs),
            "constructor": {
                "module": getattr(entry.constructor, "__module__", "<unknown>"),
                "qualname": getattr(entry.constructor, "__qualname__", "<unknown>"),
            },
        }


FEATURE_REGISTRY = FeatureRegistry()


def register_feature_extractor(
    name: str,
    *,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for registering a feature extractor constructor."""

    def decorator(constructor: Callable[..., Any]) -> Callable[..., Any]:
        FEATURE_REGISTRY.register(
            name,
            constructor,
            tags=tags,
            metadata=metadata,
            overwrite=overwrite,
        )
        return constructor

    return decorator


def create_feature_extractor(name: str, *args, **kwargs):
    """Create a feature extractor instance from its registered name."""

    ctor = FEATURE_REGISTRY.get(name)
    return ctor(*args, **kwargs)


def list_feature_extractors(*, tags: Optional[Iterable[str]] = None) -> List[str]:
    """List available feature extractor names, optionally filtered by tags."""

    return FEATURE_REGISTRY.available(tags=tags)


def feature_info(name: str) -> Dict[str, Any]:
    """Return a stable, JSON-friendly feature extractor info payload."""

    return FEATURE_REGISTRY.feature_info(name)


def resolve_feature_extractor(spec: Any):
    """Resolve a feature extractor from a JSON-friendly spec.

    Accepted inputs:
    - None -> None
    - An object with a callable `.extract` -> returned unchanged
    - A string -> treated as a registered extractor name
    - A mapping -> {"name": "<registered_name>", "kwargs": {...}}
    """

    if spec is None:
        return None

    # Already an extractor-like object.
    extract = getattr(spec, "extract", None)
    if callable(extract):
        return spec

    if isinstance(spec, str):
        return create_feature_extractor(spec)

    if isinstance(spec, Mapping):
        name = spec.get("name", None) or spec.get("type", None)
        if name is None:
            raise ValueError("feature_extractor spec must include 'name'")
        kwargs = spec.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise ValueError("feature_extractor.kwargs must be an object/dict")
        return create_feature_extractor(str(name), **dict(kwargs))

    raise TypeError(
        "feature_extractor must be an extractor object, a registered name string, "
        "or a spec dict {'name': ..., 'kwargs': {...}}"
    )
