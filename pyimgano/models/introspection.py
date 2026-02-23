from __future__ import annotations

import inspect
from typing import Any, Callable, Mapping, Protocol, Sequence


class _ModelEntryLike(Protocol):
    name: str
    constructor: Callable[..., Any]
    tags: Sequence[str]
    metadata: Mapping[str, Any]


class _ModelRegistryLike(Protocol):
    def info(self, name: str) -> _ModelEntryLike: ...


def get_constructor_signature_info(
    constructor: Callable[..., Any],
) -> tuple[inspect.Signature, set[str], bool]:
    """Return (signature, accepted_kwarg_names, accepts_var_kwargs) for a constructor."""

    signature = inspect.signature(constructor)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    accepted = {
        name
        for name, p in signature.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return signature, accepted, accepts_var_kwargs


def model_entry_info(entry: _ModelEntryLike) -> dict[str, Any]:
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


def model_info(registry: _ModelRegistryLike, name: str) -> dict[str, Any]:
    return model_entry_info(registry.info(name))

