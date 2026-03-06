# -*- coding: utf-8 -*-
"""Utilities to resolve JSON-friendly model specs into detector instances.

Why
---
Industrial deployments often want configs like:

```json
{
  "name": "vision_score_ensemble",
  "kwargs": {
    "detectors": [
      {"name": "vision_ecod", "kwargs": {"feature_extractor": "hog"}},
      {"name": "vision_knn", "kwargs": {"feature_extractor": "hog", "n_neighbors": 7}}
    ]
  }
}
```

This module keeps the spec format small and stable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

from .registry import create_model


def resolve_model_spec(
    spec: Any,
    *,
    default_contamination: Optional[float] = None,
) -> Any:
    """Resolve a model spec into a detector instance.

    Accepted forms:
    - Already-instantiated detector (has .decision_function) -> returned as-is
    - String model name -> `create_model(name, contamination=default_contamination)`
    - Mapping: {"name": "...", "kwargs": {...}} (also accepts "type"/"model")
      If mapping contains "contamination", it overrides default.
    """

    if spec is None:
        raise ValueError("model spec cannot be None")

    if callable(getattr(spec, "decision_function", None)):
        return spec

    if isinstance(spec, str):
        kwargs: dict[str, Any] = {}
        if default_contamination is not None:
            kwargs["contamination"] = float(default_contamination)
        return create_model(str(spec), **kwargs)

    if isinstance(spec, Mapping):
        name = spec.get("name", None) or spec.get("type", None) or spec.get("model", None)
        if name is None:
            raise ValueError("model spec dict must include 'name'")
        kwargs = spec.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, Mapping):
            raise ValueError("model spec 'kwargs' must be an object/dict")

        merged: dict[str, Any] = dict(kwargs)
        if "contamination" in spec:
            merged["contamination"] = float(spec["contamination"])
        elif default_contamination is not None and "contamination" not in merged:
            merged["contamination"] = float(default_contamination)

        return create_model(str(name), **merged)

    raise TypeError(
        "model spec must be a detector instance, a model name string, "
        "or a dict {'name': ..., 'kwargs': {...}}"
    )


def resolve_model_specs(
    specs: Sequence[Any],
    *,
    default_contamination: Optional[float] = None,
) -> list[Any]:
    if specs is None:
        raise ValueError("specs cannot be None")
    out: list[Any] = []
    for s in list(specs):
        out.append(resolve_model_spec(s, default_contamination=default_contamination))
    return out
