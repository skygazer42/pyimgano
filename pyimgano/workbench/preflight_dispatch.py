from __future__ import annotations

from typing import Any


def resolve_preflight_dataset_dispatch(*, config: Any) -> str:
    dataset = str(config.dataset.name)
    return "manifest" if dataset.lower() == "manifest" else "non_manifest"


__all__ = ["resolve_preflight_dataset_dispatch"]
