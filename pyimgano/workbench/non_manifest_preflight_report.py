from __future__ import annotations

from typing import Any


def build_non_manifest_preflight_report(*, root: str, categories: list[str]) -> dict[str, Any]:
    return {
        "dataset_root": str(root),
        "categories": categories,
        "ok": True,
    }


__all__ = ["build_non_manifest_preflight_report"]
