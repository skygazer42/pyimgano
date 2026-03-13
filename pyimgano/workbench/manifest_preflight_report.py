from __future__ import annotations

from typing import Any


def build_manifest_preflight_report(
    *,
    manifest_path: str,
    root_fallback: str | None,
    policy: Any,
    categories: list[str],
    per_category: dict[str, Any],
    requested_all: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "root_fallback": (str(root_fallback) if root_fallback is not None else None),
        "split_policy": {
            "mode": str(policy.mode),
            "scope": str(policy.scope),
            "seed": int(policy.seed),
            "test_normal_fraction": float(policy.test_normal_fraction),
        },
        "categories": categories,
        "per_category": per_category if requested_all else None,
    }
    if not requested_all and categories:
        report.update(per_category.get(str(categories[0]), {}))
    report["manifest"] = {"ok": True}
    return report


__all__ = ["build_manifest_preflight_report"]
