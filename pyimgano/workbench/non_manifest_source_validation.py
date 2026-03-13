from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig


def resolve_non_manifest_preflight_source(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    dataset = str(config.dataset.name)
    root = Path(str(config.dataset.root))

    if not root.exists():
        issues.append(
            issue_builder(
                "DATASET_ROOT_MISSING",
                "error",
                "Dataset root does not exist.",
                context={"dataset": dataset, "root": str(root)},
            )
        )
        return {
            "dataset": dataset,
            "root": root,
            "summary": {"dataset_root": str(root), "ok": False},
        }

    if dataset.lower() == "custom":
        try:
            from pyimgano.utils.datasets import CustomDataset

            CustomDataset(root=str(root), load_masks=True).validate_structure()
        except Exception as exc:  # noqa: BLE001 - validation boundary
            issues.append(
                issue_builder(
                    "CUSTOM_DATASET_INVALID_STRUCTURE",
                    "error",
                    "Custom dataset layout validation failed.",
                    context={"root": str(root), "error": str(exc)},
                )
            )

    return {
        "dataset": dataset,
        "root": root,
        "summary": None,
    }


__all__ = ["resolve_non_manifest_preflight_source"]
