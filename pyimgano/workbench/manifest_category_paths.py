from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from pyimgano.workbench.manifest_record_preflight import resolve_manifest_path_best_effort


def inspect_manifest_category_paths(
    *,
    category: str,
    records: Sequence[Any],
    manifest_path: Path,
    root_fallback: Path | None,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[int, bool]:
    image_seen: dict[str, int] = {}
    mask_exists_by_index: dict[int, bool] = {}

    for idx, record in enumerate(records):
        resolved, exists, source = resolve_manifest_path_best_effort(
            record.image_path, manifest_path=manifest_path, root_fallback=root_fallback
        )
        image_seen[resolved] = int(image_seen.get(resolved, 0)) + 1
        if not exists:
            issues.append(
                issue_builder(
                    "MANIFEST_MISSING_IMAGE",
                    "error",
                    "Manifest image_path does not exist.",
                    context={
                        "category": str(category),
                        "image_path": str(record.image_path),
                        "resolved": str(resolved),
                        "resolution": str(source),
                    },
                )
            )

        if record.mask_path is None:
            continue

        mask_resolved, mask_exists, mask_source = resolve_manifest_path_best_effort(
            record.mask_path, manifest_path=manifest_path, root_fallback=root_fallback
        )
        mask_exists_by_index[int(idx)] = bool(mask_exists)
        if not mask_exists:
            issues.append(
                issue_builder(
                    "MANIFEST_MISSING_MASK",
                    "warning",
                    "Manifest mask_path does not exist.",
                    context={
                        "category": str(category),
                        "mask_path": str(record.mask_path),
                        "resolved": str(mask_resolved),
                        "resolution": str(mask_source),
                    },
                )
            )

    for path, count in sorted(image_seen.items()):
        if int(count) > 1:
            issues.append(
                issue_builder(
                    "MANIFEST_DUPLICATE_IMAGE",
                    "warning",
                    "Duplicate image_path detected within category.",
                    context={"category": str(category), "resolved": str(path), "count": int(count)},
                )
            )

    return mask_exists_by_index


__all__ = ["inspect_manifest_category_paths"]
