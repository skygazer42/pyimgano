from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pyimgano.workbench.config import WorkbenchConfig


def resolve_manifest_preflight_source(
    *,
    config: WorkbenchConfig,
    issues: list[Any],
    issue_builder: Callable[..., Any],
) -> dict[str, Any]:
    if str(config.dataset.input_mode) != "paths":
        issues.append(
            issue_builder(
                "MANIFEST_UNSUPPORTED_INPUT_MODE",
                "error",
                "dataset.name='manifest' supports only dataset.input_mode='paths'.",
                context={"input_mode": str(config.dataset.input_mode)},
            )
        )

    manifest_path_raw = config.dataset.manifest_path
    if manifest_path_raw is None:
        issues.append(
            issue_builder(
                "MANIFEST_PATH_MISSING",
                "error",
                "dataset.manifest_path is required when dataset.name='manifest'.",
            )
        )
        return {
            "manifest_path": None,
            "root_fallback": None,
            "summary": {"manifest": {"ok": False}},
        }

    manifest_path = Path(str(manifest_path_raw))
    if not manifest_path.exists():
        issues.append(
            issue_builder(
                "MANIFEST_NOT_FOUND",
                "error",
                "Manifest file not found.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return {
            "manifest_path": manifest_path,
            "root_fallback": None,
            "summary": {"manifest_path": str(manifest_path), "manifest": {"ok": False}},
        }
    if not manifest_path.is_file():
        issues.append(
            issue_builder(
                "MANIFEST_NOT_A_FILE",
                "error",
                "Manifest path must be a file.",
                context={"manifest_path": str(manifest_path)},
            )
        )
        return {
            "manifest_path": manifest_path,
            "root_fallback": None,
            "summary": {"manifest_path": str(manifest_path), "manifest": {"ok": False}},
        }

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            handle.read(1)
    except Exception as exc:  # noqa: BLE001 - validation boundary
        issues.append(
            issue_builder(
                "MANIFEST_NOT_READABLE",
                "error",
                "Manifest file is not readable.",
                context={"manifest_path": str(manifest_path), "error": str(exc)},
            )
        )
        return {
            "manifest_path": manifest_path,
            "root_fallback": None,
            "summary": {"manifest_path": str(manifest_path), "manifest": {"ok": False}},
        }

    root_fallback = Path(str(config.dataset.root)) if config.dataset.root is not None else None
    if root_fallback is not None and not root_fallback.exists():
        issues.append(
            issue_builder(
                "DATASET_ROOT_MISSING",
                "warning",
                "dataset.root does not exist; root fallback will not be used for resolving relative paths.",
                context={"root": str(root_fallback)},
            )
        )

    return {
        "manifest_path": manifest_path,
        "root_fallback": root_fallback,
        "summary": None,
    }


__all__ = ["resolve_manifest_preflight_source"]
