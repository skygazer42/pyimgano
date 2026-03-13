"""Lazy compatibility facade for workbench workflows and config types."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_WORKBENCH_EXPORT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "config",
        (
            ("SplitPolicyConfig", "pyimgano.workbench.config"),
            ("DatasetConfig", "pyimgano.workbench.config"),
            ("ModelConfig", "pyimgano.workbench.config"),
            ("OutputConfig", "pyimgano.workbench.config"),
            ("TrainingConfig", "pyimgano.workbench.config"),
            ("DefectsConfig", "pyimgano.workbench.config"),
            ("PreprocessingConfig", "pyimgano.workbench.config"),
            ("WorkbenchConfig", "pyimgano.workbench.config"),
        ),
    ),
    (
        "dataset",
        (
            ("WorkbenchSplit", "pyimgano.workbench.dataset_loader"),
            ("list_workbench_categories", "pyimgano.workbench.dataset_loader"),
            ("load_workbench_split", "pyimgano.workbench.dataset_loader"),
        ),
    ),
    (
        "preflight",
        (
            ("PreflightIssue", "pyimgano.workbench.preflight_types"),
            ("PreflightReport", "pyimgano.workbench.preflight_types"),
            ("run_preflight", "pyimgano.workbench.preflight"),
        ),
    ),
    (
        "execution",
        (
            ("build_infer_config_payload", "pyimgano.workbench.runner"),
            ("run_workbench", "pyimgano.workbench.runner"),
        ),
    ),
)


def _iter_workbench_export_items() -> list[tuple[str, str]]:
    return [item for _group_name, items in _WORKBENCH_EXPORT_GROUPS for item in items]


def _build_workbench_export_sources() -> dict[str, str]:
    sources: dict[str, str] = {}
    for export_name, module_name in _iter_workbench_export_items():
        if export_name in sources:
            raise ValueError(f"Duplicate workbench root export declared: {export_name}")
        sources[export_name] = module_name
    return sources


_WORKBENCH_EXPORT_SOURCES = _build_workbench_export_sources()


__all__ = list(_WORKBENCH_EXPORT_SOURCES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _WORKBENCH_EXPORT_SOURCES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} could not resolve export {name!r}") from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
