"""Lazy compatibility facade for workbench workflows and config types."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_WORKBENCH_CONFIG_MODULE = "pyimgano.workbench.config"
_WORKBENCH_DATASET_LOADER_MODULE = "pyimgano.workbench.dataset_loader"
_WORKBENCH_PREFLIGHT_MODULE = "pyimgano.workbench.preflight"
_WORKBENCH_PREFLIGHT_TYPES_MODULE = "pyimgano.workbench.preflight_types"
_WORKBENCH_RUNNER_MODULE = "pyimgano.workbench.runner"


_WORKBENCH_EXPORT_GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "config",
        (
            ("SplitPolicyConfig", _WORKBENCH_CONFIG_MODULE),
            ("DatasetConfig", _WORKBENCH_CONFIG_MODULE),
            ("ModelConfig", _WORKBENCH_CONFIG_MODULE),
            ("OutputConfig", _WORKBENCH_CONFIG_MODULE),
            ("TrainingConfig", _WORKBENCH_CONFIG_MODULE),
            ("DefectsConfig", _WORKBENCH_CONFIG_MODULE),
            ("PreprocessingConfig", _WORKBENCH_CONFIG_MODULE),
            ("WorkbenchConfig", _WORKBENCH_CONFIG_MODULE),
        ),
    ),
    (
        "dataset",
        (
            ("WorkbenchSplit", _WORKBENCH_DATASET_LOADER_MODULE),
            ("list_workbench_categories", _WORKBENCH_DATASET_LOADER_MODULE),
            ("load_workbench_split", _WORKBENCH_DATASET_LOADER_MODULE),
        ),
    ),
    (
        "preflight",
        (
            ("PreflightIssue", _WORKBENCH_PREFLIGHT_TYPES_MODULE),
            ("PreflightReport", _WORKBENCH_PREFLIGHT_TYPES_MODULE),
            ("run_preflight", _WORKBENCH_PREFLIGHT_MODULE),
        ),
    ),
    (
        "execution",
        (
            ("build_infer_config_payload", _WORKBENCH_RUNNER_MODULE),
            ("run_workbench", _WORKBENCH_RUNNER_MODULE),
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
