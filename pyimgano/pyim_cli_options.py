from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyimgano.pyim_contracts as pyim_contracts
from pyimgano.discovery import resolve_family_tags, resolve_type_tags, resolve_year_filter
from pyimgano.pyim_list_spec import PYIM_LIST_KIND_CHOICES, get_pyim_list_kind_spec


@dataclass(frozen=True)
class PyimListOptions:
    list_kind: str
    tags: Any
    family: str | None
    algorithm_type: str | None
    year: str | None
    deployable_only: bool
    objective: str | None = None
    selection_profile: str | None = None
    topk: int | None = None

    @property
    def include_core_sections(self) -> bool:
        return get_pyim_list_kind_spec(self.list_kind).include_core_sections

    @property
    def include_recipes(self) -> bool:
        return get_pyim_list_kind_spec(self.list_kind).include_recipes

    @property
    def include_datasets(self) -> bool:
        return get_pyim_list_kind_spec(self.list_kind).include_datasets

    def to_request(self) -> pyim_contracts.PyimListRequest:
        return pyim_contracts.PyimListRequest(
            list_kind=self.list_kind,
            tags=self.tags,
            family=self.family,
            algorithm_type=self.algorithm_type,
            year=self.year,
            deployable_only=self.deployable_only,
            objective=self.objective,
            selection_profile=self.selection_profile,
            topk=self.topk,
        )


_ALLOWED_OBJECTIVES = {"balanced", "latency", "localization"}
_ALLOWED_SELECTION_PROFILES = {
    "balanced",
    "benchmark-parity",
    "cpu-screening",
    "deploy-readiness",
}


def _normalize_objective(objective: Any) -> str | None:
    if objective is None:
        return None
    key = str(objective).strip().lower()
    if key not in _ALLOWED_OBJECTIVES:
        raise ValueError(f"--objective must be one of: {', '.join(sorted(_ALLOWED_OBJECTIVES))}.")
    return key


def _normalize_selection_profile(selection_profile: Any) -> str | None:
    if selection_profile is None:
        return None
    key = str(selection_profile).strip().lower()
    if key not in _ALLOWED_SELECTION_PROFILES:
        raise ValueError(
            "--selection-profile must be one of: "
            f"{', '.join(sorted(_ALLOWED_SELECTION_PROFILES))}."
        )
    return key


def _normalize_topk(topk: Any) -> int | None:
    if topk is None:
        return None
    value = int(topk)
    if value < 1:
        raise ValueError("--topk must be >= 1.")
    return value


def resolve_pyim_list_options(
    *,
    list_kind: Any,
    tags: Any,
    family: Any = None,
    algorithm_type: Any = None,
    year: Any = None,
    deployable_only: bool = False,
    objective: Any = None,
    selection_profile: Any = None,
    topk: Any = None,
) -> PyimListOptions:
    list_kind_value = "all" if list_kind is None else str(list_kind)
    kind_spec = get_pyim_list_kind_spec(list_kind_value)
    family_value = str(family) if family is not None else None
    algorithm_type_value = str(algorithm_type) if algorithm_type is not None else None
    year_value = str(year) if year is not None else None
    deployable_only_value = bool(deployable_only)
    objective_value = _normalize_objective(objective)
    selection_profile_value = _normalize_selection_profile(selection_profile)
    topk_value = _normalize_topk(topk)

    if family_value is not None and not kind_spec.supports_family_filter:
        raise ValueError(
            "--family is supported only with --list models, --list model-presets, or --list."
        )
    if algorithm_type_value is not None and not kind_spec.supports_algorithm_type_filter:
        raise ValueError("--type is supported only with --list models.")
    if year_value is not None and not kind_spec.supports_year_filter:
        raise ValueError("--year is supported only with --list models.")
    if deployable_only_value and not kind_spec.supports_deployable_only:
        raise ValueError("--deployable-only is supported only with --list preprocessing or --list.")
    if objective_value is not None and list_kind_value != "models":
        raise ValueError("--objective is supported only with --list models.")
    if selection_profile_value is not None and list_kind_value != "models":
        raise ValueError("--selection-profile is supported only with --list models.")
    if topk_value is not None and list_kind_value != "models":
        raise ValueError("--topk is supported only with --list models.")

    if family_value is not None:
        resolve_family_tags(family_value)
    if algorithm_type_value is not None:
        resolve_type_tags(algorithm_type_value)
    if year_value is not None:
        resolve_year_filter(year_value)

    return PyimListOptions(
        list_kind=list_kind_value,
        tags=tags,
        family=family_value,
        algorithm_type=algorithm_type_value,
        year=year_value,
        deployable_only=deployable_only_value,
        objective=objective_value,
        selection_profile=selection_profile_value,
        topk=topk_value,
    )


__all__ = [
    "PYIM_LIST_KIND_CHOICES",
    "PyimListOptions",
    "resolve_pyim_list_options",
]
