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
        )


def resolve_pyim_list_options(
    *,
    list_kind: Any,
    tags: Any,
    family: Any = None,
    algorithm_type: Any = None,
    year: Any = None,
    deployable_only: bool = False,
) -> PyimListOptions:
    list_kind_value = "all" if list_kind is None else str(list_kind)
    kind_spec = get_pyim_list_kind_spec(list_kind_value)
    family_value = str(family) if family is not None else None
    algorithm_type_value = str(algorithm_type) if algorithm_type is not None else None
    year_value = str(year) if year is not None else None
    deployable_only_value = bool(deployable_only)

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
    )


__all__ = [
    "PYIM_LIST_KIND_CHOICES",
    "PyimListOptions",
    "resolve_pyim_list_options",
]
