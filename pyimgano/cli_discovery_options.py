from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from pyimgano.discovery import resolve_family_tags, resolve_type_tags, resolve_year_filter


@dataclass(frozen=True)
class ModelListDiscoveryOptions:
    tags: Any
    family: str | None
    algorithm_type: str | None
    year: str | None


def _format_flag_names(names: Sequence[str]) -> str:
    items = [str(name) for name in names]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def validate_mutually_exclusive_flags(flags: Iterable[tuple[str, bool]]) -> None:
    normalized_flags = [(str(name), bool(active)) for name, active in flags]
    if sum(1 for _name, active in normalized_flags if active) > 1:
        raise ValueError(f"{_format_flag_names([name for name, _ in normalized_flags])} are mutually exclusive.")


def resolve_model_list_discovery_options(
    *,
    list_models: bool,
    tags: Any,
    family: Any = None,
    algorithm_type: Any = None,
    year: Any = None,
    allow_family_without_list_models: bool = False,
) -> ModelListDiscoveryOptions:
    family_value = str(family) if family is not None else None
    algorithm_type_value = str(algorithm_type) if algorithm_type is not None else None
    year_value = str(year) if year is not None else None

    if family_value is not None and not bool(list_models) and not bool(
        allow_family_without_list_models
    ):
        raise ValueError("--family is supported only with --list-models.")
    if algorithm_type_value is not None and not bool(list_models):
        raise ValueError("--type is supported only with --list-models.")
    if year_value is not None and not bool(list_models):
        raise ValueError("--year is supported only with --list-models.")

    if family_value is not None:
        resolve_family_tags(family_value)
    if algorithm_type_value is not None:
        resolve_type_tags(algorithm_type_value)
    if year_value is not None:
        resolve_year_filter(year_value)

    return ModelListDiscoveryOptions(
        tags=tags,
        family=family_value,
        algorithm_type=algorithm_type_value,
        year=year_value,
    )
