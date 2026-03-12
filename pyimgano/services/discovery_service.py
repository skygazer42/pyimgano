from __future__ import annotations

from typing import Any, Iterable

from pyimgano.presets.catalog import (
    list_model_preset_infos,
    list_model_presets,
    model_preset_info,
    resolve_model_preset_filter_tags,
    resolve_model_preset,
)


def list_discovery_model_names(
    *,
    tags: Iterable[str] | None = None,
    family: str | None = None,
    algorithm_type: str | None = None,
    year: str | int | None = None,
) -> list[str]:
    from pyimgano.discovery import list_model_names

    return list_model_names(
        tags=tags,
        family=family,
        algorithm_type=algorithm_type,
        year=year,
    )


def list_discovery_feature_names(*, tags: Iterable[str] | None = None) -> list[str]:
    from pyimgano.discovery import list_feature_names

    return list_feature_names(tags=tags)


def list_dataset_categories_payload(
    *,
    dataset: str,
    root: str,
    manifest_path: str | None = None,
) -> list[str]:
    from pyimgano.datasets.catalog import list_dataset_categories

    return list_dataset_categories(dataset=dataset, root=root, manifest_path=manifest_path)


def list_baseline_suites_payload() -> list[str]:
    from pyimgano.baselines import list_baseline_suites

    return list_baseline_suites()


def build_suite_info_payload(name: str) -> dict[str, Any]:
    from pyimgano.baselines import get_baseline_suite, resolve_suite_baselines

    suite = get_baseline_suite(str(name))
    baselines = resolve_suite_baselines(str(name))
    return {
        "name": str(suite.name),
        "description": str(suite.description),
        "entries": list(suite.entries),
        "baselines": [
            {
                "name": str(b.name),
                "model": str(b.model),
                "optional": bool(b.optional),
                "requires_extras": list(getattr(b, "requires_extras", ())),
                "description": str(b.description),
                "kwargs": dict(b.kwargs),
            }
            for b in baselines
        ],
    }


def list_sweeps_payload() -> list[str]:
    from pyimgano.baselines.sweeps import list_sweeps

    return list_sweeps()


def build_sweep_info_payload(name: str) -> dict[str, Any]:
    from pyimgano.baselines.sweeps import resolve_sweep

    plan = resolve_sweep(str(name))
    variants_by_entry: dict[str, list[dict[str, Any]]] = {}
    for entry_name in sorted(plan.variants_by_entry.keys()):
        variants_by_entry[str(entry_name)] = [
            {
                "name": str(v.name),
                "description": str(v.description),
                "override": dict(v.override),
            }
            for v in plan.variants_by_entry[entry_name]
        ]

    return {
        "name": str(plan.name),
        "description": str(plan.description),
        "entries": sorted(variants_by_entry.keys()),
        "variants_by_entry": variants_by_entry,
    }


def build_model_info_payload(name: str) -> dict[str, Any]:
    from pyimgano.models.registry import model_info, materialize_model_constructor

    model_name = str(name)
    try:
        materialize_model_constructor(model_name)
        return model_info(model_name)
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_name!r}") from exc


def build_feature_info_payload(name: str) -> dict[str, Any]:
    from pyimgano.features import feature_info

    return feature_info(str(name))


def list_model_preset_names(
    *,
    tags: Iterable[str] | None = None,
    family: str | None = None,
) -> list[str]:
    resolved_tags = resolve_model_preset_filter_tags(tags=tags, family=family)
    return list_model_presets(tags=resolved_tags or None)


def list_model_preset_infos_payload(
    *,
    tags: Iterable[str] | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    resolved_tags = resolve_model_preset_filter_tags(tags=tags, family=family)
    return list_model_preset_infos(tags=resolved_tags or None)


def build_model_preset_info_payload(name: str) -> dict[str, Any]:
    preset_name = str(name)
    preset = resolve_model_preset(preset_name)
    if preset is None:
        raise ValueError(f"Unknown model preset: {preset_name!r}")

    return model_preset_info(preset_name)


__all__ = [
    "build_feature_info_payload",
    "build_model_info_payload",
    "build_model_preset_info_payload",
    "build_suite_info_payload",
    "build_sweep_info_payload",
    "list_baseline_suites_payload",
    "list_dataset_categories_payload",
    "list_discovery_feature_names",
    "list_discovery_model_names",
    "list_model_preset_infos_payload",
    "list_model_preset_names",
    "list_sweeps_payload",
]
