from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from pyimgano.pyim_contracts import (
    PyimDatasetSummary,
    PyimListRequest,
    PyimMetadataContractField,
    PyimModelFacetSummary,
    PyimPreprocessingSchemeSummary,
    PyimYearSummary,
)
from pyimgano.pyim_list_spec import ALL_PAYLOAD_FIELDS

PyimPayloadFieldCollector = Callable[[PyimListRequest], Any]


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    out: list[str] = []
    for item in tags:
        for piece in str(item).split(","):
            tag = piece.strip()
            if tag:
                out.append(tag)
    return out


def _build_model_facet_summaries(
    items: Iterable[Mapping[str, Any]],
) -> list[PyimModelFacetSummary]:
    return [PyimModelFacetSummary.from_mapping(item) for item in items]


def _build_year_summaries(items: Iterable[Mapping[str, Any]]) -> list[PyimYearSummary]:
    return [PyimYearSummary.from_mapping(item) for item in items]


def _build_metadata_contract_fields(
    items: Iterable[Mapping[str, Any]],
) -> list[PyimMetadataContractField]:
    return [PyimMetadataContractField.from_mapping(item) for item in items]


def _build_preprocessing_summaries(
    items: Iterable[Mapping[str, Any]],
) -> list[PyimPreprocessingSchemeSummary]:
    return [PyimPreprocessingSchemeSummary.from_mapping(item) for item in items]


def _build_dataset_summaries(converters: Iterable[Any]) -> list[PyimDatasetSummary]:
    return [
        PyimDatasetSummary(
            name=str(converter.name),
            description=str(converter.description),
            requires_category=bool(converter.requires_category),
        )
        for converter in converters
    ]


def _virtual_dataset_summaries() -> list[PyimDatasetSummary]:
    return [
        PyimDatasetSummary(
            name="manifest",
            description="JSONL manifest dataset with explicit paths/splits for industrial workflows.",
            requires_category=True,
        ),
        PyimDatasetSummary(
            name="mvtec",
            description="MVTec AD public benchmark dataset layout.",
            requires_category=True,
        ),
    ]


def empty_pyim_payload_kwargs() -> dict[str, Any]:
    return {field_name: [] for field_name in ALL_PAYLOAD_FIELDS}


def _collect_models(request: PyimListRequest) -> list[str]:
    import pyimgano.services.discovery_service as discovery_service

    return discovery_service.list_discovery_model_names(
        tags=request.tags,
        family=request.family,
        algorithm_type=request.algorithm_type,
        year=request.year,
    )


def _collect_families(_request: PyimListRequest) -> list[PyimModelFacetSummary]:
    from pyimgano.discovery import list_model_families

    return _build_model_facet_summaries(list_model_families())


def _collect_types(_request: PyimListRequest) -> list[PyimModelFacetSummary]:
    from pyimgano.discovery import list_model_types

    return _build_model_facet_summaries(list_model_types())


def _collect_years(_request: PyimListRequest) -> list[PyimYearSummary]:
    from pyimgano.discovery import list_model_years

    return _build_year_summaries(list_model_years())


def _collect_metadata_contract(_request: PyimListRequest) -> list[PyimMetadataContractField]:
    from pyimgano.models.registry import model_metadata_contract

    return _build_metadata_contract_fields(model_metadata_contract())


def _collect_preprocessing(request: PyimListRequest) -> list[PyimPreprocessingSchemeSummary]:
    from pyimgano.discovery import list_preprocessing_schemes

    return _build_preprocessing_summaries(
        list_preprocessing_schemes(deployable_only=bool(request.deployable_only))
    )


def _collect_features(request: PyimListRequest) -> list[str]:
    import pyimgano.services.discovery_service as discovery_service

    return discovery_service.list_discovery_feature_names(tags=request.tags)


def _collect_model_presets(request: PyimListRequest) -> list[str]:
    import pyimgano.services.discovery_service as discovery_service

    return discovery_service.list_model_preset_names(
        tags=request.tags,
        family=request.family,
    )


def _collect_model_preset_infos(request: PyimListRequest) -> list[dict[str, Any]]:
    import pyimgano.services.discovery_service as discovery_service

    return discovery_service.list_model_preset_infos_payload(
        tags=request.tags,
        family=request.family,
    )


def list_filtered_model_names(request: PyimListRequest) -> list[str]:
    return _collect_models(request)


def build_model_info_payload(model_name: str) -> dict[str, Any]:
    import pyimgano.services.discovery_service as discovery_service

    return discovery_service.build_model_info_payload(model_name)


def _collect_defects_presets(_request: PyimListRequest) -> list[str]:
    from pyimgano.presets.catalog import list_defects_presets

    return list_defects_presets()


def _collect_recipe_payloads(request: PyimListRequest) -> list[dict[str, Any]]:
    import pyimgano.recipes  # noqa: F401
    from pyimgano.recipes.registry import list_recipes as _list_recipes
    from pyimgano.recipes.registry import recipe_info as _recipe_info

    tag_filter = _normalize_tags(request.tags) or None
    recipe_names = _list_recipes(tags=tag_filter)
    return [_recipe_info(name) for name in recipe_names]


def _collect_datasets(_request: PyimListRequest) -> list[PyimDatasetSummary]:
    from pyimgano.datasets.converters import list_dataset_converters

    return [
        *_virtual_dataset_summaries(),
        *_build_dataset_summaries(list_dataset_converters()),
    ]


_PAYLOAD_FIELD_COLLECTORS: dict[str, PyimPayloadFieldCollector] = {
    "models": _collect_models,
    "families": _collect_families,
    "types": _collect_types,
    "years": _collect_years,
    "metadata_contract": _collect_metadata_contract,
    "preprocessing": _collect_preprocessing,
    "features": _collect_features,
    "model_presets": _collect_model_presets,
    "model_preset_infos": _collect_model_preset_infos,
    "defects_presets": _collect_defects_presets,
    "recipes": _collect_recipe_payloads,
    "datasets": _collect_datasets,
}


def collect_pyim_payload_field(field_name: str, request: PyimListRequest) -> Any:
    field_name_value = str(field_name)
    try:
        collector = _PAYLOAD_FIELD_COLLECTORS[field_name_value]
    except KeyError as exc:
        raise KeyError(f"Unsupported pyim payload field: {field_name_value}") from exc
    return collector(request)


__all__ = [
    "collect_pyim_payload_field",
    "empty_pyim_payload_kwargs",
]
