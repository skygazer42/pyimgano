from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PyimListKindSpec:
    name: str
    text_field: str | None
    json_field: str | None
    request_fields: tuple[str, ...]
    text_title: str | None
    text_render_kind: str | None
    include_core_sections: bool
    include_recipes: bool
    include_datasets: bool
    render_when_empty_in_all: bool = True
    supports_family_filter: bool = False
    supports_algorithm_type_filter: bool = False
    supports_year_filter: bool = False
    supports_deployable_only: bool = False


CORE_PAYLOAD_FIELDS = (
    "models",
    "families",
    "types",
    "years",
    "metadata_contract",
    "preprocessing",
    "features",
    "model_presets",
    "model_preset_infos",
    "defects_presets",
)

ALL_PAYLOAD_FIELDS = CORE_PAYLOAD_FIELDS + ("recipes", "datasets")
PYIM_ALL_TEXT_LIST_KINDS = (
    "models",
    "families",
    "types",
    "years",
    "metadata-contract",
    "preprocessing",
    "features",
    "model-presets",
    "defects-presets",
    "recipes",
    "datasets",
)


_PYIM_LIST_KIND_SPECS = (
    PyimListKindSpec(
        name="all",
        text_field=None,
        json_field=None,
        request_fields=ALL_PAYLOAD_FIELDS,
        text_title=None,
        text_render_kind=None,
        include_core_sections=True,
        include_recipes=True,
        include_datasets=True,
        supports_family_filter=True,
        supports_deployable_only=True,
    ),
    PyimListKindSpec(
        name="models",
        text_field="models",
        json_field="models",
        request_fields=("models",),
        text_title="Models",
        text_render_kind="named-items",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
        supports_family_filter=True,
        supports_algorithm_type_filter=True,
        supports_year_filter=True,
    ),
    PyimListKindSpec(
        name="families",
        text_field="families",
        json_field="families",
        request_fields=("families",),
        text_title="Families",
        text_render_kind="counted-sections",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="types",
        text_field="types",
        json_field="types",
        request_fields=("types",),
        text_title="Types",
        text_render_kind="counted-sections",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="years",
        text_field="years",
        json_field="years",
        request_fields=("years",),
        text_title="Years",
        text_render_kind="counted-sections",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="metadata-contract",
        text_field="metadata_contract",
        json_field="metadata_contract",
        request_fields=("metadata_contract",),
        text_title="Metadata Contract",
        text_render_kind="metadata-contract",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="features",
        text_field="features",
        json_field="features",
        request_fields=("features",),
        text_title="Feature Extractors",
        text_render_kind="named-items",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="model-presets",
        text_field="model_presets",
        json_field="model_preset_infos",
        request_fields=("model_presets", "model_preset_infos"),
        text_title="Model Presets",
        text_render_kind="named-items",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
        supports_family_filter=True,
    ),
    PyimListKindSpec(
        name="defects-presets",
        text_field="defects_presets",
        json_field="defects_presets",
        request_fields=("defects_presets",),
        text_title="Defects Presets",
        text_render_kind="named-items",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
    ),
    PyimListKindSpec(
        name="preprocessing",
        text_field="preprocessing",
        json_field="preprocessing",
        request_fields=("preprocessing",),
        text_title="Preprocessing Schemes",
        text_render_kind="preprocessing",
        include_core_sections=True,
        include_recipes=False,
        include_datasets=False,
        supports_deployable_only=True,
    ),
    PyimListKindSpec(
        name="recipes",
        text_field="recipes",
        json_field="recipes",
        request_fields=("recipes",),
        text_title="Recipes",
        text_render_kind="recipes",
        include_core_sections=False,
        include_recipes=True,
        include_datasets=False,
        render_when_empty_in_all=False,
    ),
    PyimListKindSpec(
        name="datasets",
        text_field="datasets",
        json_field="datasets",
        request_fields=("datasets",),
        text_title="Datasets",
        text_render_kind="datasets",
        include_core_sections=False,
        include_recipes=False,
        include_datasets=True,
        render_when_empty_in_all=False,
    ),
)

PYIM_LIST_KIND_CHOICES = tuple(spec.name for spec in _PYIM_LIST_KIND_SPECS)
_PYIM_LIST_KIND_SPEC_MAP = {spec.name: spec for spec in _PYIM_LIST_KIND_SPECS}


def get_pyim_list_kind_spec(list_kind: str) -> PyimListKindSpec:
    list_kind_value = str(list_kind)
    try:
        return _PYIM_LIST_KIND_SPEC_MAP[list_kind_value]
    except KeyError as exc:
        raise KeyError(f"Unsupported pyim list kind: {list_kind_value}") from exc


__all__ = [
    "ALL_PAYLOAD_FIELDS",
    "CORE_PAYLOAD_FIELDS",
    "PYIM_ALL_TEXT_LIST_KINDS",
    "PYIM_LIST_KIND_CHOICES",
    "PyimListKindSpec",
    "get_pyim_list_kind_spec",
]
