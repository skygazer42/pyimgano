from __future__ import annotations

import pytest


def test_pyim_list_spec_exports_shared_choices_in_cli_order() -> None:
    from pyimgano.pyim_list_spec import PYIM_LIST_KIND_CHOICES

    assert PYIM_LIST_KIND_CHOICES == (
        "all",
        "models",
        "families",
        "types",
        "years",
        "metadata-contract",
        "features",
        "model-presets",
        "defects-presets",
        "preprocessing",
        "recipes",
        "datasets",
    )


def test_pyim_list_spec_exposes_shared_flags_for_representative_kinds() -> None:
    from pyimgano.pyim_list_spec import PYIM_ALL_TEXT_LIST_KINDS, get_pyim_list_kind_spec

    all_spec = get_pyim_list_kind_spec("all")
    models_spec = get_pyim_list_kind_spec("models")
    recipes_spec = get_pyim_list_kind_spec("recipes")
    preprocessing_spec = get_pyim_list_kind_spec("preprocessing")
    model_presets_spec = get_pyim_list_kind_spec("model-presets")
    metadata_contract_spec = get_pyim_list_kind_spec("metadata-contract")
    datasets_spec = get_pyim_list_kind_spec("datasets")

    assert all_spec.include_core_sections is True
    assert all_spec.include_recipes is True
    assert all_spec.include_datasets is True

    assert models_spec.include_core_sections is True
    assert models_spec.include_recipes is False
    assert models_spec.include_datasets is False
    assert models_spec.supports_family_filter is True
    assert models_spec.supports_algorithm_type_filter is True
    assert models_spec.supports_year_filter is True
    assert models_spec.supports_deployable_only is False

    assert recipes_spec.include_core_sections is False
    assert recipes_spec.include_recipes is True
    assert recipes_spec.include_datasets is False

    assert preprocessing_spec.include_core_sections is True
    assert preprocessing_spec.supports_deployable_only is True

    assert model_presets_spec.supports_family_filter is True
    assert model_presets_spec.text_field == "model_presets"
    assert model_presets_spec.json_field == "model_preset_infos"
    assert model_presets_spec.request_fields == ("model_presets", "model_preset_infos")
    assert model_presets_spec.text_title == "Model Presets"
    assert model_presets_spec.text_render_kind == "named-items"

    assert models_spec.text_title == "Models"
    assert models_spec.text_render_kind == "named-items"
    assert metadata_contract_spec.text_title == "Metadata Contract"
    assert metadata_contract_spec.text_render_kind == "metadata-contract"
    assert preprocessing_spec.text_title == "Preprocessing Schemes"
    assert preprocessing_spec.text_render_kind == "preprocessing"
    assert datasets_spec.text_title == "Datasets"
    assert datasets_spec.text_render_kind == "datasets"

    assert PYIM_ALL_TEXT_LIST_KINDS == (
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


def test_pyim_list_spec_rejects_unknown_kind() -> None:
    from pyimgano.pyim_list_spec import get_pyim_list_kind_spec

    with pytest.raises(KeyError, match="Unsupported pyim list kind: unknown"):
        get_pyim_list_kind_spec("unknown")
