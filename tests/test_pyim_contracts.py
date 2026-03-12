from __future__ import annotations


def test_pyim_contracts_module_exports_request_and_payload() -> None:
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListPayload,
        PyimListRequest,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    request = PyimListRequest(tags=["vision"])
    payload = PyimListPayload(models=["vision_patchcore"])
    facet = PyimModelFacetSummary(
        name="neighbors",
        description="Neighborhood methods",
        model_count=3,
        tags=["neighbors"],
        sample_models=["vision_lof"],
    )
    year = PyimYearSummary.from_mapping(
        {
            "name": "unknown",
            "label": "Unknown",
            "description": "Models without an annotated publication year in registry metadata.",
            "year": None,
            "model_count": 5,
            "sample_models": ["vision_patchcore"],
        }
    )
    field = PyimMetadataContractField(
        name="paper",
        source="registry_metadata",
        requirement="recommended",
        description="Canonical paper title.",
        value_type="non-empty string",
        required_when=None,
    )
    scheme = PyimPreprocessingSchemeSummary.from_mapping(
        {
            "name": "identity",
            "description": "No-op preprocessing.",
            "deployable": True,
            "tags": ["deployable"],
        }
    )
    dataset = PyimDatasetSummary(name="custom", description="Custom dataset", requires_category=False)

    assert request.tags == ["vision"]
    assert payload.models == ["vision_patchcore"]
    assert facet.to_payload() == {
        "name": "neighbors",
        "description": "Neighborhood methods",
        "tags": ["neighbors"],
        "model_count": 3,
        "sample_models": ["vision_lof"],
    }
    assert year.to_payload() == {
        "name": "unknown",
        "label": "Unknown",
        "description": "Models without an annotated publication year in registry metadata.",
        "year": None,
        "model_count": 5,
        "sample_models": ["vision_patchcore"],
    }
    assert field.to_payload() == {
        "name": "paper",
        "source": "registry_metadata",
        "requirement": "recommended",
        "description": "Canonical paper title.",
        "value_type": "non-empty string",
        "required_when": None,
    }
    assert scheme.to_payload() == {
        "name": "identity",
        "description": "No-op preprocessing.",
        "deployable": True,
        "tags": ["deployable"],
    }
    assert dataset.to_payload() == {
        "name": "custom",
        "description": "Custom dataset",
        "requires_category": False,
    }


def test_pyim_year_summary_preserves_pre_2001_range_shape() -> None:
    from pyimgano.pyim_contracts import PyimYearSummary

    item = PyimYearSummary.from_mapping(
        {
            "name": "pre-2001",
            "label": "Pre-2001",
            "description": "Models annotated with publication years earlier than 2001.",
            "year_start": None,
            "year_end": 2000,
            "model_count": 8,
            "sample_models": ["vision_lof"],
        }
    )

    assert item.to_payload() == {
        "name": "pre-2001",
        "label": "Pre-2001",
        "description": "Models annotated with publication years earlier than 2001.",
        "year_start": None,
        "year_end": 2000,
        "model_count": 8,
        "sample_models": ["vision_lof"],
    }


def test_pyim_service_reexports_neutral_contracts() -> None:
    import pyimgano.services.pyim_service as pyim_service
    from pyimgano.pyim_contracts import PyimListPayload, PyimListRequest

    assert pyim_service.PyimListRequest is PyimListRequest
    assert pyim_service.PyimListPayload is PyimListPayload


def test_pyim_list_request_can_derive_section_flags_from_list_kind() -> None:
    from pyimgano.pyim_contracts import PyimListRequest

    legacy_request = PyimListRequest()
    datasets_request = PyimListRequest(list_kind="datasets")
    override_request = PyimListRequest(list_kind="datasets", include_core_sections=True)
    families_request = PyimListRequest(list_kind="families")

    assert legacy_request.list_kind is None
    assert legacy_request.include_core_sections is True
    assert legacy_request.include_recipes is False
    assert legacy_request.include_datasets is False
    assert legacy_request.requested_payload_fields() == (
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

    assert datasets_request.list_kind == "datasets"
    assert datasets_request.include_core_sections is False
    assert datasets_request.include_recipes is False
    assert datasets_request.include_datasets is True
    assert datasets_request.requested_payload_fields() == ("datasets",)

    assert override_request.include_core_sections is True
    assert override_request.include_recipes is False
    assert override_request.include_datasets is True
    assert override_request.requested_payload_fields() == (
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
        "datasets",
    )

    assert families_request.requested_payload_fields() == ("families",)


def test_pyim_list_payload_exposes_shared_section_access_and_json_payloads() -> None:
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
    )

    payload = PyimListPayload(
        models=["vision_patchcore"],
        families=[
            PyimModelFacetSummary(
                name="neighbors",
                description="Neighborhood methods",
                model_count=1,
                tags=["neighbors"],
                sample_models=["vision_lof"],
            )
        ],
        metadata_contract=[
            PyimMetadataContractField(
                name="paper",
                source="metadata",
                requirement="recommended",
                description="Paper reference",
                value_type="string",
                required_when=None,
            )
        ],
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
        datasets=[
            PyimDatasetSummary(
                name="custom",
                description="Custom dataset",
                requires_category=False,
            )
        ],
    )

    assert payload.get_section_value("models") == ["vision_patchcore"]
    assert payload.get_section_value("model-presets") == ["industrial-structural-ecod"]
    assert payload.to_json_payload("families") == [
        {
            "name": "neighbors",
            "description": "Neighborhood methods",
            "tags": ["neighbors"],
            "model_count": 1,
            "sample_models": ["vision_lof"],
        }
    ]
    assert payload.to_json_payload("metadata-contract") == [
        {
            "name": "paper",
            "source": "metadata",
            "requirement": "recommended",
            "description": "Paper reference",
            "value_type": "string",
            "required_when": None,
        }
    ]
    assert payload.to_json_payload("model-presets") == [
        {"name": "industrial-structural-ecod", "tags": ["graph"]}
    ]
    assert payload.to_json_payload("datasets") == [
        {
            "name": "custom",
            "description": "Custom dataset",
            "requires_category": False,
        }
    ]
