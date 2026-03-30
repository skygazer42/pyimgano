from __future__ import annotations


def test_collect_pyim_listing_payload_builds_typed_sections_before_payload_coercion(
    monkeypatch,
) -> None:
    import pyimgano.datasets.converters as dataset_converters
    import pyimgano.discovery as discovery
    import pyimgano.models.registry as model_registry
    import pyimgano.presets.catalog as preset_catalog
    import pyimgano.services.discovery_service as discovery_service
    import pyimgano.services.pyim_service as pyim_service
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListRequest,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    monkeypatch.setattr(pyim_service, "PyimListPayload", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr(
        discovery_service,
        "list_discovery_model_names",
        lambda **_kwargs: ["model-a"],
    )
    monkeypatch.setattr(
        discovery,
        "list_model_families",
        lambda: [
            {
                "name": "neighbors",
                "description": "Neighborhood methods",
                "model_count": 1,
                "tags": ["neighbors"],
                "sample_models": ["model-a"],
            }
        ],
    )
    monkeypatch.setattr(
        discovery,
        "list_model_types",
        lambda: [
            {
                "name": "deep-vision",
                "description": "Deep models",
                "model_count": 1,
                "tags": ["deep-vision"],
                "sample_models": ["model-a"],
            }
        ],
    )
    monkeypatch.setattr(
        discovery,
        "list_model_years",
        lambda: [
            {
                "name": "2024",
                "label": "2024",
                "description": "Models from 2024",
                "year": 2024,
                "model_count": 1,
                "sample_models": ["model-a"],
            }
        ],
    )
    monkeypatch.setattr(
        model_registry,
        "model_metadata_contract",
        lambda: [
            {
                "name": "paper",
                "source": "metadata",
                "requirement": "recommended",
                "description": "Paper reference",
                "value_type": "string",
                "required_when": None,
            }
        ],
    )
    monkeypatch.setattr(
        discovery,
        "list_preprocessing_schemes",
        lambda deployable_only=False: [
            {
                "name": "identity",
                "description": "No-op preprocessing",
                "deployable": bool(deployable_only),
                "tags": ["deployable"],
            }
        ],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_discovery_feature_names",
        lambda **_kwargs: ["identity"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_names",
        lambda **_kwargs: ["preset-a"],
    )
    monkeypatch.setattr(
        discovery_service,
        "list_model_preset_infos_payload",
        lambda **_kwargs: [{"name": "preset-a", "tags": ["graph"]}],
    )
    monkeypatch.setattr(
        preset_catalog,
        "list_defects_presets",
        lambda: ["defects-a"],
    )

    class _Converter:
        name = "custom"
        description = "Custom dataset"
        requires_category = True

    monkeypatch.setattr(
        dataset_converters,
        "list_dataset_converters",
        lambda: [_Converter()],
    )

    payload_kwargs = pyim_service.collect_pyim_listing_payload(
        PyimListRequest(include_datasets=True)
    )

    assert isinstance(payload_kwargs["families"][0], PyimModelFacetSummary)
    assert isinstance(payload_kwargs["types"][0], PyimModelFacetSummary)
    assert isinstance(payload_kwargs["years"][0], PyimYearSummary)
    assert isinstance(payload_kwargs["metadata_contract"][0], PyimMetadataContractField)
    assert isinstance(payload_kwargs["preprocessing"][0], PyimPreprocessingSchemeSummary)
    assert isinstance(payload_kwargs["datasets"][0], PyimDatasetSummary)
    assert payload_kwargs["model_preset_infos"] == [{"name": "preset-a", "tags": ["graph"]}]
    assert payload_kwargs["recipes"] == []


def test_collect_pyim_listing_payload_returns_core_sections() -> None:
    from pyimgano.pyim_contracts import (
        PyimListPayload,
        PyimListRequest,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(PyimListRequest())

    assert isinstance(payload, PyimListPayload)
    assert "vision_patchcore" in payload.models
    assert any(
        isinstance(item, PyimModelFacetSummary) and item.name == "neighbors"
        for item in payload.families
    )
    assert any(
        isinstance(item, PyimModelFacetSummary) and item.name == "deep-vision"
        for item in payload.types
    )
    assert any(isinstance(item, PyimYearSummary) and item.name == "2001" for item in payload.years)
    assert any(
        isinstance(item, PyimMetadataContractField) and item.name == "paper"
        for item in payload.metadata_contract
    )
    assert "identity" in payload.features
    assert "industrial-structural-ecod" in payload.model_presets
    assert any(item["name"] == "industrial-structural-ecod" for item in payload.model_preset_infos)
    assert "industrial-defects-fp40" in payload.defects_presets
    assert any(
        isinstance(item, PyimPreprocessingSchemeSummary)
        and item.name == "illumination-contrast-balanced"
        for item in payload.preprocessing
    )
    assert payload.recipes == []
    assert payload.datasets == []


def test_collect_pyim_listing_payload_optionally_includes_recipes_and_datasets() -> None:
    from pyimgano.pyim_contracts import PyimDatasetSummary, PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(
        PyimListRequest(include_recipes=True, include_datasets=True)
    )

    assert any(item.get("name") == "industrial-adapt" for item in payload.recipes)
    assert any(
        isinstance(item, PyimDatasetSummary) and item.name == "custom" for item in payload.datasets
    )


def test_collect_pyim_listing_payload_respects_model_preset_family_filter() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(PyimListRequest(family="graph"))

    assert "industrial-structural-rgraph" in payload.model_presets
    assert "industrial-structural-lof" not in payload.model_presets


def test_collect_pyim_listing_payload_can_skip_core_sections_for_recipes_only() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(
        PyimListRequest(include_core_sections=False, include_recipes=True)
    )

    assert payload.models == []
    assert payload.families == []
    assert payload.types == []
    assert payload.years == []
    assert payload.metadata_contract == []
    assert payload.preprocessing == []
    assert payload.features == []
    assert payload.model_presets == []
    assert payload.model_preset_infos == []
    assert payload.defects_presets == []
    assert any(item.get("name") == "industrial-adapt" for item in payload.recipes)
    assert payload.datasets == []


def test_collect_pyim_listing_payload_can_skip_core_sections_for_datasets_only() -> None:
    from pyimgano.pyim_contracts import PyimDatasetSummary, PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(
        PyimListRequest(include_core_sections=False, include_datasets=True)
    )

    assert payload.models == []
    assert payload.families == []
    assert payload.types == []
    assert payload.years == []
    assert payload.metadata_contract == []
    assert payload.preprocessing == []
    assert payload.features == []
    assert payload.model_presets == []
    assert payload.model_preset_infos == []
    assert payload.defects_presets == []
    assert payload.recipes == []
    assert any(
        isinstance(item, PyimDatasetSummary) and item.name == "custom" for item in payload.datasets
    )


def test_collect_pyim_listing_payload_can_derive_sections_from_list_kind() -> None:
    from pyimgano.pyim_contracts import PyimDatasetSummary, PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(PyimListRequest(list_kind="datasets"))

    assert payload.models == []
    assert payload.families == []
    assert payload.types == []
    assert payload.years == []
    assert payload.metadata_contract == []
    assert payload.preprocessing == []
    assert payload.features == []
    assert payload.model_presets == []
    assert payload.model_preset_infos == []
    assert payload.defects_presets == []
    assert payload.recipes == []
    assert any(
        isinstance(item, PyimDatasetSummary) and item.name == "custom" for item in payload.datasets
    )


def test_collect_pyim_listing_payload_for_families_only_populates_requested_section() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(PyimListRequest(list_kind="families"))

    assert payload.models == []
    assert payload.families != []
    assert payload.types == []
    assert payload.years == []
    assert payload.metadata_contract == []
    assert payload.preprocessing == []
    assert payload.features == []
    assert payload.model_presets == []
    assert payload.model_preset_infos == []
    assert payload.defects_presets == []
    assert payload.recipes == []
    assert payload.datasets == []


def test_collect_pyim_listing_payload_for_model_presets_only_populates_preset_sections() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_listing_payload

    payload = collect_pyim_listing_payload(PyimListRequest(list_kind="model-presets"))

    assert payload.models == []
    assert payload.families == []
    assert payload.types == []
    assert payload.years == []
    assert payload.metadata_contract == []
    assert payload.preprocessing == []
    assert payload.features == []
    assert payload.model_presets != []
    assert payload.model_preset_infos != []
    assert payload.defects_presets == []
    assert payload.recipes == []
    assert payload.datasets == []


def test_collect_pyim_model_selection_payload_prefers_curated_candidates() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_model_selection_payload

    payload = collect_pyim_model_selection_payload(
        PyimListRequest(
            list_kind="models",
            objective="latency",
            selection_profile="cpu-screening",
            topk=2,
        )
    )

    assert payload["selection_context"] == {
        "objective": "latency",
        "selection_profile": "cpu-screening",
        "topk": 2,
    }
    picks = payload["starter_picks"]
    assert isinstance(picks, list)
    assert len(picks) == 2
    assert picks[0]["name"] == "vision_ecod"
    assert isinstance(picks[0]["summary"], str)
    assert picks[0]["supports_pixel_map"] is False
    assert picks[0]["tested_runtime"] == "numpy"
    assert (
        picks[0]["doctor_command"]
        == "pyimgano-doctor --recommend-extras --for-model vision_ecod --json"
    )
    assert (
        picks[0]["model_info_command"]
        == "pyimgano-benchmark --model-info vision_ecod --json"
    )
    assert payload["suggested_commands"] == [
        "pyimgano-doctor --recommend-extras --for-model vision_ecod --json",
        "pyimgano-benchmark --model-info vision_ecod --json",
    ]


def test_collect_pyim_model_selection_payload_surfaces_localization_hints() -> None:
    from pyimgano.pyim_contracts import PyimListRequest
    from pyimgano.services.pyim_service import collect_pyim_model_selection_payload

    payload = collect_pyim_model_selection_payload(
        PyimListRequest(
            list_kind="models",
            objective="localization",
            selection_profile="balanced",
            topk=3,
        )
    )

    by_name = {item["name"]: item for item in payload["starter_picks"]}
    template = by_name["ssim_template_map"]
    assert template["supports_pixel_map"] is True
    assert template["tested_runtime"] == "numpy"
    assert template["deployment_family"] == ["template"]


def test_pyim_list_payload_builds_all_json_payload_without_model_preset_infos() -> None:
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    payload = PyimListPayload(
        models=["vision_patchcore"],
        families=[
            PyimModelFacetSummary(
                name="neighbors",
                description="d",
                tags=["neighbors"],
                model_count=1,
                sample_models=["vision_lof"],
            )
        ],
        types=[
            PyimModelFacetSummary(
                name="deep-vision",
                description="d",
                tags=["deep-vision"],
                model_count=1,
                sample_models=["vision_patchcore"],
            )
        ],
        years=[
            PyimYearSummary.from_mapping(
                {
                    "name": "pre-2001",
                    "label": "Pre-2001",
                    "description": "d",
                    "year_start": None,
                    "year_end": 2000,
                    "model_count": 1,
                    "sample_models": ["vision_lof"],
                }
            ),
            PyimYearSummary.from_mapping(
                {
                    "name": "unknown",
                    "label": "Unknown",
                    "description": "d2",
                    "year": None,
                    "model_count": 2,
                    "sample_models": ["vision_patchcore"],
                }
            ),
        ],
        metadata_contract=[
            PyimMetadataContractField(
                name="paper",
                source="metadata",
                requirement="recommended",
                description="d",
                value_type="string",
                required_when=None,
            )
        ],
        preprocessing=[
            PyimPreprocessingSchemeSummary.from_mapping(
                {
                    "name": "identity",
                    "description": "No-op",
                    "deployable": True,
                    "tags": ["deployable"],
                }
            )
        ],
        features=["identity"],
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
        defects_presets=["industrial-defects-fp40"],
        recipes=[{"name": "industrial-adapt", "metadata": {}}],
        datasets=[
            PyimDatasetSummary(
                name="custom",
                description="Custom dataset",
                requires_category=False,
            )
        ],
    )

    assert payload.to_all_json_payload() == {
        "models": ["vision_patchcore"],
        "families": [
            {
                "name": "neighbors",
                "description": "d",
                "tags": ["neighbors"],
                "model_count": 1,
                "sample_models": ["vision_lof"],
            }
        ],
        "types": [
            {
                "name": "deep-vision",
                "description": "d",
                "tags": ["deep-vision"],
                "model_count": 1,
                "sample_models": ["vision_patchcore"],
            }
        ],
        "years": [
            {
                "name": "pre-2001",
                "label": "Pre-2001",
                "description": "d",
                "year_start": None,
                "year_end": 2000,
                "model_count": 1,
                "sample_models": ["vision_lof"],
            },
            {
                "name": "unknown",
                "label": "Unknown",
                "description": "d2",
                "year": None,
                "model_count": 2,
                "sample_models": ["vision_patchcore"],
            },
        ],
        "metadata_contract": [
            {
                "name": "paper",
                "source": "metadata",
                "requirement": "recommended",
                "description": "d",
                "value_type": "string",
                "required_when": None,
            }
        ],
        "preprocessing": [
            {
                "name": "identity",
                "description": "No-op",
                "deployable": True,
                "tags": ["deployable"],
            }
        ],
        "features": ["identity"],
        "model_presets": ["industrial-structural-ecod"],
        "defects_presets": ["industrial-defects-fp40"],
        "recipes": [{"name": "industrial-adapt", "metadata": {}}],
        "datasets": [
            {"name": "custom", "description": "Custom dataset", "requires_category": False}
        ],
    }
