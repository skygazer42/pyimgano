from __future__ import annotations


def test_resolve_pyim_json_payload_model_presets_uses_info_payload() -> None:
    from pyimgano.pyim_contracts import PyimListPayload
    from pyimgano.pyim_section_views import resolve_pyim_json_payload

    payload = PyimListPayload(
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
    )

    assert resolve_pyim_json_payload(payload, "model-presets") == [
        {"name": "industrial-structural-ecod", "tags": ["graph"]}
    ]


def test_resolve_pyim_text_section_view_uses_shared_spec_metadata() -> None:
    from pyimgano.pyim_contracts import PyimListPayload
    from pyimgano.pyim_section_views import PyimTextSectionView, resolve_pyim_text_section_view

    payload = PyimListPayload(
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
    )

    assert resolve_pyim_text_section_view(payload, "model-presets") == PyimTextSectionView(
        list_kind="model-presets",
        title="Model Presets",
        render_kind="named-items",
        value=["industrial-structural-ecod"],
    )


def test_iter_pyim_all_text_section_views_uses_shared_order_and_skips_empty_optional_sections() -> None:
    from pyimgano.pyim_contracts import (
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )
    from pyimgano.pyim_section_views import iter_pyim_all_text_section_views

    payload = PyimListPayload(
        models=["vision_patchcore"],
        families=[
            PyimModelFacetSummary(
                name="neighbors",
                description="Neighborhood methods",
                tags=["neighbors"],
                model_count=1,
                sample_models=["vision_lof"],
            )
        ],
        types=[
            PyimModelFacetSummary(
                name="deep-vision",
                description="Vision deep models",
                tags=["deep-vision"],
                model_count=1,
                sample_models=["vision_patchcore"],
            )
        ],
        years=[
            PyimYearSummary.from_mapping(
                {
                    "name": "2022",
                    "label": "2022",
                    "description": "Models from 2022",
                    "year": 2022,
                    "model_count": 1,
                    "sample_models": ["vision_patchcore"],
                }
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
        preprocessing=[
            PyimPreprocessingSchemeSummary.from_mapping(
                {
                    "name": "illumination-balanced",
                    "description": "Balanced preprocessing",
                    "deployable": True,
                    "tags": ["deployable"],
                }
            )
        ],
        features=["identity"],
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
        defects_presets=["industrial-defects-fp40"],
    )

    section_views = list(iter_pyim_all_text_section_views(payload))

    assert [section.list_kind for section in section_views] == [
        "models",
        "families",
        "types",
        "years",
        "metadata-contract",
        "preprocessing",
        "features",
        "model-presets",
        "defects-presets",
    ]
    assert [section.title for section in section_views] == [
        "Models",
        "Families",
        "Types",
        "Years",
        "Metadata Contract",
        "Preprocessing Schemes",
        "Feature Extractors",
        "Model Presets",
        "Defects Presets",
    ]
