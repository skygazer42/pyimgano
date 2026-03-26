from __future__ import annotations


def test_emit_pyim_list_payload_model_presets_json_uses_info_payload(monkeypatch) -> None:
    import pyimgano.pyim_cli_rendering as rendering
    from pyimgano.pyim_contracts import PyimListPayload

    calls = []
    monkeypatch.setattr(
        rendering,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 11
                )
            },
        ),
        raising=False,
    )

    payload = PyimListPayload(
        model_presets=["industrial-structural-ecod"],
        model_preset_infos=[{"name": "industrial-structural-ecod", "tags": ["graph"]}],
    )

    rc = rendering.emit_pyim_list_payload(
        payload,
        list_kind="model-presets",
        json_output=True,
    )

    assert rc == 11
    assert calls == [([{"name": "industrial-structural-ecod", "tags": ["graph"]}], {})]


def test_emit_pyim_list_payload_all_json_preserves_existing_shape(monkeypatch) -> None:
    import pyimgano.pyim_cli_rendering as rendering
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    calls = []
    monkeypatch.setattr(
        rendering,
        "cli_output",
        type(
            "_StubCliOutput",
            (),
            {
                "emit_json": staticmethod(
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 17
                )
            },
        ),
        raising=False,
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
                    "name": "2022",
                    "label": "2022",
                    "description": "d",
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

    rc = rendering.emit_pyim_list_payload(
        payload,
        list_kind="all",
        json_output=True,
    )

    assert rc == 17
    assert calls == [
        (
            {
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
                        "name": "2022",
                        "label": "2022",
                        "description": "d",
                        "year": 2022,
                        "model_count": 1,
                        "sample_models": ["vision_patchcore"],
                    }
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
                    {
                        "name": "custom",
                        "description": "Custom dataset",
                        "requires_category": False,
                    }
                ],
            },
            {},
        )
    ]


def test_emit_pyim_list_payload_renders_all_text_sections(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import (
        PyimDatasetSummary,
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    rc = emit_pyim_list_payload(
        PyimListPayload(
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
            recipes=[
                {"name": "industrial-adapt", "metadata": {"description": "Recipe description"}}
            ],
            datasets=[
                PyimDatasetSummary(
                    name="custom", description="Custom dataset", requires_category=True
                )
            ],
        ),
        list_kind="all",
        json_output=False,
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Models" in out
    assert "Families" in out
    assert "Types" in out
    assert "Years" in out
    assert "Metadata Contract" in out
    assert "Preprocessing Schemes" in out
    assert "Feature Extractors" in out
    assert "Model Presets" in out
    assert "Defects Presets" in out
    assert "Recipes" in out
    assert "industrial-adapt: Recipe description" in out
    assert "Datasets" in out
    assert "custom (category required): Custom dataset" in out


def test_emit_pyim_list_payload_renders_all_text_sections_in_shared_order(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import (
        PyimListPayload,
        PyimMetadataContractField,
        PyimModelFacetSummary,
        PyimPreprocessingSchemeSummary,
        PyimYearSummary,
    )

    rc = emit_pyim_list_payload(
        PyimListPayload(
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
        ),
        list_kind="all",
        json_output=False,
    )

    assert rc == 0
    out = capsys.readouterr().out
    titles = [
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
    positions = [out.index(title) for title in titles]
    assert positions == sorted(positions)
    assert "Recipes" not in out
    assert "Datasets" not in out
