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


def test_emit_pyim_list_payload_model_json_can_include_selection_metadata(monkeypatch) -> None:
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
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 23
                )
            },
        ),
        raising=False,
    )

    rc = rendering.emit_pyim_list_payload(
        PyimListPayload(models=["vision_ecod", "ssim_template_map"]),
        list_kind="models",
        json_output=True,
        selection_payload={
            "selection_context": {
                "objective": "latency",
                "selection_profile": "cpu-screening",
                "topk": 2,
            },
            "suggested_commands": [
                "pyimgano-doctor --recommend-extras --for-model vision_ecod --json"
            ],
            "starter_picks": [
                {
                    "name": "vision_ecod",
                    "summary": "Fast CPU baseline.",
                    "required_extras": [],
                    "supports_pixel_map": False,
                    "tested_runtime": "numpy",
                    "deployment_family": [],
                }
            ],
        },
    )

    assert rc == 23
    assert calls == [
        (
            {
                "items": ["vision_ecod", "ssim_template_map"],
                "selection_context": {
                    "objective": "latency",
                    "selection_profile": "cpu-screening",
                    "topk": 2,
                },
                "suggested_commands": [
                    "pyimgano-doctor --recommend-extras --for-model vision_ecod --json"
                ],
                "starter_picks": [
                    {
                        "name": "vision_ecod",
                        "summary": "Fast CPU baseline.",
                        "required_extras": [],
                        "supports_pixel_map": False,
                        "tested_runtime": "numpy",
                        "deployment_family": [],
                    }
                ],
            },
            {},
        )
    ]


def test_emit_pyim_list_payload_json_can_include_goal_payload(monkeypatch) -> None:
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
                    lambda payload, **kwargs: calls.append((payload, kwargs)) or 29
                )
            },
        ),
        raising=False,
    )

    rc = rendering.emit_pyim_list_payload(
        PyimListPayload(
            models=["vision_ecod"], recipes=[{"name": "industrial-adapt", "metadata": {}}]
        ),
        list_kind="all",
        json_output=True,
        goal_payload={
            "goal_context": {"goal": "first-run"},
            "goal_picks": {
                "models": [{"name": "vision_ecod", "why_this_pick": "Fast CPU baseline."}],
                "recipes": [{"name": "industrial-adapt"}],
                "datasets": [{"name": "custom"}],
            },
            "suggested_commands": ["pyimgano-doctor --profile first-run --json"],
        },
    )

    assert rc == 29
    assert calls == [
        (
            {
                "models": ["vision_ecod"],
                "families": [],
                "types": [],
                "years": [],
                "metadata_contract": [],
                "preprocessing": [],
                "features": [],
                "model_presets": [],
                "defects_presets": [],
                "recipes": [{"name": "industrial-adapt", "metadata": {}}],
                "datasets": [],
                "goal_context": {"goal": "first-run"},
                "goal_picks": {
                    "models": [{"name": "vision_ecod", "why_this_pick": "Fast CPU baseline."}],
                    "recipes": [{"name": "industrial-adapt"}],
                    "datasets": [{"name": "custom"}],
                },
                "suggested_commands": ["pyimgano-doctor --profile first-run --json"],
            },
            {},
        )
    ]


def test_emit_pyim_list_payload_model_text_renders_runtime_and_pixel_hints(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import PyimListPayload

    rc = emit_pyim_list_payload(
        PyimListPayload(models=["vision_ecod", "ssim_template_map"]),
        list_kind="models",
        json_output=False,
        selection_payload={
            "selection_context": {
                "objective": "localization",
                "selection_profile": "balanced",
                "topk": 2,
            },
            "starter_picks": [
                {
                    "name": "ssim_template_map",
                    "summary": "Reference-style pixel localization baseline.",
                    "required_extras": [],
                    "supports_pixel_map": True,
                    "tested_runtime": "numpy",
                    "deployment_family": ["template"],
                }
            ],
        },
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Selection Context" in out
    assert "objective=localization" in out
    assert "selection_profile=balanced" in out
    assert "Starter Picks" in out
    assert "runtime=numpy" in out
    assert "pixel_map=yes" in out
    assert "family=template" in out


def test_emit_pyim_list_payload_model_text_renders_install_hint_for_required_extras(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import PyimListPayload

    rc = emit_pyim_list_payload(
        PyimListPayload(models=["vision_patchcore"]),
        list_kind="models",
        json_output=False,
        selection_payload={
            "selection_context": {
                "objective": "localization",
                "selection_profile": "balanced",
                "topk": 1,
            },
            "starter_picks": [
                {
                    "name": "vision_patchcore",
                    "summary": "Strong localization baseline.",
                    "required_extras": ["torch"],
                    "supports_pixel_map": True,
                    "tested_runtime": "torch",
                    "deployment_family": ["patchcore", "memory_bank"],
                    "install_hint": "pip install 'pyimgano[torch]'",
                }
            ],
        },
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "extras: torch" in out
    assert "install: pip install 'pyimgano[torch]'" in out


def test_emit_pyim_list_payload_model_text_renders_suggested_commands(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import PyimListPayload

    rc = emit_pyim_list_payload(
        PyimListPayload(models=["vision_ecod"]),
        list_kind="models",
        json_output=False,
        selection_payload={
            "selection_context": {
                "objective": "latency",
                "selection_profile": "cpu-screening",
                "topk": 1,
            },
            "suggested_commands": [
                "pyimgano-doctor --recommend-extras --for-model vision_ecod --json",
                "pyimgano-benchmark --model-info vision_ecod --json",
            ],
            "starter_picks": [
                {
                    "name": "vision_ecod",
                    "summary": "Fast CPU baseline.",
                    "required_extras": [],
                    "supports_pixel_map": False,
                    "tested_runtime": "numpy",
                    "deployment_family": [],
                }
            ],
        },
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Suggested Commands" in out
    assert "pyimgano-doctor --recommend-extras --for-model vision_ecod --json" in out
    assert "pyimgano-benchmark --model-info vision_ecod --json" in out


def test_emit_pyim_list_payload_text_renders_goal_context_and_goal_picks(capsys) -> None:
    from pyimgano.pyim_cli_rendering import emit_pyim_list_payload
    from pyimgano.pyim_contracts import PyimListPayload

    rc = emit_pyim_list_payload(
        PyimListPayload(
            models=["vision_ecod"],
            recipes=[{"name": "industrial-adapt", "metadata": {"description": "recipe"}}],
            datasets=[],
        ),
        list_kind="all",
        json_output=False,
        goal_payload={
            "goal_context": {
                "goal": "deployable",
                "objective": "balanced",
                "selection_profile": "deploy-readiness",
            },
            "goal_picks": {
                "models": [
                    {
                        "name": "vision_ecod",
                        "summary": "Fast CPU baseline.",
                        "why_this_pick": "Native CPU route with minimal setup.",
                        "install_hint": None,
                    }
                ],
                "recipes": [
                    {
                        "name": "industrial-adapt",
                        "summary": "Audited train/export loop.",
                        "config_path": "examples/configs/deploy_smoke_custom_cpu.json",
                        "runtime_profile": "cpu-offline",
                    }
                ],
                "datasets": [
                    {
                        "name": "custom",
                        "summary": "Folder-layout starter dataset path.",
                    }
                ],
            },
            "suggested_commands": [
                "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json",
            ],
        },
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "Goal Context" in out
    assert "goal=deployable" in out
    assert "Goal Picks" in out
    assert "model=vision_ecod" in out
    assert "why=Native CPU route with minimal setup." in out
    assert "recipe=industrial-adapt" in out
    assert "config=examples/configs/deploy_smoke_custom_cpu.json" in out
    assert "profile=cpu-offline" in out
    assert "dataset=custom" in out
    assert "Suggested Commands" in out
    assert "install=None" not in out
