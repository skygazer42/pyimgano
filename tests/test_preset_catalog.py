from __future__ import annotations


def test_preset_catalog_resolves_model_preset() -> None:
    from pyimgano.presets.catalog import resolve_model_preset

    preset = resolve_model_preset("industrial-structural-ecod")

    assert preset is not None
    assert preset.model == "vision_feature_pipeline"


def test_preset_catalog_resolves_defects_preset() -> None:
    from pyimgano.presets.catalog import resolve_defects_preset

    preset = resolve_defects_preset("industrial-defects-fp40")

    assert preset is not None
    assert preset.payload["min_area"] == 16


def test_preset_catalog_resolves_model_preset_filter_tags(monkeypatch) -> None:
    import pyimgano.presets.catalog as preset_catalog

    calls: list[str] = []
    monkeypatch.setattr(
        preset_catalog,
        "resolve_family_tags",
        lambda family: calls.append(str(family)) or ("neighbors", "graph"),
        raising=False,
    )

    tags = preset_catalog.resolve_model_preset_filter_tags(
        tags=["embeddings, gaussian", "calibration"],
        family="graph",
    )

    assert tags == ["embeddings", "gaussian", "calibration", "neighbors", "graph"]
    assert calls == ["graph"]


def test_cli_presets_resolve_model_preset_delegates_to_preset_catalog(monkeypatch) -> None:
    import pyimgano.cli_presets as cli_presets
    import pyimgano.presets.catalog as preset_catalog

    calls: list[str] = []
    monkeypatch.setattr(
        preset_catalog,
        "resolve_model_preset",
        lambda name: calls.append(str(name)) or {"delegated": True},
    )

    assert cli_presets.resolve_model_preset("delegated-preset") == {"delegated": True}
    assert calls == ["delegated-preset"]
