from __future__ import annotations


def test_cli_preset_industrial_embedding_core_balanced_resolves() -> None:
    from pyimgano.cli_presets import list_model_presets, resolve_model_preset

    assert "industrial-embedding-core-balanced" in list_model_presets()

    preset = resolve_model_preset("industrial-embedding-core-balanced")
    assert preset is not None
    assert preset.model == "vision_embedding_core"
    assert preset.optional is False

    kwargs = dict(preset.kwargs)
    assert kwargs["embedding_extractor"] == "torchvision_backbone"
    embed_kwargs = dict(kwargs["embedding_kwargs"])
    assert embed_kwargs.get("pretrained", None) is False
    assert embed_kwargs.get("pool", None) in {"avg", "max", "gem", "cls"}

    assert kwargs["core_detector"] == "core_score_standardizer"
    core_kwargs = dict(kwargs["core_kwargs"])
    assert core_kwargs["base_detector"] == "core_mahalanobis_shrinkage"
    assert core_kwargs["method"] == "rank"

