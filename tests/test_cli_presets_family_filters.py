from __future__ import annotations


def test_list_model_presets_supports_family_tag_filters() -> None:
    from pyimgano.cli_presets import list_model_presets

    graph = set(list_model_presets(tags=["graph"]))
    neighbors = set(list_model_presets(tags=["neighbors"]))
    gaussian = set(list_model_presets(tags=["gaussian"]))
    ensemble = set(list_model_presets(tags=["ensemble"]))
    clustering = set(list_model_presets(tags=["clustering"]))
    reconstruction = set(list_model_presets(tags=["reconstruction"]))
    distillation = set(list_model_presets(tags=["distillation"]))

    assert "industrial-structural-rgraph" in graph
    assert "industrial-structural-lof" in neighbors
    assert "industrial-structural-mahalanobis" in gaussian
    assert "industrial-structural-suod" in ensemble
    assert "industrial-structural-kmeans" in clustering
    assert "industrial-embed-torch-autoencoder" in reconstruction
    assert "industrial-reverse-distillation" in distillation


def test_resolve_model_preset_exposes_family_tags() -> None:
    from pyimgano.cli_presets import resolve_model_preset

    preset = resolve_model_preset("industrial-structural-rgraph")
    assert preset is not None
    assert "graph" in set(preset.tags)
    assert "structural" in set(preset.tags)


def test_resolve_new_model_presets_for_additional_algorithm_families() -> None:
    from pyimgano.cli_presets import resolve_model_preset

    kmeans = resolve_model_preset("industrial-structural-kmeans")
    ae = resolve_model_preset("industrial-embed-torch-autoencoder")
    rd = resolve_model_preset("industrial-reverse-distillation")

    assert kmeans is not None
    assert kmeans.model == "vision_feature_pipeline"
    assert "clustering" in set(kmeans.tags)

    assert ae is not None
    assert ae.model == "vision_embedding_torch_autoencoder"
    assert "reconstruction" in set(ae.tags)
    assert ae.optional is True

    assert rd is not None
    assert rd.model == "vision_reverse_distillation"
    assert "distillation" in set(rd.tags)
    assert rd.optional is True
