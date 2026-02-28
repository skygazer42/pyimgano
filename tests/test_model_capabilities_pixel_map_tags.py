from __future__ import annotations


def test_pixel_map_models_are_tagged_for_discovery() -> None:
    from pyimgano.models.capabilities import compute_model_capabilities
    from pyimgano.models.registry import MODEL_REGISTRY

    # A small set of representative pixel-map models.
    candidates = [
        "ssim_template_map",
        "vision_patchcore",
        "vision_winclip",
        "vision_template_ncc_map",
        "vision_phase_correlation_map",
    ]

    for name in candidates:
        entry = MODEL_REGISTRY.info(name)
        caps = compute_model_capabilities(entry)
        assert caps.supports_pixel_map is True
        assert "pixel_map" in entry.tags


def test_template_models_are_tagged_as_template() -> None:
    from pyimgano.models.registry import MODEL_REGISTRY

    for name in [
        "ssim_template_map",
        "vision_template_ncc_map",
        "vision_phase_correlation_map",
    ]:
        entry = MODEL_REGISTRY.info(name)
        assert "template" in entry.tags

