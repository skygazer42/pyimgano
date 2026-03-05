from __future__ import annotations


def test_openclip_patch_map_model_is_registered_and_optional() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects
    from pyimgano.models.registry import MODEL_REGISTRY, create_model, list_models

    assert "vision_openclip_patch_map" in list_models()
    entry = MODEL_REGISTRY.info("vision_openclip_patch_map")
    assert "openclip" in entry.tags

    # Construction may fail if optional deps are missing; the important part is
    # that the error is actionable and doesn't break registry discovery.
    try:
        det = create_model(
            "vision_openclip_patch_map",
            openclip_pretrained=None,
            device="cpu",
        )
    except ImportError as exc:
        msg = str(exc).lower()
        assert "open_clip" in msg
        assert "pyimgano[clip]" in msg or "open_clip_torch" in msg
    else:
        # If deps exist, ensure the safe default (no implicit downloads).
        assert getattr(det, "openclip_pretrained", None) is None
