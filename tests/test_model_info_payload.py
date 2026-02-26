def test_registry_model_info_includes_capabilities_payload() -> None:
    import pyimgano.models  # noqa: F401 - registry population side effects

    from pyimgano.models.registry import model_info

    info = model_info("vision_ecod")

    assert info["name"] == "vision_ecod"
    assert "capabilities" in info
    caps = info["capabilities"]
    assert "input_modes" in caps
    assert "paths" in caps["input_modes"]
    assert isinstance(caps["supports_pixel_map"], bool)
    assert isinstance(caps["supports_checkpoint"], bool)
    assert isinstance(caps["requires_checkpoint"], bool)
    assert isinstance(caps["supports_save_load"], bool)

    # Convenience aliases
    assert info["input_modes"] == caps["input_modes"]
    assert info["supports_save_load"] == caps["supports_save_load"]

