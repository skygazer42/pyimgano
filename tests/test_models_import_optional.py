def test_pyimgano_models_import_does_not_require_optional_deps():
    # Importing the models package should not hard-require optional 3rd party
    # dependencies used by specific model implementations.
    #
    # This test intentionally does not import those optional deps.
    import pyimgano.models as models

    available = models.list_models()
    assert "vision_openclip_patchknn" in available
    assert "vision_openclip_promptscore" in available
    assert "winclip" in available
    assert "vision_winclip" in available
