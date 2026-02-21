def test_pyimgano_models_import_does_not_require_optional_deps():
    # Importing the models package should not hard-require optional 3rd party
    # dependencies used by specific model implementations.
    #
    # This test intentionally does not import those optional deps.
    import pyimgano.models  # noqa: F401

