def test_anomalydino_is_registered_in_model_registry():
    from pyimgano.models import list_models

    assert "vision_anomalydino" in list_models()

