import pytest


def test_anomalib_checkpoint_wrapper_is_registered():
    from pyimgano.models import list_models

    assert "vision_anomalib_checkpoint" in list_models()


def test_anomalib_checkpoint_wrapper_requires_anomalib_if_no_inferencer():
    from pyimgano.models import create_model

    with pytest.raises(ImportError):
        create_model(
            "vision_anomalib_checkpoint",
            checkpoint_path="does-not-exist.ckpt",
            device="cpu",
        )

