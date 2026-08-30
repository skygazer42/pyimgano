import pytest


def test_anomalib_checkpoint_wrapper_is_registered():
    from pyimgano.models import list_models

    assert "vision_anomalib_checkpoint" in list_models()


def test_anomalib_aliases_are_registered():
    from pyimgano.models import list_models

    anomalib_models = set(list_models(tags=["anomalib"]))
    assert "vision_dinomaly_anomalib" in anomalib_models
    assert "vision_cfa_anomalib" in anomalib_models


def test_more_anomalib_aliases_are_registered():
    from pyimgano.models import list_models

    anomalib_models = set(list_models(tags=["anomalib"]))
    assert "vision_csflow_anomalib" in anomalib_models
    assert "vision_dsr_anomalib" in anomalib_models
    assert "vision_uflow_anomalib" in anomalib_models
    assert "vision_winclip_anomalib" in anomalib_models
    assert "vision_fre_anomalib" in anomalib_models
    assert "vision_supersimplenet_anomalib" in anomalib_models
    assert "vision_vlmad_anomalib" in anomalib_models


def test_anomalib_checkpoint_wrapper_requires_anomalib_if_no_inferencer(monkeypatch):
    from pyimgano.models import anomalib_backend, create_model

    def missing_anomalib(*_args, **_kwargs):
        raise ImportError("simulated missing anomalib")

    monkeypatch.setattr(anomalib_backend, "require", missing_anomalib)

    with pytest.raises(ImportError):
        create_model(
            "vision_anomalib_checkpoint",
            checkpoint_path="does-not-exist.ckpt",
            device="cpu",
        )
