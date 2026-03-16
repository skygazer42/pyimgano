import pytest


def test_parse_model_kwargs_none_returns_empty_dict():
    from pyimgano.cli_common import parse_model_kwargs

    assert parse_model_kwargs(None) == {}


def test_parse_model_kwargs_requires_json_object():
    from pyimgano.cli_common import parse_model_kwargs

    with pytest.raises(ValueError, match="JSON object"):
        parse_model_kwargs("[1, 2, 3]")


def test_merge_checkpoint_path_sets_checkpoint_path():
    from pyimgano.cli_common import merge_checkpoint_path

    out = merge_checkpoint_path({}, checkpoint_path="/x.ckpt")
    assert out["checkpoint_path"] == "/x.ckpt"


def test_merge_checkpoint_path_detects_conflict():
    from pyimgano.cli_common import merge_checkpoint_path

    with pytest.raises(ValueError, match="conflict"):
        merge_checkpoint_path({"checkpoint_path": "/a.ckpt"}, checkpoint_path="/b.ckpt")


def test_build_model_kwargs_filters_auto_kwargs_for_strict_models():
    import pyimgano.models  # noqa: F401 - populate registry
    from pyimgano.cli_common import build_model_kwargs

    out = build_model_kwargs(
        "vision_abod",
        user_kwargs={},
        preset_kwargs=None,
        auto_kwargs={"device": "cpu", "contamination": 0.2, "pretrained": False},
    )
    assert out == {"contamination": 0.2}


def test_build_model_kwargs_materializes_lazy_constructor_signature_for_auto_kwargs():
    import pyimgano.models  # noqa: F401 - populate lazy registry
    from pyimgano.cli_common import build_model_kwargs
    from pyimgano.models.registry import MODEL_REGISTRY

    entry = MODEL_REGISTRY.info("ae_resnet_unet")
    was_lazy_placeholder = bool(entry.metadata.get("_lazy_placeholder", False))

    out = build_model_kwargs(
        "ae_resnet_unet",
        user_kwargs={},
        preset_kwargs=None,
        auto_kwargs={
            "device": "cpu",
            "contamination": 0.2,
            "pretrained": False,
            "random_state": 123,
            "random_seed": 123,
        },
    )
    assert out["device"] == "cpu"
    assert out["contamination"] == pytest.approx(0.2)
    assert out["random_state"] == 123
    assert "pretrained" not in out
    assert "random_seed" not in out
    if was_lazy_placeholder:
        entry_after = MODEL_REGISTRY.info("ae_resnet_unet")
        assert bool(entry_after.metadata.get("_lazy_placeholder", False)) is False
