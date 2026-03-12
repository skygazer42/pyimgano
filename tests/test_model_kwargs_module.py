import pytest


def test_model_kwargs_merge_checkpoint_path_sets_checkpoint_path():
    from pyimgano.models.model_kwargs import merge_checkpoint_path

    out = merge_checkpoint_path({}, checkpoint_path="/x.ckpt")
    assert out["checkpoint_path"] == "/x.ckpt"


def test_model_kwargs_validate_user_kwargs_rejects_unknown_keys_for_strict_models():
    import pyimgano.models  # noqa: F401 - populate registry
    from pyimgano.models.model_kwargs import validate_user_model_kwargs

    with pytest.raises(TypeError, match="does not accept"):
        validate_user_model_kwargs("vision_abod", {"not_a_param": 1})


def test_cli_common_build_model_kwargs_delegates_to_models_module(monkeypatch):
    import pyimgano.cli_common as cli_common
    import pyimgano.models.model_kwargs as model_kwargs_module

    monkeypatch.setattr(
        model_kwargs_module,
        "build_model_kwargs",
        lambda model_name, *, user_kwargs, preset_kwargs=None, auto_kwargs: {
            "model_name": model_name,
            "user_kwargs": dict(user_kwargs),
            "preset_kwargs": None if preset_kwargs is None else dict(preset_kwargs),
            "auto_kwargs": dict(auto_kwargs),
            "delegated": True,
        },
    )

    out = cli_common.build_model_kwargs(
        "vision_patchcore",
        user_kwargs={"device": "cpu"},
        preset_kwargs={"backbone": "resnet50"},
        auto_kwargs={"contamination": 0.1},
    )

    assert out == {
        "model_name": "vision_patchcore",
        "user_kwargs": {"device": "cpu"},
        "preset_kwargs": {"backbone": "resnet50"},
        "auto_kwargs": {"contamination": 0.1},
        "delegated": True,
    }
