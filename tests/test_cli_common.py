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

