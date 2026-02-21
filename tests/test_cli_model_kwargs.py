import pytest


def test_parse_model_kwargs_none_returns_empty_dict():
    from pyimgano.cli import _parse_model_kwargs

    assert _parse_model_kwargs(None) == {}


def test_parse_model_kwargs_requires_json_object():
    from pyimgano.cli import _parse_model_kwargs

    with pytest.raises(ValueError, match="JSON object"):
        _parse_model_kwargs("[1, 2, 3]")


def test_merge_checkpoint_path_sets_checkpoint_path():
    from pyimgano.cli import _merge_checkpoint_path

    out = _merge_checkpoint_path({}, checkpoint_path="/x.ckpt")
    assert out["checkpoint_path"] == "/x.ckpt"


def test_merge_checkpoint_path_detects_conflict():
    from pyimgano.cli import _merge_checkpoint_path

    with pytest.raises(ValueError, match="conflict"):
        _merge_checkpoint_path({"checkpoint_path": "/a.ckpt"}, checkpoint_path="/b.ckpt")
