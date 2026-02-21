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


def test_cli_requires_checkpoint_for_checkpoint_backed_models(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "load_benchmark_split", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "evaluate_split", lambda *_a, **_k: {"image_metrics": {"auroc": 0.0}})
    monkeypatch.setattr(cli, "create_model", lambda *_a, **_k: object())

    code = cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_anomalib_checkpoint",
            "--device",
            "cpu",
        ]
    )
    assert code != 0


def test_validate_user_kwargs_rejects_unknown_keys_for_strict_models():
    from pyimgano.cli import _validate_user_model_kwargs

    with pytest.raises(TypeError, match="does not accept"):
        _validate_user_model_kwargs("vision_abod", {"not_a_param": 1})


def test_build_model_kwargs_does_not_override_user_values():
    from pyimgano.cli import _build_model_kwargs

    out = _build_model_kwargs(
        "vision_patchcore",
        user_kwargs={"device": "cpu"},
        auto_kwargs={"device": "cuda", "contamination": 0.1},
    )
    assert out["device"] == "cpu"
    assert out["contamination"] == 0.1
