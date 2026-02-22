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


def test_cli_filters_auto_kwargs_for_strict_models(monkeypatch):
    import pyimgano.cli as cli

    captured: dict[str, object] = {}

    def fake_create_model(_name: str, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(cli, "load_benchmark_split", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "evaluate_split", lambda *_a, **_k: {"image_metrics": {"auroc": 0.0}})
    monkeypatch.setattr(cli, "create_model", fake_create_model)

    code = cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_abod",
            "--device",
            "cpu",
            "--contamination",
            "0.2",
        ]
    )
    assert code == 0
    assert "device" not in captured
    assert "pretrained" not in captured
    assert captured["contamination"] == 0.2


def test_cli_merges_checkpoint_path_for_anomalib_models(monkeypatch):
    import pyimgano.cli as cli

    captured: dict[str, object] = {}

    def fake_create_model(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(cli, "load_benchmark_split", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "evaluate_split", lambda *_a, **_k: {"image_metrics": {"auroc": 0.0}})
    monkeypatch.setattr(cli, "create_model", fake_create_model)

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
            "--checkpoint-path",
            "/x.ckpt",
            "--device",
            "cpu",
            "--contamination",
            "0.2",
        ]
    )
    assert code == 0
    assert captured["name"] == "vision_anomalib_checkpoint"
    kwargs = captured["kwargs"]
    assert kwargs["checkpoint_path"] == "/x.ckpt"
    assert kwargs["device"] == "cpu"
    assert kwargs["contamination"] == 0.2
    # The checkpoint wrapper doesn't accept `pretrained`, so the CLI should not pass it.
    assert "pretrained" not in kwargs


def test_cli_parser_accepts_preset_industrial_balanced():
    import pyimgano.cli as cli

    parser = cli._build_parser()
    try:
        parser.parse_args(
            [
                "--dataset",
                "mvtec",
                "--root",
                "/tmp",
                "--category",
                "bottle",
                "--preset",
                "industrial-balanced",
            ]
        )
    except SystemExit as exc:
        raise AssertionError(f"parser should accept --preset, got SystemExit({exc.code})") from exc


def test_cli_parser_accepts_preset_industrial_fast():
    import pyimgano.cli as cli

    parser = cli._build_parser()
    try:
        parser.parse_args(
            [
                "--dataset",
                "mvtec",
                "--root",
                "/tmp",
                "--category",
                "bottle",
                "--preset",
                "industrial-fast",
            ]
        )
    except SystemExit as exc:
        raise AssertionError(f"parser should accept --preset, got SystemExit({exc.code})") from exc


def test_cli_parser_accepts_preset_industrial_accurate():
    import pyimgano.cli as cli

    parser = cli._build_parser()
    try:
        parser.parse_args(
            [
                "--dataset",
                "mvtec",
                "--root",
                "/tmp",
                "--category",
                "bottle",
                "--preset",
                "industrial-accurate",
            ]
        )
    except SystemExit as exc:
        raise AssertionError(f"parser should accept --preset, got SystemExit({exc.code})") from exc


def test_resolve_preset_kwargs_patchcore_prefers_sklearn_when_no_faiss(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_patchcore")
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["knn_backend"] == "sklearn"


def test_resolve_preset_kwargs_patchcore_prefers_faiss_when_available(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: True, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_patchcore")
    assert kwargs["knn_backend"] == "faiss"


def test_build_model_kwargs_user_overrides_preset_values():
    from pyimgano.cli import _build_model_kwargs

    out = _build_model_kwargs(
        "vision_patchcore",
        user_kwargs={"coreset_sampling_ratio": 0.2},
        preset_kwargs={"coreset_sampling_ratio": 0.05, "n_neighbors": 5},
        auto_kwargs={"device": "cpu"},
    )
    assert out["coreset_sampling_ratio"] == 0.2
    assert out["n_neighbors"] == 5
    assert out["device"] == "cpu"


def test_cli_applies_preset_for_patchcore(monkeypatch):
    import pyimgano.cli as cli

    captured: dict[str, object] = {}

    def fake_create_model(name: str, **kwargs):
        captured["name"] = name
        captured["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    monkeypatch.setattr(cli, "load_benchmark_split", lambda *_a, **_k: object())
    monkeypatch.setattr(cli, "evaluate_split", lambda *_a, **_k: {"image_metrics": {"auroc": 0.0}})
    monkeypatch.setattr(cli, "create_model", fake_create_model)

    code = cli.main(
        [
            "--dataset",
            "mvtec",
            "--root",
            "/tmp",
            "--category",
            "bottle",
            "--model",
            "vision_patchcore",
            "--preset",
            "industrial-balanced",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert captured["name"] == "vision_patchcore"
    kwargs = captured["kwargs"]
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["coreset_sampling_ratio"] == 0.05
    assert kwargs["knn_backend"] == "sklearn"


def test_resolve_preset_kwargs_anomalydino_includes_balanced_defaults(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_anomalydino")
    assert kwargs["knn_backend"] == "sklearn"
    assert kwargs["coreset_sampling_ratio"] == 0.2
    assert kwargs["image_size"] == 448


def test_resolve_preset_kwargs_softpatch_includes_robust_defaults(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: True, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_softpatch")
    assert kwargs["knn_backend"] == "faiss"
    assert kwargs["coreset_sampling_ratio"] == 0.2
    assert kwargs["train_patch_outlier_quantile"] == 0.1
    assert kwargs["image_size"] == 448


def test_resolve_preset_kwargs_simplenet_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_simplenet")
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["epochs"] == 10
    assert kwargs["batch_size"] == 16


def test_resolve_preset_kwargs_stfpm_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_stfpm")
    assert kwargs["epochs"] == 50
    assert kwargs["batch_size"] == 32


def test_resolve_preset_kwargs_reverse_distillation_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_reverse_distillation")
    assert kwargs["epoch_num"] == 10
    assert kwargs["batch_size"] == 32


def test_resolve_preset_kwargs_draem_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_draem")
    assert kwargs["epochs"] == 50
    assert kwargs["batch_size"] == 16
    assert kwargs["image_size"] == 256


def test_resolve_preset_kwargs_padim_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_padim")
    assert kwargs["backbone"] == "resnet18"
    assert kwargs["d_reduced"] == 64
    assert kwargs["image_size"] == 224


def test_resolve_preset_kwargs_spade_includes_balanced_defaults():
    import pyimgano.cli as cli

    kwargs = cli._resolve_preset_kwargs("industrial-balanced", "vision_spade")
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["image_size"] == 256
    assert kwargs["k_neighbors"] == 50
    assert kwargs["gaussian_sigma"] == 4.0


def test_resolve_preset_kwargs_reverse_dist_alias_matches_reverse_distillation():
    import pyimgano.cli as cli

    a = cli._resolve_preset_kwargs("industrial-balanced", "vision_reverse_distillation")
    b = cli._resolve_preset_kwargs("industrial-balanced", "vision_reverse_dist")
    assert a == b


def test_resolve_preset_kwargs_fast_patchcore_includes_speed_defaults(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-fast", "vision_patchcore")
    assert kwargs["backbone"] == "resnet50"
    assert kwargs["coreset_sampling_ratio"] == 0.02
    assert kwargs["n_neighbors"] == 3
    assert kwargs["knn_backend"] == "sklearn"


def test_resolve_preset_kwargs_accurate_anomalydino_includes_accuracy_defaults(monkeypatch):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: True, raising=False)
    kwargs = cli._resolve_preset_kwargs("industrial-accurate", "vision_anomalydino")
    assert kwargs["knn_backend"] == "faiss"
    assert kwargs["coreset_sampling_ratio"] == 0.5
    assert kwargs["image_size"] == 518


def test_resolve_preset_kwargs_reverse_dist_alias_matches_across_presets():
    import pyimgano.cli as cli

    for preset in ("industrial-fast", "industrial-balanced", "industrial-accurate"):
        a = cli._resolve_preset_kwargs(preset, "vision_reverse_distillation")
        b = cli._resolve_preset_kwargs(preset, "vision_reverse_dist")
        assert a == b


@pytest.mark.parametrize(
    "preset,model_name",
    [
        ("industrial-fast", "vision_patchcore"),
        ("industrial-fast", "vision_padim"),
        ("industrial-fast", "vision_spade"),
        ("industrial-fast", "vision_anomalydino"),
        ("industrial-fast", "vision_softpatch"),
        ("industrial-fast", "vision_simplenet"),
        ("industrial-fast", "vision_fastflow"),
        ("industrial-fast", "vision_cflow"),
        ("industrial-fast", "vision_stfpm"),
        ("industrial-fast", "vision_reverse_distillation"),
        ("industrial-fast", "vision_reverse_dist"),
        ("industrial-fast", "vision_draem"),
        ("industrial-balanced", "vision_patchcore"),
        ("industrial-balanced", "vision_padim"),
        ("industrial-balanced", "vision_spade"),
        ("industrial-balanced", "vision_anomalydino"),
        ("industrial-balanced", "vision_softpatch"),
        ("industrial-balanced", "vision_simplenet"),
        ("industrial-balanced", "vision_fastflow"),
        ("industrial-balanced", "vision_cflow"),
        ("industrial-balanced", "vision_stfpm"),
        ("industrial-balanced", "vision_reverse_distillation"),
        ("industrial-balanced", "vision_reverse_dist"),
        ("industrial-balanced", "vision_draem"),
        ("industrial-accurate", "vision_patchcore"),
        ("industrial-accurate", "vision_padim"),
        ("industrial-accurate", "vision_spade"),
        ("industrial-accurate", "vision_anomalydino"),
        ("industrial-accurate", "vision_softpatch"),
        ("industrial-accurate", "vision_simplenet"),
        ("industrial-accurate", "vision_fastflow"),
        ("industrial-accurate", "vision_cflow"),
        ("industrial-accurate", "vision_stfpm"),
        ("industrial-accurate", "vision_reverse_distillation"),
        ("industrial-accurate", "vision_reverse_dist"),
        ("industrial-accurate", "vision_draem"),
    ],
)
def test_industrial_presets_cover_expected_models(monkeypatch, preset, model_name):
    import pyimgano.cli as cli

    monkeypatch.setattr(cli, "_faiss_available", lambda: False, raising=False)
    kwargs = cli._resolve_preset_kwargs(preset, model_name)
    assert kwargs != {}
