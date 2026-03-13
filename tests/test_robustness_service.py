from __future__ import annotations

from pyimgano.services.robustness_service import RobustnessRunRequest, run_robustness_request


def test_run_robustness_request_delegates_to_benchmark(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    calls: list[dict[str, object]] = []

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(robustness_service, "create_model", lambda *_a, **_k: object())

    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: calls.append(kwargs) or {"clean": {}, "corruptions": {}},
    )

    payload = run_robustness_request(
        RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert "robustness" in payload
    assert payload["model"] == "vision_ecod"
    assert isinstance(calls, list)


def test_run_robustness_request_uses_request_checkpoint_path(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(
        robustness_service,
        "create_model",
        lambda name, **kwargs: captured.update(name=str(name), kwargs=dict(kwargs)) or object(),
    )
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    payload = robustness_service.run_robustness_request(
        robustness_service.RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_patchcore_anomalib",
            checkpoint_path="/tmp/checkpoint.ckpt",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert payload["model"] == "vision_patchcore_anomalib"
    assert captured["name"] == "vision_patchcore_anomalib"
    assert captured["kwargs"]["checkpoint_path"] == "/tmp/checkpoint.ckpt"


def test_run_robustness_request_accepts_model_preset_alias(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **_kwargs: dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(
        robustness_service,
        "create_model",
        lambda name, **kwargs: captured.update(name=str(name), kwargs=dict(kwargs)) or object(),
    )
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    payload = robustness_service.run_robustness_request(
        robustness_service.RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="industrial-structural-ecod",
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert payload["model"] == "industrial-structural-ecod"
    assert captured["name"] == "vision_feature_pipeline"


def test_run_robustness_request_delegates_split_loading_through_service(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service
    import pyimgano.services.dataset_split_service as dataset_split_service

    calls: list[dict[str, object]] = []

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(
        dataset_split_service,
        "load_benchmark_style_split",
        lambda **kwargs: calls.append(dict(kwargs))
        or dataset_split_service.LoadedBenchmarkSplit(
            split=_Split(),
            pixel_skip_reason=None,
        ),
    )
    monkeypatch.setattr(robustness_service, "create_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        robustness_service,
        "_run_robustness_benchmark",
        lambda *args, **kwargs: {"clean": {}, "corruptions": {}},
    )

    robustness_service.run_robustness_request(
        RobustnessRunRequest(
            dataset="mvtec",
            root="/tmp/root",
            category="bottle",
            model="vision_ecod",
            resize=(32, 32),
            input_mode="paths",
            pixel_segf1=False,
        )
    )

    assert len(calls) == 1
    assert calls[0]["dataset"] == "mvtec"
    assert calls[0]["root"] == "/tmp/root"
    assert calls[0]["category"] == "bottle"
    assert calls[0]["resize"] == (32, 32)
    assert calls[0]["load_masks"] is True
