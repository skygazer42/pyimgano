from __future__ import annotations

from pyimgano.services.robustness_service import RobustnessRunRequest, run_robustness_request


def test_run_robustness_request_delegates_to_benchmark(monkeypatch) -> None:
    import pyimgano.services.robustness_service as robustness_service

    calls: list[dict[str, object]] = []

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    monkeypatch.setattr(robustness_service, "_load_benchmark_split", lambda **_kwargs: _Split())
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

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(robustness_service, "_load_benchmark_split", lambda **_kwargs: _Split())
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

    class _Split:
        train_paths = ["train_0.png"]
        test_paths = ["test_0.png"]
        test_labels = [0]
        test_masks = None

    captured: dict[str, object] = {}

    monkeypatch.setattr(robustness_service, "_load_benchmark_split", lambda **_kwargs: _Split())
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
