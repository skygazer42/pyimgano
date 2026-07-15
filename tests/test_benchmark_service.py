from __future__ import annotations

import json

import numpy as np
import pytest

from pyimgano.services.benchmark_service import (
    BenchmarkRunRequest,
    PixelPostprocessConfig,
    SuiteRunRequest,
    build_pixel_postprocess,
    run_benchmark_request,
    run_suite_request,
)


def test_run_benchmark_request_delegates_to_pipeline(monkeypatch) -> None:
    import pyimgano.services.benchmark_service as benchmark_service

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        benchmark_service,
        "_run_benchmark_pipeline",
        lambda **kwargs: calls.append(kwargs) or {"ok": True},
    )

    request = BenchmarkRunRequest(
        dataset="custom",
        root="/tmp/custom",
        manifest_path=None,
        category="custom",
        model="vision_ecod",
        input_mode="paths",
        resize=(16, 16),
    )

    payload = run_benchmark_request(request)

    assert payload["ok"] is True
    assert calls[0]["model"] == "vision_ecod"


def test_build_pixel_postprocess_materializes_anomaly_map_postprocess() -> None:
    postprocess = build_pixel_postprocess(
        PixelPostprocessConfig(
            normalize_method="percentile",
            percentile_range=(2.0, 98.0),
            gaussian_sigma=1.5,
            morph_open_ksize=3,
            morph_close_ksize=5,
            component_threshold=0.6,
            min_component_area=11,
        )
    )

    assert postprocess is not None
    assert postprocess.normalize is True
    assert postprocess.normalize_method == "percentile"
    assert postprocess.percentile_range == (2.0, 98.0)
    assert postprocess.gaussian_sigma == pytest.approx(1.5)
    assert postprocess.morph_open_ksize == 3
    assert postprocess.morph_close_ksize == 5
    assert postprocess.component_threshold == pytest.approx(0.6)
    assert postprocess.min_component_area == 11


def test_run_benchmark_request_delegates_pixel_mode_to_pixel_runner(monkeypatch) -> None:
    import pyimgano.services.benchmark_service as benchmark_service

    calls: list[BenchmarkRunRequest] = []

    monkeypatch.setattr(
        benchmark_service,
        "_run_pixel_benchmark_request",
        lambda request: calls.append(request) or {"pixel": True},
    )

    payload = run_benchmark_request(
        BenchmarkRunRequest(
            dataset="custom",
            root="/tmp/custom",
            category="custom",
            model="vision_pixel_mean_absdiff_map",
            pixel=True,
            resize=(16, 16),
        )
    )

    assert payload["pixel"] is True
    assert calls[0].pixel is True
    assert calls[0].model == "vision_pixel_mean_absdiff_map"


def test_pixel_run_honors_limits_and_saves_traceable_artifacts(tmp_path, monkeypatch) -> None:
    import pyimgano.services.benchmark_service as benchmark_service
    from pyimgano.pipelines.mvtec_visa import BenchmarkSplit
    from pyimgano.services.dataset_split_service import LoadedBenchmarkSplit

    class _Detector:
        def decision_function(self, paths):  # noqa: ANN001
            return np.arange(len(paths), dtype=np.float64)

    detector = _Detector()
    split = BenchmarkSplit(
        train_paths=["train-0.png", "train-1.png"],
        test_paths=["test-0.png", "test-1.png", "test-2.png"],
        test_labels=np.asarray([0, 1, 1], dtype=np.int64),
        test_masks=np.zeros((3, 4, 4), dtype=np.uint8),
    )

    monkeypatch.setattr(
        benchmark_service,
        "_resolve_model_run_options",
        lambda request: ("resolved_model", {"paper_setting": 7}, object()),
    )
    monkeypatch.setattr(benchmark_service, "create_model", lambda *args, **kwargs: detector)
    monkeypatch.setattr(
        benchmark_service.dataset_split_service,
        "load_benchmark_style_split",
        lambda **kwargs: LoadedBenchmarkSplit(split=split),
    )

    evaluated: list[BenchmarkSplit] = []

    def _fake_evaluate(_detector, limited_split, _request):  # noqa: ANN001
        evaluated.append(limited_split)
        return {"auroc": 1.0, "threshold": 0.5, "pixel_metrics": {"pixel_auroc": 1.0}}

    monkeypatch.setattr(benchmark_service, "_evaluate_pixel_split", _fake_evaluate)
    monkeypatch.setattr(
        "pyimgano.reporting.environment.collect_environment",
        lambda: {"fingerprint_sha256": "test-fingerprint"},
    )

    payload = benchmark_service.run_benchmark_request(
        BenchmarkRunRequest(
            dataset="custom",
            root="/tmp/custom",
            category="widget",
            model="requested_model",
            pixel=True,
            limit_train=1,
            limit_test=2,
            output_dir=str(tmp_path / "pixel-run"),
        )
    )

    assert len(evaluated[0].train_paths) == 1
    assert len(evaluated[0].test_paths) == 2
    assert payload["dataset_summary"] == {
        "train_count": 1,
        "test_count": 2,
        "test_anomaly_count": 1,
    }
    assert payload["run_dir"] == str(tmp_path / "pixel-run")

    run_dir = tmp_path / "pixel-run"
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "environment.json").is_file()
    assert (run_dir / "categories" / "widget" / "report.json").is_file()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    assert config["model"] == "resolved_model"
    assert config["requested_model"] == "requested_model"
    assert config["model_kwargs"] == {"paper_setting": 7}
    records = [
        json.loads(line)
        for line in (run_dir / "categories" / "widget" / "per_image.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["input"] for record in records] == ["test-0.png", "test-1.png"]
    assert [record["pred"] for record in records] == [0, 1]


def test_run_suite_request_normalizes_filters_and_delegates(monkeypatch) -> None:
    import pyimgano.services.benchmark_service as benchmark_service

    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        benchmark_service,
        "_run_suite_pipeline",
        lambda **kwargs: calls.append(kwargs) or {"suite": kwargs["suite"], "rows": []},
    )

    payload = run_suite_request(
        SuiteRunRequest(
            suite="industrial-v1",
            dataset="custom",
            root="/tmp/custom",
            category="custom",
            resize=(16, 16),
            include_baselines=["alpha,beta", " gamma "],
            exclude_baselines=["delta"],
        )
    )

    assert payload["suite"] == "industrial-v1"
    assert calls[0]["include_baselines"] == ["alpha", "beta", "gamma"]
    assert calls[0]["exclude_baselines"] == ["delta"]
    assert calls[0]["resize"] == (16, 16)
