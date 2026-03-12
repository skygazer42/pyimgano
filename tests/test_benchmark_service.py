from __future__ import annotations

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
    assert postprocess.gaussian_sigma == 1.5
    assert postprocess.morph_open_ksize == 3
    assert postprocess.morph_close_ksize == 5
    assert postprocess.component_threshold == 0.6
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
