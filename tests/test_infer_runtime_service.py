from __future__ import annotations

import pytest

from pyimgano.services.infer_runtime_service import (
    InferRuntimePlanRequest,
    prepare_infer_runtime_plan,
)


def test_prepare_infer_runtime_plan_builds_postprocess_and_infer_config_threshold() -> None:
    result = prepare_infer_runtime_plan(
        InferRuntimePlanRequest(
            detector=object(),
            include_maps_requested=False,
            include_maps_by_default=False,
            postprocess_requested=False,
            infer_config_postprocess={
                "normalize": True,
                "normalize_method": "percentile",
                "percentile_range": [2.0, 98.0],
                "gaussian_sigma": 1.5,
                "morph_open_ksize": 3,
                "morph_close_ksize": 5,
                "component_threshold": 0.6,
                "min_component_area": 11,
            },
            defects_enabled=True,
            defects_payload={"pixel_threshold": 0.5},
            defects_payload_source="infer_config",
            pixel_threshold=None,
            pixel_threshold_strategy="normal_pixel_quantile",
            pixel_normal_quantile=0.999,
            roi_xyxy_norm=None,
            train_paths=[],
            batch_size=None,
            amp=False,
        )
    )

    assert result.include_maps is True
    assert result.postprocess is not None
    assert result.postprocess.normalize is True
    assert result.postprocess.normalize_method == "percentile"
    assert result.postprocess.percentile_range == (2.0, 98.0)
    assert result.postprocess.gaussian_sigma == pytest.approx(1.5)
    assert result.postprocess.morph_open_ksize == 3
    assert result.postprocess.morph_close_ksize == 5
    assert result.postprocess.component_threshold == pytest.approx(0.6)
    assert result.postprocess.min_component_area == 11
    assert result.pixel_threshold_value == pytest.approx(0.5)
    assert result.pixel_threshold_provenance is not None
    assert result.pixel_threshold_provenance["source"] == "infer_config"


def test_prepare_infer_runtime_plan_delegates_infer_config_postprocess_to_service_boundary(
    monkeypatch,
) -> None:
    import pyimgano.services.infer_runtime_service as infer_runtime_service

    calls = []

    monkeypatch.setattr(
        infer_runtime_service.workbench_adaptation_service,
        "build_postprocess_from_payload",
        lambda payload: calls.append(dict(payload)) or "POSTPROCESS",
    )

    result = infer_runtime_service.prepare_infer_runtime_plan(
        InferRuntimePlanRequest(
            detector=object(),
            include_maps_requested=False,
            include_maps_by_default=False,
            postprocess_requested=False,
            infer_config_postprocess={
                "normalize": False,
                "normalize_method": "minmax",
            },
            defects_enabled=True,
            defects_payload={"pixel_threshold": 0.5},
            defects_payload_source="infer_config",
            pixel_threshold=None,
            pixel_threshold_strategy="normal_pixel_quantile",
            pixel_normal_quantile=0.999,
            roi_xyxy_norm=None,
            train_paths=[],
            batch_size=None,
            amp=False,
        )
    )

    assert calls == [
        {
            "normalize": False,
            "normalize_method": "minmax",
        }
    ]
    assert result.postprocess == "POSTPROCESS"
