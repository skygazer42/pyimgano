from __future__ import annotations

import math

import pytest


def test_workbench_adaptation_service_exports_expected_boundary() -> None:
    import pyimgano.services.workbench_adaptation_service as workbench_adaptation_service

    assert workbench_adaptation_service.__all__ == ["build_postprocess_from_payload"]


def test_workbench_adaptation_service_delegates_to_workbench_adaptation(monkeypatch) -> None:
    import pyimgano.services.workbench_adaptation_service as workbench_adaptation_service
    import pyimgano.workbench.adaptation as adaptation

    calls = []

    monkeypatch.setattr(
        adaptation,
        "build_postprocess",
        lambda config: calls.append(config) or {"kind": "postprocess"},
    )

    result = workbench_adaptation_service.build_postprocess_from_payload(
        {
            "normalize": True,
            "normalize_method": "percentile",
            "percentile_range": [2.0, 98.0],
            "gaussian_sigma": 1.5,
            "morph_open_ksize": 3,
            "morph_close_ksize": 5,
            "component_threshold": 0.6,
            "min_component_area": 11,
        }
    )

    assert result == {"kind": "postprocess"}
    assert len(calls) == 1
    config = calls[0]
    assert isinstance(config, adaptation.MapPostprocessConfig)
    assert config.normalize is True
    assert config.normalize_method == "percentile"
    assert config.percentile_range == (2.0, 98.0)
    assert math.isclose(config.gaussian_sigma, 1.5)
    assert config.morph_open_ksize == 3
    assert config.morph_close_ksize == 5
    assert config.component_threshold == pytest.approx(0.6)
    assert config.min_component_area == 11
