from __future__ import annotations

import pytest

from pyimgano.services.infer_context_service import ConfigBackedInferContext
from pyimgano.services.infer_load_service import (
    ConfigBackedInferLoadRequest,
    DirectInferLoadRequest,
    load_config_backed_infer_detector,
    load_direct_infer_detector,
)


def test_load_direct_infer_detector_accepts_model_preset_alias() -> None:
    created: dict[str, object] = {}

    result = load_direct_infer_detector(
        DirectInferLoadRequest(requested_model="industrial-structural-ecod"),
        create_detector=lambda name, **kwargs: created.update(
            name=str(name),
            kwargs=dict(kwargs),
        )
        or object(),
    )

    assert result.model_name == "vision_feature_pipeline"
    assert created["name"] == "vision_feature_pipeline"


def test_load_config_backed_infer_detector_restores_checkpoint_and_threshold() -> None:
    created: dict[str, object] = {}
    loaded: list[str] = []

    class _Det:
        threshold_ = None

    detector = _Det()

    result = load_config_backed_infer_detector(
        ConfigBackedInferLoadRequest(
            context=ConfigBackedInferContext(
                model_name="vision_ecod",
                preset=None,
                device="cpu",
                contamination=0.2,
                pretrained=False,
                base_user_kwargs={},
                checkpoint_path=None,
                trained_checkpoint_path="/tmp/trained-model.pt",
                threshold=0.73,
                defects_payload=None,
                prediction_payload=None,
                defects_payload_source=None,
                illumination_contrast_knobs=None,
                tiling_payload=None,
                infer_config_postprocess=None,
                enable_maps_by_default=False,
                warnings=(),
            ),
            seed=123,
            user_kwargs={},
        ),
        create_detector=lambda name, **kwargs: created.update(
            name=str(name),
            kwargs=dict(kwargs),
        )
        or detector,
        load_checkpoint=lambda det, path: loaded.append(f"{id(det)}:{path}"),
    )

    assert result.model_name == "vision_ecod"
    assert result.detector is detector
    assert result.model_kwargs == created["kwargs"]
    assert created["name"] == "vision_ecod"
    assert created["kwargs"]["contamination"] == pytest.approx(0.2)
    assert loaded == [f"{id(detector)}:/tmp/trained-model.pt"]
    assert detector.threshold_ == pytest.approx(0.73)
