from __future__ import annotations

from dataclasses import fields

import numpy as np

from pyimgano.inference.api import InferenceResult
from pyimgano.services.infer_artifact_service import (
    InferArtifactOptions,
    InferResultArtifactAssemblyRequest,
    build_infer_result_artifact_request_from_options,
)
from pyimgano.services.infer_context_service import ConfigBackedInferContext
from pyimgano.services.infer_load_service import ConfigBackedInferLoadRequest
from pyimgano.services.infer_output_service import InferOutputWriteResult
from pyimgano.services.infer_runtime_service import InferRuntimePlanResult


def test_config_backed_infer_context_contract_fields_are_stable() -> None:
    assert [field.name for field in fields(ConfigBackedInferContext)] == [
        "model_name",
        "preset",
        "device",
        "contamination",
        "pretrained",
        "base_user_kwargs",
        "checkpoint_path",
        "trained_checkpoint_path",
        "threshold",
        "defects_payload",
        "prediction_payload",
        "defects_payload_source",
        "illumination_contrast_knobs",
        "tiling_payload",
        "infer_config_postprocess",
        "enable_maps_by_default",
        "postprocess_summary",
        "warnings",
    ]


def test_selected_infer_result_contract_fields_are_stable() -> None:
    assert [field.name for field in fields(ConfigBackedInferLoadRequest)] == [
        "context",
        "seed",
        "user_kwargs",
    ]
    assert [field.name for field in fields(InferRuntimePlanResult)] == [
        "include_maps",
        "postprocess",
        "pixel_threshold_value",
        "pixel_threshold_provenance",
        "postprocess_summary",
    ]
    assert [field.name for field in fields(InferOutputWriteResult)] == [
        "output_written",
        "regions_written",
    ]


def test_build_infer_result_artifact_request_from_options_copies_mutable_option_payloads() -> None:
    roi_xyxy_norm = [0.1, 0.2, 0.9, 0.8]
    pixel_threshold_provenance = {"source": "infer_config"}

    request = build_infer_result_artifact_request_from_options(
        InferResultArtifactAssemblyRequest(
            index=0,
            input_path="sample.png",
            result=InferenceResult(
                score=0.5, label=0, anomaly_map=np.zeros((2, 2), dtype=np.float32)
            ),
            options=InferArtifactOptions(
                defects_enabled=True,
                pixel_threshold_value=0.5,
                pixel_threshold_provenance=pixel_threshold_provenance,
                roi_xyxy_norm=roi_xyxy_norm,
            ),
        )
    )

    roi_xyxy_norm[0] = 0.0
    pixel_threshold_provenance["source"] = "mutated"

    assert request.defects is not None
    assert request.defects.roi_xyxy_norm == [0.1, 0.2, 0.9, 0.8]
    assert request.defects.pixel_threshold_provenance == {"source": "infer_config"}
