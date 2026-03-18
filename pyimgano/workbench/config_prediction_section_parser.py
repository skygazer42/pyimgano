from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _optional_int,
    _require_mapping,
)
from pyimgano.workbench.config_types import PredictionConfig


def _parse_prediction_config(top: Mapping[str, Any]) -> PredictionConfig:
    prediction_raw = top.get("prediction", None)
    if prediction_raw is None:
        return PredictionConfig()

    p_map = _require_mapping(prediction_raw, name="prediction")
    reject_confidence_below = _optional_float(
        p_map.get("reject_confidence_below", None),
        name="prediction.reject_confidence_below",
    )
    reject_label = _optional_int(
        p_map.get("reject_label", None),
        name="prediction.reject_label",
    )

    if reject_confidence_below is not None and not (0.0 < float(reject_confidence_below) <= 1.0):
        raise ValueError("prediction.reject_confidence_below must be in (0,1] or null")

    return PredictionConfig(
        reject_confidence_below=(
            float(reject_confidence_below) if reject_confidence_below is not None else None
        ),
        reject_label=(int(reject_label) if reject_label is not None else None),
    )


__all__ = ["_parse_prediction_config"]
