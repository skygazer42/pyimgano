from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import (
    _optional_float,
    _optional_int,
    _parse_checkpoint_name,
    _require_mapping,
)
from pyimgano.workbench.config_types import TrainingConfig


def _parse_training_config(top: Mapping[str, Any]) -> TrainingConfig:
    training_raw = top.get("training", None)
    if training_raw is None:
        return TrainingConfig()

    t_map = _require_mapping(training_raw, name="training")
    epochs = _optional_int(t_map.get("epochs", None), name="training.epochs")
    if epochs is not None and epochs <= 0:
        raise ValueError("training.epochs must be positive or null")
    lr = _optional_float(t_map.get("lr", None), name="training.lr")
    if lr is not None and lr <= 0:
        raise ValueError("training.lr must be positive or null")
    return TrainingConfig(
        enabled=bool(t_map.get("enabled", False)),
        epochs=epochs,
        lr=lr,
        checkpoint_name=_parse_checkpoint_name(t_map.get("checkpoint_name", None)),
    )


__all__ = ["_parse_training_config"]
