from __future__ import annotations

from typing import Any, Mapping

from pyimgano.workbench.config_parse_primitives import _optional_float, _require_mapping
from pyimgano.workbench.config_types import ModelConfig, OutputConfig


def _parse_model_config(top: Mapping[str, Any]) -> ModelConfig:
    model_raw = _require_mapping(top.get("model", {}), name="model")
    model_name = model_raw.get("name", None)
    if model_name is None:
        raise ValueError("model.name is required")

    mk_raw = model_raw.get("model_kwargs", None)
    if mk_raw is None:
        model_kwargs = {}
    else:
        model_kwargs = dict(_require_mapping(mk_raw, name="model.model_kwargs"))

    contamination = _optional_float(
        model_raw.get("contamination", 0.1),
        name="model.contamination",
    )
    return ModelConfig(
        name=str(model_name),
        device=str(model_raw.get("device", "cpu")),
        preset=(
            str(model_raw["preset"]) if model_raw.get("preset", None) is not None else None
        ),
        pretrained=bool(model_raw.get("pretrained", True)),
        contamination=float(contamination if contamination is not None else 0.1),
        model_kwargs=model_kwargs,
        checkpoint_path=(
            str(model_raw["checkpoint_path"])
            if model_raw.get("checkpoint_path", None) is not None
            else None
        ),
    )


def _parse_output_config(top: Mapping[str, Any]) -> OutputConfig:
    out_raw = top.get("output", None)
    if out_raw is None:
        return OutputConfig()

    out_map = _require_mapping(out_raw, name="output")
    return OutputConfig(
        output_dir=(
            str(out_map["output_dir"]) if out_map.get("output_dir", None) is not None else None
        ),
        save_run=bool(out_map.get("save_run", True)),
        per_image_jsonl=bool(out_map.get("per_image_jsonl", True)),
    )


__all__ = ["_parse_model_config", "_parse_output_config"]
