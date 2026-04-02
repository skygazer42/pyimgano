from __future__ import annotations

from typing import Any, Mapping, TypeVar

from pyimgano.workbench.config_parse_primitives import _optional_int, _require_mapping
from pyimgano.workbench.config_section_parsers import (
    _parse_adaptation_config,
    _parse_dataset_config,
    _parse_defects_config,
    _parse_model_config,
    _parse_output_config,
    _parse_prediction_config,
    _parse_preprocessing_config,
    _parse_training_config,
)
from pyimgano.workbench.config_types import MetaConfig
from pyimgano.workbench.config_types import WorkbenchConfig

WorkbenchConfigT = TypeVar("WorkbenchConfigT", bound=WorkbenchConfig)


def _parse_meta_config(top: Mapping[str, Any]) -> MetaConfig:
    raw_meta = top.get("meta", None)
    if raw_meta is None:
        return MetaConfig()

    meta = _require_mapping(raw_meta, name="meta")

    def _optional_text(name: str) -> str | None:
        value = meta.get(name, None)
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError(f"meta.{name} must be a non-empty string or null")
        return text

    def _string_list(name: str) -> tuple[str, ...]:
        value = meta.get(name, ())
        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"meta.{name} must be a list of strings or null")
        out: list[str] = []
        for index, item in enumerate(value):
            text = str(item).strip()
            if not text:
                raise ValueError(f"meta.{name}[{index}] must be a non-empty string")
            out.append(text)
        return tuple(out)

    return MetaConfig(
        purpose=_optional_text("purpose"),
        runtime_profile=_optional_text("runtime_profile"),
        required_extras=_string_list("required_extras"),
        expected_artifacts=_string_list("expected_artifacts"),
    )


def build_workbench_config_from_dict(
    raw: Mapping[str, Any],
    *,
    config_cls: type[WorkbenchConfigT] = WorkbenchConfig,
) -> WorkbenchConfigT:
    top = _require_mapping(raw, name="config")

    recipe = str(top.get("recipe", "industrial-adapt"))
    seed = _optional_int(top.get("seed", None), name="seed")

    return config_cls(
        dataset=_parse_dataset_config(top, seed=seed),
        model=_parse_model_config(top),
        recipe=recipe,
        seed=seed,
        meta=_parse_meta_config(top),
        output=_parse_output_config(top),
        adaptation=_parse_adaptation_config(top),
        preprocessing=_parse_preprocessing_config(top),
        training=_parse_training_config(top),
        defects=_parse_defects_config(top),
        prediction=_parse_prediction_config(top),
    )


__all__ = ["build_workbench_config_from_dict"]
