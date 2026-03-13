from __future__ import annotations

from typing import Any

from pyimgano.presets.catalog import resolve_defects_preset, resolve_preprocessing_preset
from pyimgano.services.infer_defects_defaults_service import apply_defects_defaults


def resolve_defects_preset_payload(name: str | None) -> dict[str, Any] | None:
    if name is None:
        return None

    preset = resolve_defects_preset(str(name))
    if preset is None:
        raise ValueError(f"Unknown defects preset: {name!r}")

    return dict(preset.payload)


def resolve_preprocessing_preset_knobs(name: str | None):
    if name is None:
        return None

    from pyimgano.inference.preprocessing import parse_illumination_contrast_knobs

    preset = resolve_preprocessing_preset(str(name))
    if preset is None:
        raise ValueError(f"Unknown preprocessing preset: {name!r}")
    if str(getattr(preset, "config_key", "")) != "preprocessing.illumination_contrast":
        raise ValueError(
            "Only preprocessing.illumination_contrast presets are currently supported by pyimgano-infer."
        )

    payload = dict(getattr(preset, "payload", {}) or {})
    return parse_illumination_contrast_knobs(payload)


__all__ = [
    "apply_defects_defaults",
    "resolve_defects_preset_payload",
    "resolve_preprocessing_preset_knobs",
]
