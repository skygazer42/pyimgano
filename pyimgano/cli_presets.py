"""Compatibility wrapper over the shared preset catalog."""

from __future__ import annotations

import pyimgano.presets.catalog as preset_catalog

CLIPreset = preset_catalog.CLIPreset
DefectsPreset = preset_catalog.DefectsPreset


def list_model_presets(*, tags=None):
    return preset_catalog.list_model_presets(tags=tags)


def list_defects_presets() -> list[str]:
    return preset_catalog.list_defects_presets()


def list_preprocessing_presets() -> list[str]:
    return preset_catalog.list_preprocessing_presets()


def model_preset_info(name: str) -> dict:
    return preset_catalog.model_preset_info(name)


def list_model_preset_infos(*, tags=None) -> list[dict]:
    return preset_catalog.list_model_preset_infos(tags=tags)


def resolve_model_preset(name: str):
    return preset_catalog.resolve_model_preset(name)


def resolve_defects_preset(name: str):
    return preset_catalog.resolve_defects_preset(name)


def resolve_preprocessing_preset(name: str):
    return preset_catalog.resolve_preprocessing_preset(name)


__all__ = [
    "CLIPreset",
    "DefectsPreset",
    "list_model_presets",
    "list_model_preset_infos",
    "model_preset_info",
    "resolve_model_preset",
    "list_defects_presets",
    "resolve_defects_preset",
    "list_preprocessing_presets",
    "resolve_preprocessing_preset",
]
