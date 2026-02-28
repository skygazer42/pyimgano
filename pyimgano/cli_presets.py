"""CLI model presets.

This is intentionally small: presets are just named (model, kwargs) pairs.
They help keep industrial command lines short while remaining fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class CLIPreset:
    name: str
    model: str
    kwargs: Mapping[str, Any]
    description: str
    optional: bool = False


@dataclass(frozen=True)
class DefectsPreset:
    """A JSON-friendly defects preset for `pyimgano-infer`.

    `payload` matches the workbench/infer-config defects schema so that the
    same defaults can be used in config files and in the CLI.
    """

    name: str
    payload: Mapping[str, Any]
    description: str


def _load_presets() -> dict[str, CLIPreset]:
    # Keep imports lightweight and avoid importing torch/torchvision here.
    from pyimgano.presets.industrial_classical import INDUSTRIAL_CLASSICAL_PRESETS

    out: dict[str, CLIPreset] = {}
    for name, preset in INDUSTRIAL_CLASSICAL_PRESETS.items():
        out[str(name)] = CLIPreset(
            name=str(preset.name),
            model=str(preset.model),
            kwargs=dict(preset.kwargs),
            description=str(preset.description),
            optional=bool(preset.optional),
        )

    # CLI-only aliases / opinionated presets (kept here so `industrial_classical`
    # remains JSON-ready and minimal).
    #
    # This preset is intentionally tuned for "embeddings + core" stability:
    # - no implicit weight downloads
    # - embedding-friendly core choice (Mahalanobis + shrinkage)
    # - rank standardization for a stable [0,1] score scale
    out["industrial-embedding-core-balanced"] = CLIPreset(
        name="industrial-embedding-core-balanced",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": {
                "backbone": "resnet18",
                "pretrained": False,
                "pool": "avg",
                "device": "cpu",
            },
            "core_detector": "core_score_standardizer",
            "core_kwargs": {
                "base_detector": "core_mahalanobis_shrinkage",
                "base_kwargs": {"assume_centered": False},
                "method": "rank",
            },
        },
        description="Balanced default: torchvision embeddings -> Mahalanobis shrinkage -> rank standardization.",
        optional=False,
    )
    return out


_PRESETS = _load_presets()


def _load_defects_presets() -> dict[str, DefectsPreset]:
    """Return defects presets (schema compatible with infer-config)."""

    return {
        "industrial-defects-fp40": DefectsPreset(
            name="industrial-defects-fp40",
            payload={
                # Match `examples/configs/industrial_adapt_defects_fp40.json`.
                "pixel_threshold_strategy": "normal_pixel_quantile",
                "pixel_normal_quantile": 0.999,
                "mask_format": "png",
                "roi_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                "border_ignore_px": 2,
                "map_smoothing": {"method": "median", "ksize": 3, "sigma": 0.0},
                "hysteresis": {"enabled": True, "low": None, "high": None},
                "shape_filters": {"min_fill_ratio": 0.15, "max_aspect_ratio": 6.0, "min_solidity": 0.8},
                "merge_nearby": {"enabled": True, "max_gap_px": 1},
                "min_area": 16,
                "min_score_max": 0.6,
                "min_score_mean": None,
                "open_ksize": 0,
                "close_ksize": 0,
                "fill_holes": False,
                "max_regions": 20,
                "max_regions_sort_by": "score_max",
            },
            description="False-positive reduction defaults for defects export (ROI/border/smoothing/hysteresis/shape filters).",
        )
    }


_DEFECTS_PRESETS = _load_defects_presets()


def list_model_presets() -> list[str]:
    """List available CLI preset names."""

    return sorted(_PRESETS.keys())


def list_defects_presets() -> list[str]:
    """List available defects preset names for `pyimgano-infer`."""

    return sorted(_DEFECTS_PRESETS.keys())


def resolve_model_preset(name: str) -> Optional[CLIPreset]:
    """Return CLIPreset if `name` is a known preset, else None."""

    key = str(name).strip()
    return _PRESETS.get(key, None)


def resolve_defects_preset(name: str) -> Optional[DefectsPreset]:
    """Return DefectsPreset if `name` is known, else None."""

    key = str(name).strip()
    return _DEFECTS_PRESETS.get(key, None)


__all__ = [
    "CLIPreset",
    "DefectsPreset",
    "list_model_presets",
    "resolve_model_preset",
    "list_defects_presets",
    "resolve_defects_preset",
]
