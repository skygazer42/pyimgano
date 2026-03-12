"""Neutral preset catalog shared by services and CLI adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from pyimgano.discovery import resolve_family_tags


@dataclass(frozen=True)
class CLIPreset:
    name: str
    model: str
    kwargs: Mapping[str, Any]
    description: str
    optional: bool = False
    requires_extras: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class DefectsPreset:
    """A JSON-friendly defects preset payload."""

    name: str
    payload: Mapping[str, Any]
    description: str


def _load_presets() -> dict[str, CLIPreset]:
    from pyimgano.presets.industrial_classical import INDUSTRIAL_CLASSICAL_PRESETS

    out: dict[str, CLIPreset] = {}
    for name, preset in INDUSTRIAL_CLASSICAL_PRESETS.items():
        out[str(name)] = CLIPreset(
            name=str(preset.name),
            model=str(preset.model),
            kwargs=dict(preset.kwargs),
            description=str(preset.description),
            optional=bool(preset.optional),
            requires_extras=tuple(getattr(preset, "requires_extras", ())),
            tags=tuple(getattr(preset, "tags", ())),
        )

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
        optional=True,
        requires_extras=("torch",),
        tags=("embeddings", "gaussian", "calibration", "classical"),
    )
    return out


_PRESETS = _load_presets()


def _load_defects_presets() -> dict[str, DefectsPreset]:
    return {
        "industrial-defects-fp40": DefectsPreset(
            name="industrial-defects-fp40",
            payload={
                "pixel_threshold_strategy": "normal_pixel_quantile",
                "pixel_normal_quantile": 0.999,
                "mask_format": "png",
                "roi_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                "border_ignore_px": 2,
                "map_smoothing": {"method": "median", "ksize": 3, "sigma": 0.0},
                "hysteresis": {"enabled": True, "low": None, "high": None},
                "shape_filters": {
                    "min_fill_ratio": 0.15,
                    "max_aspect_ratio": 6.0,
                    "min_solidity": 0.8,
                },
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


def _load_preprocessing_presets():
    from pyimgano.preprocessing.catalog import list_preprocessing_schemes

    out = {}
    for scheme in list_preprocessing_schemes(deployable_only=True):
        if scheme.payload is None or scheme.config_key is None:
            continue
        out[str(scheme.name)] = scheme
    return out


_PREPROCESSING_PRESETS = _load_preprocessing_presets()


def _normalize_tag_filter(tags: Optional[Iterable[str]]) -> list[str]:
    out: list[str] = []
    if tags is None:
        return out
    for item in tags:
        for piece in str(item).split(","):
            tag = piece.strip()
            if tag:
                out.append(tag)
    return out


def resolve_model_preset_filter_tags(
    *,
    tags: Optional[Iterable[str]] = None,
    family: str | None = None,
) -> list[str]:
    out = _normalize_tag_filter(tags)
    if family is not None:
        out.extend(resolve_family_tags(str(family)))
    return out


def list_model_presets(*, tags: Optional[Iterable[str]] = None) -> list[str]:
    required = set(_normalize_tag_filter(tags))
    if not required:
        return sorted(_PRESETS.keys())
    return sorted(name for name, preset in _PRESETS.items() if required.issubset(set(preset.tags)))


def list_defects_presets() -> list[str]:
    return sorted(_DEFECTS_PRESETS.keys())


def list_preprocessing_presets() -> list[str]:
    return sorted(_PREPROCESSING_PRESETS.keys())


def model_preset_info(name: str) -> dict[str, Any]:
    preset = resolve_model_preset(name)
    if preset is None:
        raise KeyError(f"Unknown model preset: {name!r}")
    return {
        "name": preset.name,
        "model": preset.model,
        "kwargs": dict(preset.kwargs),
        "description": preset.description,
        "optional": bool(preset.optional),
        "requires_extras": list(preset.requires_extras),
        "tags": list(preset.tags),
    }


def list_model_preset_infos(*, tags: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
    return [model_preset_info(name) for name in list_model_presets(tags=tags)]


def resolve_model_preset(name: str) -> Optional[CLIPreset]:
    return _PRESETS.get(str(name).strip(), None)


def resolve_defects_preset(name: str) -> Optional[DefectsPreset]:
    return _DEFECTS_PRESETS.get(str(name).strip(), None)


def resolve_preprocessing_preset(name: str):
    return _PREPROCESSING_PRESETS.get(str(name).strip(), None)


__all__ = [
    "CLIPreset",
    "DefectsPreset",
    "list_defects_presets",
    "list_model_preset_infos",
    "list_model_presets",
    "list_preprocessing_presets",
    "model_preset_info",
    "resolve_defects_preset",
    "resolve_model_preset_filter_tags",
    "resolve_model_preset",
    "resolve_preprocessing_preset",
]
