"""Curated baseline suites (import-light).

Suites are collections of model presets meant for industrial algorithm
selection. They are used by `pyimgano-benchmark --suite <name>`.

This module must remain import-light: do not import optional heavy dependencies
here. Use preset names + JSON-ready kwargs only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class Baseline:
    """A single runnable baseline entry (resolved to model + kwargs)."""

    name: str
    model: str
    kwargs: Mapping[str, Any]
    description: str
    optional: bool = False
    requires_extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineSuite:
    name: str
    description: str
    # Entries are preset names (preferred) or direct model names.
    entries: tuple[str, ...]


def _default_suites() -> dict[str, BaselineSuite]:
    return {
        "industrial-ci": BaselineSuite(
            name="industrial-ci",
            description="Fast, core-only industrial baselines for CI/smoke selection (no torch/skimage required).",
            entries=(
                "industrial-structural-ecod",
                "industrial-pixel-mean-absdiff-map",
                "industrial-template-ncc-map",
            ),
        ),
        "industrial-v1": BaselineSuite(
            name="industrial-v1",
            description=(
                "Recommended industrial baseline pack (core + optional extras when installed). "
                "Includes structural score-only, pixel-first template baselines, and (optional) "
                "embedding-based baselines."
            ),
            entries=(
                # Core (lightweight)
                "industrial-structural-ecod",
                "industrial-structural-iforest",
                "industrial-pixel-mean-absdiff-map",
                "industrial-pixel-gaussian-map",
                "industrial-pixel-mad-map",
                "industrial-template-ncc-map",
                # Optional (skimage)
                "industrial-ssim-template-map",
                "industrial-ssim-struct-map",
                "industrial-phase-correlation-map",
                # Optional (torch)
                "industrial-embed-mahalanobis-shrinkage-rank",
                "industrial-embed-knn-cosine",
            ),
        ),
        "industrial-v2": BaselineSuite(
            name="industrial-v2",
            description=(
                "Expanded industrial suite (v1 + structural MST + optional OpenCLIP). "
                "Use for broader offline selection; optional entries are skipped when extras are missing."
            ),
            entries=(
                # Core (lightweight)
                "industrial-structural-ecod",
                "industrial-structural-iforest",
                "industrial-structural-mst",
                "industrial-pixel-mean-absdiff-map",
                "industrial-pixel-gaussian-map",
                "industrial-pixel-mad-map",
                "industrial-template-ncc-map",
                # Optional (skimage)
                "industrial-ssim-template-map",
                "industrial-ssim-struct-map",
                "industrial-phase-correlation-map",
                # Optional (torch)
                "industrial-embed-mahalanobis-shrinkage-rank",
                "industrial-embed-knn-cosine",
                # Optional (clip + torch)
                "industrial-openclip-knn",
            ),
        ),
        "industrial-v3": BaselineSuite(
            name="industrial-v3",
            description=(
                "Expanded industrial suite (v2 + optional deep pixel-map baselines). "
                "Includes PatchCore-lite-map and patch-embedding-core-map for pixel-level anomaly maps when "
                "torch extras are installed."
            ),
            entries=(
                # Core (lightweight)
                "industrial-structural-ecod",
                "industrial-structural-iforest",
                "industrial-structural-mst",
                "industrial-pixel-mean-absdiff-map",
                "industrial-pixel-gaussian-map",
                "industrial-pixel-mad-map",
                "industrial-template-ncc-map",
                # Optional (skimage)
                "industrial-ssim-template-map",
                "industrial-ssim-struct-map",
                "industrial-phase-correlation-map",
                # Optional (torch)
                "industrial-embed-mahalanobis-shrinkage-rank",
                "industrial-embed-knn-cosine",
                "industrial-patchcore-lite-map",
                "industrial-patch-embedding-core-map",
                # Optional (clip + torch)
                "industrial-openclip-knn",
            ),
        ),
    }


_SUITES = _default_suites()


def list_baseline_suites() -> list[str]:
    return sorted(_SUITES.keys())


def get_baseline_suite(name: str) -> BaselineSuite:
    key = str(name).strip()
    if key not in _SUITES:
        available = ", ".join(list_baseline_suites()) or "<none>"
        raise KeyError(f"Unknown baseline suite {name!r}. Available: {available}")
    return _SUITES[key]


def _resolve_entry(ref: str) -> Optional[Baseline]:
    """Resolve a suite entry ref into a runnable (model, kwargs) baseline.

    Resolution order:
    1) CLI model preset name from `pyimgano.cli_presets`
    2) Direct registered model name (kwargs empty)
    """

    from pyimgano.cli_presets import resolve_model_preset

    preset = resolve_model_preset(str(ref))
    if preset is not None:
        return Baseline(
            name=str(preset.name),
            model=str(preset.model),
            kwargs=dict(preset.kwargs),
            description=str(preset.description),
            optional=bool(preset.optional),
            requires_extras=tuple(getattr(preset, "requires_extras", ())),
        )

    # Fall back to a direct model reference. We intentionally do not validate
    # against the registry here to keep discovery import-light.
    return Baseline(
        name=str(ref),
        model=str(ref),
        kwargs={},
        description=f"Direct model baseline: {ref}",
        optional=False,
        requires_extras=(),
    )


def resolve_suite_baselines(name: str) -> list[Baseline]:
    suite = get_baseline_suite(name)
    out: list[Baseline] = []
    for ref in suite.entries:
        b = _resolve_entry(str(ref))
        if b is not None:
            out.append(b)
    return out


__all__ = [
    "Baseline",
    "BaselineSuite",
    "list_baseline_suites",
    "get_baseline_suite",
    "resolve_suite_baselines",
]
