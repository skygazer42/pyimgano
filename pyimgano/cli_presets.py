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
    return out


_PRESETS = _load_presets()


def list_model_presets() -> list[str]:
    """List available CLI preset names."""

    return sorted(_PRESETS.keys())


def resolve_model_preset(name: str) -> Optional[CLIPreset]:
    """Return CLIPreset if `name` is a known preset, else None."""

    key = str(name).strip()
    return _PRESETS.get(key, None)


__all__ = ["CLIPreset", "list_model_presets", "resolve_model_preset"]

