"""Preset configs for industrial pipelines.

Presets are JSON-friendly dictionaries that describe recommended model setups.
They are intentionally lightweight: no weights, no assets, no network access.
"""

from __future__ import annotations

from .industrial_classical import INDUSTRIAL_CLASSICAL_PRESETS

__all__ = ["INDUSTRIAL_CLASSICAL_PRESETS"]

