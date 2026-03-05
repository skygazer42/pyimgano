from __future__ import annotations

"""Reporting schema helpers.

This module centralizes small, schema-shaping helpers so reporting code can stay
focused on producing results rather than on conditional payload wiring.

The intent is to keep reports:
- JSON-friendly
- stable across versions
- explicit about industrial context (e.g. reference-based inspection)
"""

from pathlib import Path
from typing import Any


def extract_reference_context(detector: Any) -> dict[str, Any] | None:
    """Return a small 'reference' context block when a detector is reference-based.

    The v4 reference-based pipeline convention is:
    - detector has a `reference_dir` attribute (str|Path)
    - optionally a `match_mode` attribute (e.g. "basename")
    """

    ref = getattr(detector, "reference_dir", None)
    if ref is None:
        return None

    match_mode = getattr(detector, "match_mode", None)
    out: dict[str, Any] = {
        "enabled": True,
        "reference_dir": str(ref),
    }
    if match_mode is not None:
        out["match_mode"] = str(match_mode)

    # Optional: best-effort existence flag (helpful in CI artifacts).
    try:
        out["reference_dir_exists"] = bool(Path(str(ref)).exists())
    except Exception:
        out["reference_dir_exists"] = None

    return out


__all__ = ["extract_reference_context"]
