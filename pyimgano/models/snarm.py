"""
Experimental SNARM-inspired entry point built on the MambaAD backend.

This module exists to expose a stable ``vision_snarm`` model name in the
registry while keeping the implementation lightweight and optional behind
``pyimgano[mamba]``. It intentionally reuses the existing Mamba-based patch
reconstruction pipeline rather than claiming a full paper reproduction.
"""

from __future__ import annotations

from .mambaad import VisionMambaAD
from .registry import register_model


@register_model(
    "vision_snarm",
    tags=("vision", "deep", "snarm", "mamba", "ssm", "numpy", "pixel_map"),
    metadata={
        "description": (
            "Experimental SNARM-inspired patch reconstruction detector reusing the "
            "MambaAD backend"
        ),
        "paper": "SNARM",
        "year": 2025,
        "requires_optional_deps": ["mamba-ssm"],
        "experimental": True,
        "notes": (
            "Lightweight compatibility entry point built on VisionMambaAD; not a full "
            "paper reproduction."
        ),
    },
)
class VisionSNARM(VisionMambaAD):
    """Experimental SNARM-inspired detector backed by VisionMambaAD."""

