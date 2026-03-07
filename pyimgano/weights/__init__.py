"""Weights utilities (manifest + validation).

`pyimgano` does not ship large pretrained weights inside the wheel. In production,
weights are expected to live on disk and be referenced explicitly.
"""

from __future__ import annotations

from pyimgano.weights.manifest import validate_weights_manifest_file

__all__ = [
    "validate_weights_manifest_file",
]
