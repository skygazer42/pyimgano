# -*- coding: utf-8 -*-
"""PaDiM-lite (image-level Gaussian embedding baseline).

Full PaDiM models *patch-level* feature distributions and produces pixel maps.
This "lite" variant keeps the industrial core idea while staying lightweight:

- Extract one embedding per image (feature extractor)
- Fit a (robust) Gaussian model on normal embeddings
- Score by Mahalanobis distance (higher => more anomalous)

Implementation note:
This is an alias of `core_elliptic_envelope` / `vision_elliptic_envelope`
with PaDiM-oriented naming and tags.
"""

from __future__ import annotations

from .elliptic_envelope import CoreEllipticEnvelope, VisionEllipticEnvelope
from .registry import register_model


@register_model(
    "core_padim_lite",
    tags=("classical", "core", "features", "padim", "gaussian"),
    metadata={
        "description": "PaDiM-lite: Gaussian embedding baseline via robust covariance (Mahalanobis distance)",
    },
)
class CorePadimLite(CoreEllipticEnvelope):
    pass


@register_model(
    "vision_padim_lite",
    tags=("vision", "classical", "padim", "gaussian"),
    metadata={
        "description": "PaDiM-lite: embedding extractor + robust covariance baseline (image-level)",
    },
)
class VisionPadimLite(VisionEllipticEnvelope):
    pass
