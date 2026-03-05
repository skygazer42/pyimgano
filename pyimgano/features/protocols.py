"""Feature extractor protocols.

Classical detectors in PyImgAno work on 2D feature matrices. We standardize the
minimal interface to plug arbitrary feature extractors into `BaseVisionDetector`.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FeatureExtractor(Protocol):
    """Minimal protocol: provide `.extract(inputs) -> (n_samples, n_features)`."""

    def extract(self, inputs: Iterable[Any]) -> np.ndarray:  # pragma: no cover - protocol
        ...


@runtime_checkable
class FittableFeatureExtractor(FeatureExtractor, Protocol):
    """Optional protocol: extractor can be fit on training inputs."""

    def fit(
        self, inputs: Iterable[Any], y: Any | None = None
    ) -> "FittableFeatureExtractor":  # pragma: no cover
        ...
