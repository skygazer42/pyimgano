"""Base classes for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np


class BaseFeatureExtractor(ABC):
    """A minimal base class for feature extractors.

    Subclasses must implement :meth:`extract`. Optionally override :meth:`fit`.
    """

    def fit(self, inputs: Iterable[Any], y: Any | None = None) -> "BaseFeatureExtractor":
        return self

    @abstractmethod
    def extract(self, inputs: Iterable[Any]) -> np.ndarray:
        raise NotImplementedError

    def fit_extract(self, inputs: Iterable[Any], y: Any | None = None) -> np.ndarray:
        self.fit(inputs, y=y)
        return self.extract(inputs)

    def extract_batched(self, inputs: Iterable[Any], *, batch_size: int) -> np.ndarray:
        """Extract features in batches.

        This is a convenience helper for extractors that can process a small
        list of inputs at a time (e.g. when decoding images).
        """

        bs = int(batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

        items = list(inputs)
        if not items:
            return np.asarray(self.extract(items))

        rows: list[np.ndarray] = []
        for i in range(0, len(items), bs):
            batch = items[i : i + bs]
            feats = np.asarray(self.extract(batch))
            if feats.ndim == 1:
                feats = feats.reshape(1, -1)
            rows.append(feats)

        return np.concatenate(rows, axis=0)
