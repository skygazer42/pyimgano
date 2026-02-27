"""Base classes for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
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

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return init parameters (sklearn-style best-effort).

        This mirrors the detector `get_params/set_params` API, which makes
        feature extractors easier to serialize in workbench configs.
        """

        sig = inspect.signature(self.__class__.__init__)
        out: dict[str, Any] = {}
        for name in sig.parameters:
            if name == "self":
                continue
            if hasattr(self, name):
                out[name] = getattr(self, name)

        if not deep:
            return out

        nested: dict[str, Any] = {}
        for k, v in out.items():
            if hasattr(v, "get_params") and callable(getattr(v, "get_params")):
                try:
                    for nk, nv in v.get_params(deep=True).items():  # type: ignore[call-arg]
                        nested[f"{k}__{nk}"] = nv
                except Exception:
                    continue
        out.update(nested)
        return out

    def set_params(self, **params: Any) -> "BaseFeatureExtractor":
        """Set init parameters (sklearn-style best-effort)."""

        valid = self.get_params(deep=False)
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Invalid parameter {k!r} for {self.__class__.__name__}.")
            setattr(self, k, v)
        return self
