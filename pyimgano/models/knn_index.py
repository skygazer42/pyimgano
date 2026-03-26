from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import optional_import, require


class KNNIndex(Protocol):
    def fit(self, x: NDArray) -> None:
        ...

    def kneighbors(self, x: NDArray, n_neighbors: Optional[int] = None) -> Tuple[NDArray, NDArray]:
        ...


@dataclass
class SklearnKNNIndex:
    _legacy_attr_aliases = {"_NearestNeighbors": "_nearest_neighbors_cls"}

    n_neighbors: int
    metric: str = "euclidean"
    n_jobs: int = -1

    def __getattr__(self, name: str):
        alias = type(self)._legacy_attr_aliases.get(name)
        if alias is not None:
            return getattr(self, alias)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value) -> None:
        alias = type(self)._legacy_attr_aliases.get(name)
        super().__setattr__(alias or name, value)

    def __post_init__(self) -> None:
        sklearn_neighbors, error = optional_import("sklearn.neighbors")
        if sklearn_neighbors is None:
            raise ImportError(
                "scikit-learn is required for the sklearn KNN backend.\n"
                "Install it via:\n  pip install 'scikit-learn'\n"
                f"Original error: {error}"
            ) from error

        self._nearest_neighbors_cls = sklearn_neighbors.NearestNeighbors  # type: ignore[attr-defined]
        self._index = self._nearest_neighbors_cls(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def fit(self, x: NDArray) -> None:
        self._index.fit(x)

    def kneighbors(
        self,
        x: NDArray,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        k = int(n_neighbors) if n_neighbors is not None else self.n_neighbors
        distances, indices = self._index.kneighbors(x, n_neighbors=k)
        return distances, indices


@dataclass
class FaissKNNIndex:
    n_neighbors: int

    def __post_init__(self) -> None:
        self._faiss = require("faiss", extra="faiss", purpose="FAISS kNN acceleration")
        self._index = None

    def fit(self, x: NDArray) -> None:
        x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if x32.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x32.shape}")

        dim = int(x32.shape[1])
        self._index = self._faiss.IndexFlatL2(dim)
        self._index.add(x32)

    def kneighbors(
        self,
        x: NDArray,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        if self._index is None:
            raise RuntimeError("FAISS index not fitted. Call fit() first.")

        k = int(n_neighbors) if n_neighbors is not None else self.n_neighbors
        x32 = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        distances_sq, indices = self._index.search(x32, k)
        distances = np.sqrt(distances_sq)
        return distances, indices


def build_knn_index(
    *,
    backend: str,
    n_neighbors: int,
    metric: str = "euclidean",
    n_jobs: int = -1,
) -> KNNIndex:
    backend_lower = backend.lower()
    if backend_lower in ("sklearn", "scikit", "scikit-learn"):
        return SklearnKNNIndex(n_neighbors=n_neighbors, metric=metric, n_jobs=n_jobs)
    if backend_lower in ("faiss",):
        return FaissKNNIndex(n_neighbors=n_neighbors)
    raise ValueError(f"Unknown KNN backend: {backend}. Choose from: sklearn, faiss")
