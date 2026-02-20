from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import optional_import, require


class KNNIndex(Protocol):
    def fit(self, X: NDArray) -> None: ...

    def kneighbors(self, X: NDArray, n_neighbors: Optional[int] = None) -> Tuple[NDArray, NDArray]: ...


@dataclass
class SklearnKNNIndex:
    n_neighbors: int
    metric: str = "euclidean"
    n_jobs: int = -1

    def __post_init__(self) -> None:
        sklearn_neighbors, error = optional_import("sklearn.neighbors")
        if sklearn_neighbors is None:
            raise ImportError(
                "scikit-learn is required for the sklearn KNN backend.\n"
                "Install it via:\n  pip install 'scikit-learn'\n"
                f"Original error: {error}"
            ) from error

        self._NearestNeighbors = sklearn_neighbors.NearestNeighbors  # type: ignore[attr-defined]
        self._index = self._NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: NDArray) -> None:
        self._index.fit(X)

    def kneighbors(
        self,
        X: NDArray,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        k = int(n_neighbors) if n_neighbors is not None else self.n_neighbors
        distances, indices = self._index.kneighbors(X, n_neighbors=k)
        return distances, indices


@dataclass
class FaissKNNIndex:
    n_neighbors: int

    def __post_init__(self) -> None:
        self._faiss = require("faiss", extra="faiss", purpose="FAISS kNN acceleration")
        self._index = None

    def fit(self, X: NDArray) -> None:
        X32 = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        if X32.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X32.shape}")

        dim = int(X32.shape[1])
        self._index = self._faiss.IndexFlatL2(dim)
        self._index.add(X32)

    def kneighbors(
        self,
        X: NDArray,
        n_neighbors: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray]:
        if self._index is None:
            raise RuntimeError("FAISS index not fitted. Call fit() first.")

        k = int(n_neighbors) if n_neighbors is not None else self.n_neighbors
        X32 = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        distances_sq, indices = self._index.search(X32, k)
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

