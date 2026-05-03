import numpy as np
import pytest

from pyimgano.models.knn_index import FaissKNNIndex, SklearnKNNIndex, build_knn_index


def test_sklearn_knn_index_basic():
    rng = np.random.default_rng(0)
    x = rng.random((20, 4)).astype(np.float32)

    index = SklearnKNNIndex(n_neighbors=3)
    index.fit(x)
    distances, indices = index.kneighbors(x[:2])

    assert distances.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert np.all(np.isfinite(distances))


def test_build_knn_index_sklearn():
    index = build_knn_index(backend="sklearn", n_neighbors=5)
    assert index is not None


def test_build_knn_index_faiss_optional():
    pytest.importorskip("faiss", exc_type=ImportError)

    index = build_knn_index(backend="faiss", n_neighbors=3)
    assert isinstance(index, FaissKNNIndex)


def test_faiss_knn_index_optional():
    pytest.importorskip("faiss", exc_type=ImportError)

    rng = np.random.default_rng(0)
    x = rng.random((20, 4)).astype(np.float32)

    index = FaissKNNIndex(n_neighbors=3)
    index.fit(x)
    distances, indices = index.kneighbors(x[:2])

    assert distances.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert np.all(np.isfinite(distances))
