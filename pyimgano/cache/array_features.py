from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyimgano.utils.hash_utils import stable_hash_array, stable_hash_json


@dataclass(frozen=True)
class ArrayFeatureCache:
    """Disk cache for feature vectors derived from in-memory numpy images."""

    cache_dir: Path
    extractor_fingerprint: str

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_for_array(self, arr: np.ndarray) -> str:
        payload = {
            "fp": str(self.extractor_fingerprint),
            "arr": stable_hash_array(arr),
        }
        return stable_hash_json(payload)

    def _npy_path_for_key(self, key: str) -> Path:
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.npy"

    def load(self, arr: np.ndarray) -> np.ndarray | None:
        key = self._key_for_array(arr)
        npy_path = self._npy_path_for_key(key)
        if not npy_path.exists():
            return None
        try:
            return np.load(npy_path, allow_pickle=False)
        except Exception:
            return None

    def save(self, arr: np.ndarray, features_1d: np.ndarray) -> None:
        key = self._key_for_array(arr)
        npy_path = self._npy_path_for_key(key)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, np.asarray(features_1d), allow_pickle=False)
        os.replace(tmp_path, npy_path)


def extract_features_with_array_cache(
    *,
    feature_extractor: Any,
    arrays: Sequence[np.ndarray],
    cache: ArrayFeatureCache,
) -> np.ndarray:
    """Extract features for `arrays`, reusing disk cache when available."""

    arrays_list = [np.asarray(a) for a in arrays]
    cached_rows: list[np.ndarray | None] = [None] * len(arrays_list)
    missing_idx: list[int] = []

    for i, a in enumerate(arrays_list):
        row = cache.load(a)
        if row is None:
            missing_idx.append(i)
        else:
            cached_rows[i] = np.asarray(row).reshape(-1)

    if missing_idx:
        missing_arrays = [arrays_list[i] for i in missing_idx]
        feats_missing = np.asarray(feature_extractor.extract(missing_arrays))
        if feats_missing.ndim == 1:
            feats_missing = feats_missing.reshape(-1, 1)
        if feats_missing.shape[0] != len(missing_arrays):
            raise ValueError(
                "feature_extractor.extract must return one row per array. "
                f"Got shape {feats_missing.shape} for {len(missing_arrays)} arrays."
            )

        for j, idx in enumerate(missing_idx):
            row = np.asarray(feats_missing[j]).reshape(-1)
            cache.save(arrays_list[idx], row)
            cached_rows[idx] = row

    assert all(r is not None for r in cached_rows)
    return np.stack([np.asarray(r) for r in cached_rows], axis=0)


@dataclass(frozen=True)
class CachedArrayFeatureExtractor:
    """Wrap a feature extractor to add disk caching for numpy-array inputs."""

    base_extractor: Any
    cache: ArrayFeatureCache

    def extract(self, inputs):  # noqa: ANN001, ANN201 - adapter for various extractors
        items = list(inputs)
        if not items:
            return self.base_extractor.extract(items)

        # Cache only when every item is a numpy array (avoid mixed-mode surprises).
        if not all(isinstance(item, np.ndarray) for item in items):
            return self.base_extractor.extract(items)

        return extract_features_with_array_cache(
            feature_extractor=self.base_extractor,
            arrays=[np.asarray(x) for x in items],
            cache=self.cache,
        )
