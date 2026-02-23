from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def fingerprint_feature_extractor(extractor: Any) -> str:
    """Return a stable fingerprint for a feature extractor (best-effort).

    This is used to avoid mixing cached features from different extractor
    configurations.
    """

    payload = {
        "type": f"{type(extractor).__module__}.{type(extractor).__qualname__}",
        "state": getattr(extractor, "__dict__", None),
    }
    try:
        encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    except Exception:
        encoded = repr(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureCache:
    cache_dir: Path
    extractor_fingerprint: str

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_for_path(self, path: str | Path) -> str:
        p = Path(path)
        meta: dict[str, Any] = {"fp": str(self.extractor_fingerprint)}
        try:
            st = p.stat()
        except FileNotFoundError:
            meta["path"] = str(p)
        else:
            meta.update(
                {
                    "path": str(p.resolve()),
                    "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
                    "size": int(st.st_size),
                }
            )
        encoded = json.dumps(meta, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _npy_path_for_key(self, key: str) -> Path:
        # Keep directory fanout reasonable to avoid huge single dirs.
        subdir = self.cache_dir / key[:2]
        return subdir / f"{key}.npy"

    def load(self, path: str | Path) -> np.ndarray | None:
        key = self._key_for_path(path)
        npy_path = self._npy_path_for_key(key)
        if not npy_path.exists():
            return None
        try:
            return np.load(npy_path, allow_pickle=False)
        except Exception:
            return None

    def save(self, path: str | Path, features_1d: np.ndarray) -> None:
        key = self._key_for_path(path)
        npy_path = self._npy_path_for_key(key)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, np.asarray(features_1d), allow_pickle=False)
        os.replace(tmp_path, npy_path)


def extract_features_with_cache(
    *,
    feature_extractor: Any,
    paths: Sequence[str],
    cache: FeatureCache,
) -> np.ndarray:
    """Extract features for `paths`, reusing disk cache when available."""

    paths_list = [str(p) for p in paths]
    cached_rows: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for p in paths_list:
        row = cache.load(p)
        if row is None:
            missing.append(p)
        else:
            cached_rows[p] = np.asarray(row).reshape(-1)

    if missing:
        feats_missing = np.asarray(feature_extractor.extract(list(missing)))
        if feats_missing.ndim == 1:
            feats_missing = feats_missing.reshape(-1, 1)
        if feats_missing.shape[0] != len(missing):
            raise ValueError(
                "feature_extractor.extract must return one row per path. "
                f"Got shape {feats_missing.shape} for {len(missing)} paths."
            )
        for i, p in enumerate(missing):
            row = np.asarray(feats_missing[i]).reshape(-1)
            cache.save(p, row)
            cached_rows[p] = row

    rows = [cached_rows[p] for p in paths_list]
    return np.stack(rows, axis=0)


@dataclass(frozen=True)
class CachedFeatureExtractor:
    """Wrap a feature extractor to add disk caching for path inputs."""

    base_extractor: Any
    cache: FeatureCache

    def extract(self, inputs):  # noqa: ANN001, ANN201 - adapter for various extractors
        items = list(inputs)
        if not items:
            return self.base_extractor.extract(items)
        if not all(isinstance(item, (str, Path)) for item in items):
            return self.base_extractor.extract(items)
        return extract_features_with_cache(
            feature_extractor=self.base_extractor,
            paths=[str(p) for p in items],
            cache=self.cache,
        )
