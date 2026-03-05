from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint_payload(payload: Any) -> str:
    """Hash an arbitrary JSON-ish payload into a stable-ish fingerprint."""

    try:
        encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    except Exception:
        encoded = repr(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EmbeddingCache:
    """Disk cache for per-path embedding rows.

    Keys are computed from:
    - absolute resolved path
    - file size + mtime_ns (best-effort)
    - extractor_fingerprint (configuration)
    """

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

    def save(self, path: str | Path, embedding_1d: np.ndarray) -> None:
        key = self._key_for_path(path)
        npy_path = self._npy_path_for_key(key)
        npy_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = npy_path.with_suffix(npy_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, np.asarray(embedding_1d), allow_pickle=False)
        os.replace(tmp_path, npy_path)


__all__ = ["EmbeddingCache", "fingerprint_payload"]
