from __future__ import annotations

"""Best-effort disk cache for deep-model eval tensors / embeddings.

This cache is intended for **evaluation/inference** paths where transforms are
deterministic. It is not recommended for training transforms with randomness.

Why this exists:
- deep models often spend a lot of time decoding + preprocessing images
- caching preprocessed tensors can speed up repeated scoring runs

This is optional and safe-by-default: nothing is cached unless explicitly
enabled by the caller.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np


def fingerprint_transform(transform: Any) -> str:
    """Return a stable-ish fingerprint for a transform pipeline.

    Notes
    -----
    We avoid trying to introspect arbitrary callables. For torchvision transforms,
    `repr(transform)` is usually informative and stable enough for caching.
    """

    payload = repr(transform).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _path_stat_fingerprint(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    key = f"{p.resolve()}:{int(st.st_size)}:{int(st.st_mtime_ns)}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


@dataclass(frozen=True)
class TensorCache:
    cache_dir: Path
    transform_fingerprint: str

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, image_path: str | Path) -> str:
        base = _path_stat_fingerprint(image_path)
        payload = f"{base}:{self.transform_fingerprint}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def path_for(self, image_path: str | Path) -> Path:
        key = self._key(image_path)
        sub = self.cache_dir / key[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.npy"

    def load(self, image_path: str | Path) -> Optional[np.ndarray]:
        p = self.path_for(image_path)
        if not p.exists():
            return None
        try:
            return np.load(str(p), allow_pickle=False)
        except Exception:
            # Cache corruption: treat as miss.
            return None

    def save(self, image_path: str | Path, tensor_chw: np.ndarray) -> None:
        p = self.path_for(image_path)
        tmp = p.with_suffix(".tmp.npy")
        np.save(str(tmp), np.asarray(tensor_chw, dtype=np.float32))
        os.replace(str(tmp), str(p))


class CachedVisionImageDataset:
    """Image-path dataset that caches transformed CHW float tensors to disk."""

    def __init__(
        self,
        image_paths: Sequence[str],
        *,
        transform: Callable[[Any], Any],
        cache: TensorCache,
        fallback_shape: Tuple[int, int, int] = (3, 224, 224),
    ) -> None:
        self.image_paths = list(image_paths)
        self.transform = transform
        self.cache = cache
        self.fallback_shape = fallback_shape

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[int(idx)]

        cached = self.cache.load(path)
        if cached is not None:
            # Store tensors as float32 CHW.
            import torch

            x = torch.from_numpy(np.ascontiguousarray(cached)).float()
            return x, x

        try:
            from PIL import Image
            import torch

            pil = Image.open(path).convert("RGB")
            out = self.transform(pil)
            out_t = torch.as_tensor(out)
            if out_t.ndim != 3:
                raise ValueError(f"Transform must return CHW tensor, got shape {tuple(out_t.shape)}")
            arr = out_t.detach().cpu().numpy().astype(np.float32, copy=False)
            self.cache.save(path, arr)
            return out_t, out_t
        except Exception:
            import torch

            fallback = torch.zeros(self.fallback_shape, dtype=torch.float32)
            return fallback, fallback.clone()


__all__ = ["TensorCache", "fingerprint_transform", "CachedVisionImageDataset"]

