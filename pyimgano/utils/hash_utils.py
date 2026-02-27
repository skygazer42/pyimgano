"""Stable hashing utilities (deterministic across processes).

These helpers are used for disk caches and for best-effort reproducibility.
They intentionally avoid Python's built-in `hash()` which is randomized per
process unless PYTHONHASHSEED is fixed.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash_str(text: str) -> str:
    return sha256_hex(str(text).encode("utf-8"))


def stable_hash_json(obj: Any) -> str:
    """Hash a JSON-ish object (best-effort).

    Falls back to `repr` for non-serializable values.
    """

    try:
        encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=repr).encode(
            "utf-8"
        )
    except Exception:
        encoded = repr(obj).encode("utf-8")
    return sha256_hex(encoded)


def stable_hash_array(arr: Any) -> str:
    """Hash a numpy array by (dtype, shape, bytes) using SHA-256."""

    a = np.asarray(arr)
    if a.dtype == object:
        raise ValueError("Object arrays are not supported for stable_hash_array().")

    meta = {
        "dtype": str(a.dtype),
        "shape": tuple(int(x) for x in a.shape),
        "order": "C",
    }
    h = hashlib.sha256()
    h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    h.update(memoryview(a).tobytes())
    return h.hexdigest()

