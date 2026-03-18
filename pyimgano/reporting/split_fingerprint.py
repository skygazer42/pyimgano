from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SPLIT_FINGERPRINT_SCHEMA_VERSION = 1


def _stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize_path(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value)
    if text == "":
        return ""
    return posixpath.normpath(text.replace("\\", "/"))


def _canonicalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _canonicalize_json_value(value.item())
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return float(value)
        return repr(value)
    if isinstance(value, Path):
        return _normalize_path(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonicalize_array(value)
    return repr(value)


def _canonicalize_array(value: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(value)
    contiguous = np.ascontiguousarray(arr)
    return {
        "kind": "ndarray",
        "dtype": str(contiguous.dtype),
        "shape": [int(dim) for dim in contiguous.shape],
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _canonicalize_input(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        return {"kind": "path", "value": _normalize_path(value)}
    if isinstance(value, np.ndarray):
        return _canonicalize_array(value)
    return {"kind": "value", "value": _canonicalize_json_value(value)}


def _sort_canonical_items(items: Sequence[Any]) -> list[Any]:
    return sorted(list(items), key=_stable_json_dumps)


def _normalize_test_labels(test_labels: Sequence[Any] | np.ndarray) -> list[Any]:
    labels = np.asarray(test_labels)
    if labels.ndim == 0:
        return [_canonicalize_json_value(labels.item())]
    return [_canonicalize_json_value(item) for item in labels.reshape(-1).tolist()]


def build_split_fingerprint(
    *,
    train_inputs: Sequence[Any],
    calibration_inputs: Sequence[Any],
    test_inputs: Sequence[Any],
    test_labels: Sequence[Any] | np.ndarray,
    input_format: str | None = None,
    test_meta: Sequence[Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    train_items = list(train_inputs)
    calibration_items = list(calibration_inputs)
    test_items = list(test_inputs)
    labels = _normalize_test_labels(test_labels)

    if len(test_items) != len(labels):
        raise ValueError(
            "split fingerprint requires len(test_inputs) == len(test_labels). "
            f"Got {len(test_items)} and {len(labels)}."
        )

    meta_items: list[Mapping[str, Any] | None] | None = None
    if test_meta is not None:
        meta_items = list(test_meta)
        if len(meta_items) != len(test_items):
            raise ValueError(
                "split fingerprint requires len(test_meta) == len(test_inputs) when test_meta is provided. "
                f"Got {len(meta_items)} and {len(test_items)}."
            )

    canonical_payload = {
        "schema_version": int(SPLIT_FINGERPRINT_SCHEMA_VERSION),
        "input_format": (str(input_format) if input_format is not None else None),
        "train": _sort_canonical_items([_canonicalize_input(item) for item in train_items]),
        "calibration": _sort_canonical_items(
            [_canonicalize_input(item) for item in calibration_items]
        ),
        "test": _sort_canonical_items(
            [
                {
                    "input": _canonicalize_input(item),
                    "label": labels[index],
                    "meta": (
                        _canonicalize_json_value(meta_items[index])
                        if meta_items is not None
                        else None
                    ),
                }
                for index, item in enumerate(test_items)
            ]
        ),
    }
    sha256 = hashlib.sha256(_stable_json_dumps(canonical_payload).encode("utf-8")).hexdigest()

    return {
        "schema_version": int(SPLIT_FINGERPRINT_SCHEMA_VERSION),
        "sha256": sha256,
        "train_count": int(len(train_items)),
        "calibration_count": int(len(calibration_items)),
        "test_count": int(len(test_items)),
        "input_format": (str(input_format) if input_format is not None else None),
    }


__all__ = [
    "SPLIT_FINGERPRINT_SCHEMA_VERSION",
    "build_split_fingerprint",
]
