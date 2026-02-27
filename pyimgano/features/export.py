from __future__ import annotations

"""Feature export helpers (features + stable IDs).

This is used for industrial workflows where downstream systems want both:
- the feature matrix (N, D)
- the stable ordering/identity of each row (ids)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureExport:
    ids: list[str]
    features: np.ndarray


def extract_features_with_ids(
    feature_extractor: Any,
    inputs: Iterable[Any],
    *,
    ids: Sequence[str] | None = None,
) -> FeatureExport:
    """Extract a feature matrix and return it with a stable list of row IDs."""

    items = list(inputs)
    if ids is None:
        derived: list[str] = []
        for i, it in enumerate(items):
            if isinstance(it, (str, Path)):
                derived.append(str(it))
            else:
                derived.append(str(i))
        ids_list = derived
    else:
        ids_list = [str(x) for x in list(ids)]
        if len(ids_list) != len(items):
            raise ValueError("ids length must match number of inputs")

    feats = np.asarray(feature_extractor.extract(items))
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.shape[0] != len(items):
        raise ValueError(
            "feature_extractor.extract must return one row per input. "
            f"Got shape {feats.shape} for {len(items)} inputs."
        )

    return FeatureExport(ids=ids_list, features=np.asarray(feats, dtype=np.float32))


def save_feature_export(path: str | Path, export: FeatureExport) -> Path:
    """Save a :class:`FeatureExport` as an `.npz` file."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ids_arr = np.asarray(list(export.ids), dtype=str)
    feats = np.asarray(export.features, dtype=np.float32)
    np.savez(str(p), ids=ids_arr, features=feats)
    return p


def load_feature_export(path: str | Path) -> FeatureExport:
    p = Path(path)
    with np.load(str(p), allow_pickle=False) as data:
        ids = [str(x) for x in data["ids"].tolist()]
        feats = np.asarray(data["features"], dtype=np.float32)
    return FeatureExport(ids=ids, features=feats)

