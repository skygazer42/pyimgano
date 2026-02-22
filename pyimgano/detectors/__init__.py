"""Compatibility detector API (legacy).

Historically, this project documented a ``pyimgano.detectors`` module with
classes such as ``IsolationForestDetector`` and ``AutoencoderDetector``.

The recommended, stable API is now the registry-driven factory in
``pyimgano.models``:

    - ``pyimgano.models.list_models()``
    - ``pyimgano.models.create_model(name, **kwargs)``

This module exists to keep older code working with minimal changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


class IdentityFeatureExtractor:
    """Default feature extractor for legacy detectors operating on feature vectors."""

    def extract(self, X):
        return np.asarray(X)


def _ensure_extractor(feature_extractor):
    return feature_extractor if feature_extractor is not None else IdentityFeatureExtractor()


# ---------------------------------------------------------------------------
# Classical / shallow wrappers (PyOD-backed)
# ---------------------------------------------------------------------------


from pyimgano.models.cof import VisionCOF as _VisionCOF
from pyimgano.models.copod import VisionCOPOD as _VisionCOPOD
from pyimgano.models.ecod import VisionECOD as _VisionECOD
from pyimgano.models.feature_bagging import VisionFeatureBagging as _VisionFeatureBagging
from pyimgano.models.gmm import VisionGMM as _VisionGMM
from pyimgano.models.hbos import VisionHBOS as _VisionHBOS
from pyimgano.models.iforest import VisionIForest as _VisionIForest
from pyimgano.models.kde import VisionKDE as _VisionKDE
from pyimgano.models.knn import VisionKNN as _VisionKNN
from pyimgano.models.kpca import VisionKPCA as _VisionKPCA
from pyimgano.models.loci import VisionLOCI as _VisionLOCI
from pyimgano.models.mad import VisionMAD as _VisionMAD
from pyimgano.models.ocsvm import VisionOCSVM as _VisionOCSVM
from pyimgano.models.pca import VisionPCA as _VisionPCA


class HistogramBasedDetector(_VisionHBOS):
    """Legacy name mapped to HBOS (histogram-based outlier score)."""

    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class MADDetector(_VisionMAD):
    def __init__(
        self,
        *,
        feature_extractor=None,
        contamination: float = 0.1,
        aggregation: str = "max",
        eps: float = 1e-12,
        consistency_correction: bool = True,
    ) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            aggregation=aggregation,
            eps=eps,
            consistency_correction=consistency_correction,
        )


class KNNDetector(_VisionKNN):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class COFDetector(_VisionCOF):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class LOCIDetector(_VisionLOCI):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class ECODDetector(_VisionECOD):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class COPODDetector(_VisionCOPOD):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class GMMDetector(_VisionGMM):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class KDEDetector(_VisionKDE):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class IsolationForestDetector(_VisionIForest):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class PCADetector(_VisionPCA):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class KernelPCADetector(_VisionKPCA):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class OCSVMDetector(_VisionOCSVM):
    def __init__(self, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


class FeatureBaggingDetector(_VisionFeatureBagging):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Deep (feature-based) wrappers
# ---------------------------------------------------------------------------


from pyimgano.models.auto_encoder import VisionAutoEncoder as _VisionAutoEncoder


def _normalize_ae_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    # Common legacy aliases
    if "epochs" in normalized and "epoch_num" not in normalized:
        normalized["epoch_num"] = normalized.pop("epochs")
    if "learning_rate" in normalized and "lr" not in normalized:
        normalized["lr"] = normalized.pop("learning_rate")

    # Legacy "hidden_dims" + "encoding_dim" -> symmetric hidden_neuron_list
    if "hidden_neuron_list" not in normalized:
        hidden_dims = normalized.pop("hidden_dims", None)
        encoding_dim = normalized.pop("encoding_dim", None)
        if hidden_dims is not None and encoding_dim is not None:
            hidden_dims_list = list(hidden_dims)
            normalized["hidden_neuron_list"] = (
                hidden_dims_list + [int(encoding_dim)] + list(reversed(hidden_dims_list))
            )
    # Legacy input_dim is inferred from features; keep as noop for compatibility.
    normalized.pop("input_dim", None)
    return normalized


class AutoencoderDetector(_VisionAutoEncoder):
    def __init__(self, *, feature_extractor=None, contamination: float = 0.1, **kwargs) -> None:
        super().__init__(
            feature_extractor=_ensure_extractor(feature_extractor),
            contamination=contamination,
            **_normalize_ae_kwargs(kwargs),
        )


__all__ = [
    "IdentityFeatureExtractor",
    # Classic/statistical
    "HistogramBasedDetector",
    "MADDetector",
    "KNNDetector",
    "COFDetector",
    "LOCIDetector",
    "ECODDetector",
    "COPODDetector",
    "GMMDetector",
    "KDEDetector",
    "IsolationForestDetector",
    "PCADetector",
    "KernelPCADetector",
    "OCSVMDetector",
    "FeatureBaggingDetector",
    # Deep (feature-based)
    "AutoencoderDetector",
]

