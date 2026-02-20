from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from pyimgano.utils.optional_deps import require

from .registry import register_model


def _build_torch_inferencer(*, checkpoint_path: str, device: str):
    require("anomalib", extra="anomalib", purpose="anomalib backend detectors")

    # Support multiple anomalib versions with different import paths.
    try:
        from anomalib.deploy import TorchInferencer  # type: ignore
    except Exception:  # pragma: no cover
        from anomalib.deploy.inferencers import TorchInferencer  # type: ignore

    return TorchInferencer(path=checkpoint_path, device=device)


@dataclass
class _AnomalibPredictResult:
    score: float
    anomaly_map: Optional[NDArray] = None


def _extract_score_and_map(result) -> _AnomalibPredictResult:
    # Newer anomalib uses structured objects; older versions may return dict-like.
    if isinstance(result, dict):
        score = result.get("anomaly_score") or result.get("pred_score") or result.get("score")
        anomaly_map = result.get("anomaly_map") or result.get("segmentation_map")
        return _AnomalibPredictResult(score=float(score), anomaly_map=anomaly_map)

    score = getattr(result, "anomaly_score", None)
    if score is None:
        score = getattr(result, "pred_score", None)
    if score is None:
        score = getattr(result, "score", None)
    if score is None:
        raise ValueError("Unable to extract anomaly score from anomalib prediction result")

    anomaly_map = getattr(result, "anomaly_map", None)
    if anomaly_map is None:
        anomaly_map = getattr(result, "segmentation_map", None)

    return _AnomalibPredictResult(score=float(score), anomaly_map=anomaly_map)


@register_model(
    "vision_patchcore_anomalib",
    tags=("vision", "deep", "backend", "anomalib", "patchcore"),
    metadata={
        "description": "PatchCore via anomalib backend (requires pyimgano[anomalib])",
        "backend": "anomalib",
    },
)
class VisionPatchCoreAnomalib:
    """Inference wrapper for anomalib PatchCore checkpoints.

    This is intentionally **inference-first** to keep `pyimgano` core lightweight.
    Training is expected to be done via anomalib, producing a checkpoint that
    this wrapper can load.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._inferencer = _build_torch_inferencer(checkpoint_path=checkpoint_path, device=device)

    def fit(self, X: Iterable[str], y=None):
        # Training is managed by anomalib; keep API compatibility.
        return self

    def decision_function(self, X: Iterable[str]) -> NDArray:
        paths = list(X)
        scores = np.zeros(len(paths), dtype=np.float64)
        for i, path in enumerate(paths):
            pred = self._inferencer.predict(path)
            scores[i] = _extract_score_and_map(pred).score
        return scores

    def get_anomaly_map(self, image_path: str) -> NDArray:
        pred = self._inferencer.predict(image_path)
        extracted = _extract_score_and_map(pred)
        if extracted.anomaly_map is None:
            raise ValueError("anomalib prediction did not include an anomaly map")
        return np.asarray(extracted.anomaly_map)

    def predict_anomaly_map(self, X: Iterable[str]) -> NDArray:
        maps = [self.get_anomaly_map(path) for path in X]
        return np.stack(maps)

