from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from .registry import register_model


def _rank_normalize(scores: NDArray) -> NDArray:
    scores_np = np.asarray(scores, dtype=np.float64)
    if scores_np.ndim != 1:
        raise ValueError(f"Expected 1D scores, got shape {scores_np.shape}")

    n = int(scores_np.shape[0])
    if n == 0:
        return np.asarray(scores_np, dtype=np.float32)
    if n == 1:
        return np.zeros((1,), dtype=np.float32)

    order = np.argsort(scores_np, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    return (ranks.astype(np.float32)) / float(n - 1)


@dataclass(frozen=True)
class EnsembleConfig:
    combine: str = "mean_rank"
    weights: Optional[Sequence[float]] = None
    trim_fraction: float = 0.1


@register_model(
    "vision_score_ensemble",
    tags=("vision", "ensemble", "score"),
    metadata={
        "description": "Score-only ensemble wrapper (mean of rank-normalized scores by default)",
    },
)
class VisionScoreEnsemble:
    """Combine scores from multiple detectors into a single image-level score.

    Notes
    -----
    - Default combine strategy is `mean_rank`: rank-normalize each detector's
      scores to [0,1] and average. This is robust when detectors emit scores on
      different scales.
    - `fit()` calls `fit()` on each sub-detector if available, then calibrates a
      quantile threshold for `predict()`.
    """

    def __init__(
        self,
        detectors: Sequence[Any],
        *,
        contamination: float = 0.1,
        combine: str = "mean_rank",
        weights: Optional[Sequence[float]] = None,
        trim_fraction: float = 0.1,
    ) -> None:
        if not detectors:
            raise ValueError("detectors must be non-empty")

        # Accept detector instances OR specs (name strings / {"name":..,"kwargs":..}).
        self.detectors = list(detectors)
        self.detectors_ = None  # resolved + fitted detectors
        self.contamination = float(contamination)
        if not (0.0 < self.contamination < 0.5):
            raise ValueError(f"contamination must be in (0,0.5), got {self.contamination}")

        tf = float(trim_fraction)
        if not (0.0 <= tf < 0.5):
            raise ValueError(f"trim_fraction must be in [0,0.5), got {trim_fraction}")

        self.config = EnsembleConfig(combine=str(combine), weights=weights, trim_fraction=tf)
        self.decision_scores_: Optional[NDArray] = None
        self.threshold_: Optional[float] = None

    def fit(self, X: Iterable[Any], _y=None):
        items = list(X)
        if not items:
            raise ValueError("X must be non-empty")

        from pyimgano.models.ensemble_spec import resolve_model_specs

        # Allow using decision_function() before fit(): in that case `detectors_`
        # may already be resolved. Reuse them to avoid losing state.
        if self.detectors_ is None:
            self.detectors_ = resolve_model_specs(
                self.detectors, default_contamination=float(self.contamination)
            )
        detectors = self.detectors_

        for det in detectors:
            fit = getattr(det, "fit", None)
            if callable(fit):
                fit(items)

        self.decision_scores_ = np.asarray(self.decision_function(items), dtype=np.float64)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def _combined_scores(self, per_detector_scores: Sequence[NDArray]) -> NDArray:
        if not per_detector_scores:
            raise ValueError("No detector scores provided")

        stacked = np.stack([np.asarray(s, dtype=np.float64) for s in per_detector_scores], axis=0)
        if stacked.ndim != 2:
            raise RuntimeError(f"Expected stacked scores shape (M,N), got {stacked.shape}")

        combine = self.config.combine
        weights = self.config.weights
        trim_fraction = float(self.config.trim_fraction)

        if weights is not None:
            if len(weights) != stacked.shape[0]:
                raise ValueError("weights length must match number of detectors")
            w = np.asarray(weights, dtype=np.float64)
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            if float(np.sum(w)) <= 0:
                raise ValueError("weights must sum to > 0")
            w = w / float(np.sum(w))
        else:
            w = None

        def _trimmed_mean(values_2d: NDArray) -> NDArray:
            m = int(values_2d.shape[0])
            if m <= 1:
                return values_2d.reshape(-1)
            k = int(np.floor(trim_fraction * m))
            if 2 * k >= m:
                raise ValueError("trim_fraction too large for number of detectors")
            sorted_vals = np.sort(values_2d, axis=0)
            core = sorted_vals[k : m - k, :] if k > 0 else sorted_vals
            return np.mean(core, axis=0)

        if combine == "mean_rank":
            norm = np.stack([_rank_normalize(stacked[i]) for i in range(stacked.shape[0])], axis=0)
            if w is None:
                return np.mean(norm, axis=0).astype(np.float32, copy=False)
            return np.sum(norm * w[:, None], axis=0).astype(np.float32, copy=False)

        if combine == "max_rank":
            norm = np.stack([_rank_normalize(stacked[i]) for i in range(stacked.shape[0])], axis=0)
            if w is not None:
                raise ValueError("weights are not supported for combine='max_rank'")
            return np.max(norm, axis=0).astype(np.float32, copy=False)

        if combine == "trimmed_mean_rank":
            norm = np.stack([_rank_normalize(stacked[i]) for i in range(stacked.shape[0])], axis=0)
            if w is not None:
                raise ValueError("weights are not supported for combine='trimmed_mean_rank'")
            return _trimmed_mean(norm).astype(np.float32, copy=False)

        if combine == "mean":
            if w is None:
                return np.mean(stacked, axis=0).astype(np.float32, copy=False)
            return np.sum(stacked * w[:, None], axis=0).astype(np.float32, copy=False)

        if combine == "max":
            if w is not None:
                raise ValueError("weights are not supported for combine='max'")
            return np.max(stacked, axis=0).astype(np.float32, copy=False)

        if combine == "trimmed_mean":
            if w is not None:
                raise ValueError("weights are not supported for combine='trimmed_mean'")
            return _trimmed_mean(stacked).astype(np.float32, copy=False)

        raise ValueError(f"Unknown combine strategy: {combine!r}")

    def decision_function(self, X: Iterable[Any]) -> NDArray:
        items = list(X)
        if not items:
            return np.zeros((0,), dtype=np.float32)

        if self.detectors_ is None:
            # Score-only ensembles should be usable without an explicit fit()
            # when detectors are already instantiated.
            from pyimgano.models.ensemble_spec import resolve_model_specs

            self.detectors_ = resolve_model_specs(
                self.detectors, default_contamination=float(self.contamination)
            )

        per_detector: list[NDArray] = []
        for det in self.detectors_:
            scores = det.decision_function(items)
            scores_np = np.asarray(scores, dtype=np.float64)
            if scores_np.shape[0] != len(items):
                raise ValueError("Sub-detector returned unexpected score length")
            per_detector.append(scores_np)

        return self._combined_scores(per_detector)

    def predict(self, X: Iterable[Any]) -> NDArray:
        if self.threshold_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        scores = np.asarray(self.decision_function(X), dtype=np.float64)
        return (scores >= float(self.threshold_)).astype(np.int64)


@register_model(
    "core_score_ensemble",
    tags=("classical", "core", "features", "ensemble", "score"),
    metadata={
        "description": "Score-only ensemble wrapper for feature-matrix detectors (spec-friendly)",
    },
)
class CoreScoreEnsemble(VisionScoreEnsemble):
    """Alias of :class:`VisionScoreEnsemble` for feature-matrix workflows (`core_*`)."""

    def __init__(
        self,
        detectors: Sequence[Any] | None = None,
        *,
        contamination: float = 0.1,
        combine: str = "mean_rank",
        weights: Optional[Sequence[float]] = None,
        trim_fraction: float = 0.1,
    ) -> None:
        # Keep core smoke tests cheap: provide a lightweight default ensemble.
        if detectors is None:
            detectors = ["core_ecod", "core_knn", "core_copod"]
        super().__init__(
            detectors=detectors,
            contamination=float(contamination),
            combine=str(combine),
            weights=weights,
            trim_fraction=float(trim_fraction),
        )
