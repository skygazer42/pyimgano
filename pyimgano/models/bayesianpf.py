from __future__ import annotations

"""Adapter for the official Bayes-PFL implementation.

Bayes-PFL depends on its custom CLIP stack, RCA module, auxiliary-data
training, and released checkpoints.  This module deliberately does not invent
a local fallback: callers must inject a trained backend that implements the
official method.
"""

from pathlib import Path
from typing import Any, Iterable, Optional, cast

import numpy as np
from numpy.typing import NDArray

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .registry import register_model

MODEL_NOT_FITTED_ERROR = "Model not calibrated. Call fit() first."
BACKEND_REQUIRED_ERROR = (
    "vision_bayesianpf requires a trained official Bayes-PFL backend; "
    "the paper method cannot run from randomly initialized local substitutes."
)


def _as_items(value: object) -> list[Any]:
    if isinstance(value, (str, Path)):
        return [value]
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return [value]
        return list(value)
    return list(cast(Iterable[Any], value))


def _as_scores(value: Any, *, expected: int) -> NDArray:
    scores = np.asarray(value, dtype=np.float64).reshape(-1)
    if scores.shape != (expected,):
        raise ValueError(
            f"Bayes-PFL backend returned {scores.shape}; expected one score for each "
            f"of {expected} inputs."
        )
    if not np.isfinite(scores).all():
        raise ValueError("Bayes-PFL backend returned non-finite scores.")
    return scores


@register_model(
    "vision_bayesianpf",
    tags=("vision", "deep", "bayesianpf", "zero-shot", "external-backend"),
    metadata={
        "description": "Adapter for an official, checkpoint-backed Bayes-PFL runtime",
        "paper": "Bayesian Prompt Flow Learning for Zero-Shot Anomaly Detection",
        "paper_url": "https://openaccess.thecvf.com/content/CVPR2025/html/Qu_Bayesian_Prompt_Flow_Learning_for_Zero-Shot_Anomaly_Detection_CVPR_2025_paper.html",
        "year": 2025,
        "conference": "CVPR",
        "supervision": "zero-shot",
        "training_regime": "auxiliary-trained zero-shot transfer",
        "implementation_status": "external-backend-adapter",
        "paper_fidelity": "external-backend",
        "backend": "official-bayes-pfl",
        "requires_checkpoint": True,
        "weights_source": "official-bayes-pfl-checkpoint-and-clip-weights",
    },
)
class VisionBayesianPF:
    """Expose a trained official Bayes-PFL backend through the detector API.

    The backend must provide ``decision_function(items)`` or be callable.  It
    may additionally implement ``fit``, ``predict_with_uncertainty``,
    ``predict_anomaly_map``, ``save_checkpoint``, and ``load_checkpoint``.
    """

    def __init__(
        self,
        *,
        backend: Any = None,
        checkpoint_path: str | Path | None = None,
        contamination: float = 0.1,
        batch_size: int = 32,
        random_state: Optional[int] = None,
        **kwargs: object,
    ) -> None:
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"Unexpected Bayes-PFL adapter argument: {unexpected!r}")
        self.backend = backend
        self.contamination = float(contamination)
        if not 0.0 < self.contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.random_state = random_state
        self.decision_scores_: NDArray | None = None
        self.threshold_: float | None = None
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _require_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError(BACKEND_REQUIRED_ERROR)
        return self.backend

    def _backend_scores(self, items: list[Any], *, batch_size: int) -> NDArray:
        backend = self._require_backend()
        scorer = getattr(backend, "decision_function", None)
        if scorer is None:
            scorer = backend if callable(backend) else None
        if scorer is None:
            raise TypeError("Bayes-PFL backend must be callable or implement decision_function().")

        try:
            output = scorer(items, batch_size=batch_size)
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            output = scorer(items)
        return _as_scores(output, expected=len(items))

    def fit(
        self,
        x: object = MISSING,
        y: Optional[NDArray] = None,
        **kwargs: object,
    ) -> "VisionBayesianPF":
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="fit"))
        if not items:
            raise ValueError("X must contain at least one calibration sample.")
        backend = self._require_backend()
        fit_backend = getattr(backend, "fit", None)
        if callable(fit_backend):
            fit_backend(items, y)
        self.decision_scores_ = self._backend_scores(items, batch_size=self.batch_size)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1.0 - self.contamination))
        return self

    def decision_function(
        self,
        x: object = MISSING,
        batch_size: Optional[int] = None,
        **kwargs: object,
    ) -> NDArray:
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="decision_function"))
        current_batch_size = self.batch_size if batch_size is None else int(batch_size)
        if current_batch_size <= 0:
            raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
        return self._backend_scores(items, batch_size=current_batch_size)

    def predict(
        self,
        x: object = MISSING,
        return_confidence: bool = False,
        **kwargs: object,
    ) -> NDArray:
        if return_confidence:
            raise NotImplementedError("return_confidence is not implemented for VisionBayesianPF")
        if self.threshold_ is None:
            raise RuntimeError(MODEL_NOT_FITTED_ERROR)
        items = _as_items(resolve_legacy_x_keyword(x, kwargs, method_name="predict"))
        scores = self._backend_scores(items, batch_size=self.batch_size)
        return (scores > self.threshold_).astype(np.int64)

    def predict_with_uncertainty(
        self,
        x: object = MISSING,
        **kwargs: object,
    ) -> tuple[NDArray, NDArray]:
        items = _as_items(
            resolve_legacy_x_keyword(x, kwargs, method_name="predict_with_uncertainty")
        )
        method = getattr(self._require_backend(), "predict_with_uncertainty", None)
        if not callable(method):
            raise NotImplementedError(
                "The injected Bayes-PFL backend does not expose posterior uncertainty."
            )
        scores, uncertainty = method(items)
        return (
            _as_scores(scores, expected=len(items)),
            _as_scores(uncertainty, expected=len(items)),
        )

    def predict_anomaly_map(self, x: object = MISSING, **kwargs: object) -> NDArray:
        items = _as_items(
            resolve_legacy_x_keyword(x, kwargs, method_name="predict_anomaly_map")
        )
        method = getattr(self._require_backend(), "predict_anomaly_map", None)
        if not callable(method):
            raise NotImplementedError(
                "The injected Bayes-PFL backend does not expose pixel anomaly maps."
            )
        maps = np.asarray(method(items), dtype=np.float32)
        if maps.ndim != 3 or maps.shape[0] != len(items):
            raise ValueError(
                "Bayes-PFL backend predict_anomaly_map() must return shape (N, H, W)."
            )
        if not np.isfinite(maps).all():
            raise ValueError("Bayes-PFL backend returned non-finite anomaly maps.")
        return maps

    def save_checkpoint(self, path: str | Path) -> Path:
        method = getattr(self._require_backend(), "save_checkpoint", None)
        if not callable(method):
            raise NotImplementedError("The injected Bayes-PFL backend cannot save checkpoints.")
        result = method(path)
        return Path(path if result is None else result)

    def load_checkpoint(self, path: str | Path) -> None:
        method = getattr(self._require_backend(), "load_checkpoint", None)
        if not callable(method):
            raise NotImplementedError("The injected Bayes-PFL backend cannot load checkpoints.")
        method(path)
