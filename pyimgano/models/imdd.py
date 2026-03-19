"""IMDD (Linear Model Deviation-based Detector) vision integration."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import stats
from sklearn.utils import check_array, check_random_state

from pyimgano.utils.optional_deps import require

from ..utils.param_check import check_parameter
from .baseml import BaseVisionDetector
from .core_feature_base import CoreFeatureDetector
from .registry import register_model

numba = require("numba", extra="numba", purpose="IMDD/LMDD detector acceleration")
njit = numba.njit


@njit
def _aad(x):
    return np.mean(np.abs(x - np.mean(x)))


def _resolve_distance_measure(dis_measure: str):
    if dis_measure == "aad":
        return _aad
    if dis_measure == "var":
        return np.var
    if dis_measure == "iqr":
        return stats.iqr
    raise ValueError("Unknown dissimilarity measure type, choose from {'aad','var','iqr'}")


class CoreIMDD:
    """Pure NumPy implementation of LMDD/IMDD detector."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_iter: int = 50,
        dis_measure: str = "aad",
        random_state=None,
    ) -> None:
        self.contamination = contamination
        check_parameter(n_iter, low=1, param_name="n_iter")
        self.n_iter = n_iter
        self.random_state = check_random_state(random_state)
        self.dis_measure_name = dis_measure
        self.dis_measure = _resolve_distance_measure(dis_measure)
        self.decision_scores_ = None

    # ------------------------------------------------------------------
    def fit(self, x, _y=None):
        del _y
        x = check_array(x)
        self.decision_scores_ = self._compute_scores(x)
        return self

    def decision_function(self, x):
        x = check_array(x)
        return self._compute_scores(x)

    # ------------------------------------------------------------------
    def _compute_scores(self, x):
        dis = np.zeros(x.shape[0])
        card = np.zeros(x.shape[0])

        # base order computation
        base_scores = self._smoothing_factor(x)
        self._update_cardinality(base_scores, card, x.shape[0])
        dis = np.maximum(dis, base_scores)

        indices = np.arange(x.shape[0])
        rs = check_random_state(self.random_state.randint(0, 2**31 - 1))
        for _ in range(self.n_iter):
            rs.shuffle(indices)
            shuffled = x[indices]
            scores = self._smoothing_factor(shuffled)[np.argsort(indices)]
            current_card = x.shape[0] - np.count_nonzero(scores)
            better = scores > dis
            dis[better] = scores[better]
            card[better] = current_card
        return dis * card

    def _update_cardinality(self, scores, card, n_samples):
        card[:] = np.where(scores > 0, n_samples - np.count_nonzero(scores), card)

    def _smoothing_factor(self, x):
        res = np.zeros(x.shape[0])
        best_delta = -np.inf
        best_idx = 0
        for i in range(1, x.shape[0]):
            delta = self.dis_measure(x[: i + 1]) - self.dis_measure(x[:i])
            if delta > best_delta:
                best_delta = delta
                best_idx = i

        if best_delta <= 0:
            return res
        res[best_idx] = best_delta
        prefix = x[: best_idx + 1]
        prefix_prev = x[:best_idx]
        for k in range(best_idx + 1, x.shape[0]):
            diff = self.dis_measure(np.vstack((prefix_prev, x[k]))) - self.dis_measure(
                np.vstack((prefix, x[k]))
            )
            if diff >= 0:
                res[k] = diff + best_delta
        return res


@register_model(
    "core_imdd",
    tags=("classical", "core", "features", "imdd", "lmdd"),
    metadata={
        "description": "IMDD/LMDD deviation detector for feature matrices (native wrapper)",
        "type": "deviation",
    },
)
class CoreIMDDDetector(CoreFeatureDetector):
    """Feature-matrix IMDD/LMDD detector (`core_*`)."""

    def __init__(
        self,
        *,
        contamination: float = 0.1,
        n_iter: int = 50,
        dis_measure: str = "aad",
        random_state=None,
    ) -> None:
        self.n_iter = int(n_iter)
        self.dis_measure = str(dis_measure)
        self.random_state = random_state
        super().__init__(contamination=contamination)

    def _build_detector(self):  # noqa: ANN201
        return CoreIMDD(
            contamination=float(self.contamination),
            n_iter=int(self.n_iter),
            dis_measure=str(self.dis_measure),
            random_state=self.random_state,
        )


@register_model(
    "vision_imdd",
    tags=("vision", "classical"),
    metadata={"description": "Vision wrapper for IMDD deviation detector"},
)
class VisionIMDD(BaseVisionDetector):
    def __init__(self, *, feature_extractor, contamination: float = 0.1, **kwargs):
        self.detector_kwargs = dict(contamination=contamination, **kwargs)
        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreIMDD(**self.detector_kwargs)

    def fit(self, x: Iterable[str], y=None):
        del y
        features = np.asarray(self.feature_extractor.extract(x))
        self.detector.fit(features)
        self.decision_scores_ = self.detector.decision_scores_
        self._process_decision_scores()
        return self

    def decision_function(self, x):
        features = np.asarray(self.feature_extractor.extract(x))
        return self.detector.decision_function(features)
