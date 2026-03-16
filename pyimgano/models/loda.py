# -*- coding: utf-8 -*-
"""LODA detectors for feature and vision inputs.

This module provides a small native implementation of LODA
(Lightweight On-line Detector of Anomalies) based on sparse random
projections and one-dimensional histograms.
"""

from __future__ import annotations

import logging
import numbers
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from tqdm import tqdm

from ..utils.fitted import require_fitted
from .base_detector import BaseDetector
from .baseml import BaseVisionDetector
from .registry import register_model

logger = logging.getLogger(__name__)

_MATPLOTLIB_PYPLOT_MODULE = "matplotlib.pyplot"
_VISION_LODA_VISUALIZATION_PURPOSE = "VisionLODA visualization"


def get_optimal_n_bins(data, max_bins: int = 50) -> int:
    """Estimate a reasonable histogram bin count for one projection."""

    values = np.asarray(data, dtype=np.float64).reshape(-1)
    n = int(values.shape[0])
    if n < 10:
        return min(5, max(1, n))

    sturges_bins = int(np.ceil(np.log2(n) + 1))

    std_dev = float(np.std(values))
    if std_dev > 0.0:
        bin_width = 3.5 * std_dev / (n ** (1 / 3))
        data_range = float(np.max(values) - np.min(values))
        scott_bins = int(np.ceil(data_range / bin_width)) if bin_width > 0.0 else sturges_bins
    else:
        scott_bins = sturges_bins

    optimal_bins = int((sturges_bins + scott_bins) / 2)
    return max(5, min(optimal_bins, int(max_bins)))


class LODAFeatureExtractor:
    """Extract lightweight image features for the vision wrapper."""

    def __init__(self, method: str = "histogram", normalize: bool = True) -> None:
        self.method = str(method)
        self.normalize = bool(normalize)
        self.scaler = StandardScaler() if self.normalize else None
        self.is_fitted = False

    def extract(self, x):
        """Return feature vectors for image paths or pass through arrays."""

        if isinstance(x, np.ndarray):
            features = np.asarray(x, dtype=np.float64)
            if self.normalize and self.is_fitted and self.scaler is not None:
                return self.scaler.transform(features)
            return features

        paths = list(x)
        logger.info("LODA: extracting image features for %d inputs", len(paths))
        features = []
        for item in tqdm(paths):
            try:
                features.append(self._extract_single_image(item))
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning("LODA: failed to extract features for %s: %s", item, exc)
                if features:
                    features.append(np.zeros_like(features[0]))
                else:
                    features.append(np.zeros(100, dtype=np.float64))

        features_np = np.asarray(features, dtype=np.float64)
        if not self.normalize or self.scaler is None:
            return features_np

        if not self.is_fitted:
            features_np = self.scaler.fit_transform(features_np)
            self.is_fitted = True
        else:
            features_np = self.scaler.transform(features_np)
        return features_np

    def _extract_single_image(self, image_path) -> np.ndarray:
        path = Path(image_path)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")

        features: list[float] = []

        if self.method in {"histogram", "combined"}:
            for idx in range(3):
                hist, _ = np.histogram(image[:, :, idx], bins=32, range=(0, 256))
                hist = hist.astype(np.float64)
                hist /= hist.sum() + 1e-6
                features.extend(hist.tolist())

        if self.method in {"statistical", "combined"}:
            for idx in range(3):
                channel = image[:, :, idx].reshape(-1).astype(np.float64)
                features.extend(
                    [
                        float(channel.mean() / 255.0),
                        float(channel.std() / 255.0),
                        float(np.percentile(channel, 25) / 255.0),
                        float(np.percentile(channel, 50) / 255.0),
                        float(np.percentile(channel, 75) / 255.0),
                        float(channel.min() / 255.0),
                        float(channel.max() / 255.0),
                    ]
                )

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            features.extend([float(grad_mag.mean() / 255.0), float(grad_mag.std() / 255.0)])

        return np.asarray(features, dtype=np.float64)


def _make_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is None:
        return np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    return np.random.default_rng(int(random_state))


@register_model(
    "core_loda",
    tags=("classical", "core", "features", "projection", "density"),
    metadata={
        "description": "Native LODA detector on feature matrices",
        "paper": "Pevny, Machine Learning 2016",
        "year": 2016,
    },
)
class CoreLODA(BaseDetector):
    """LODA on tabular or embedding features."""

    def __init__(
        self,
        contamination: float = 0.1,
        n_bins=10,
        n_random_cuts: int = 100,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(contamination=contamination)
        self.n_bins = n_bins
        self.n_random_cuts = int(n_random_cuts)
        self.random_state = random_state
        self.weights = np.ones(self.n_random_cuts, dtype=np.float64) / float(self.n_random_cuts)

        self.projections_ = None
        self.histograms_ = None
        self.limits_ = None
        self.n_bins_ = None
        self.decision_scores_ = None

    def fit(self, x, y=None):
        x_arr = check_array(x, dtype=np.float64)
        self._set_n_classes(y)

        n_samples, n_components = x_arr.shape
        rng = _make_rng(self.random_state)
        pred_scores = np.zeros((n_samples,), dtype=np.float64)

        n_nonzero_components = max(1, int(np.sqrt(n_components)))
        n_zero_components = max(0, n_components - n_nonzero_components)
        self.projections_ = rng.standard_normal((self.n_random_cuts, n_components))

        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
            self.histograms_ = []
            self.limits_ = []
            self.n_bins_ = []

            for idx in range(self.n_random_cuts):
                zero_idx = rng.permutation(n_components)[:n_zero_components]
                self.projections_[idx, zero_idx] = 0.0

                projected = self.projections_[idx].dot(x_arr.T)
                n_bins = get_optimal_n_bins(projected)
                histogram, limits = np.histogram(projected, bins=n_bins, density=False)
                histogram = histogram.astype(np.float64)
                histogram += 1e-12
                histogram /= histogram.sum()

                self.histograms_.append(histogram)
                self.limits_.append(limits)
                self.n_bins_.append(int(n_bins))

                indices = np.searchsorted(limits, projected, side="right") - 1
                indices = np.clip(indices, 0, n_bins - 1)
                pred_scores += -self.weights[idx] * np.log(histogram[indices])
        elif isinstance(self.n_bins, numbers.Integral):
            n_bins = int(self.n_bins)
            self.histograms_ = np.zeros((self.n_random_cuts, n_bins), dtype=np.float64)
            self.limits_ = np.zeros((self.n_random_cuts, n_bins + 1), dtype=np.float64)
            self.n_bins_ = None

            for idx in range(self.n_random_cuts):
                zero_idx = rng.permutation(n_components)[:n_zero_components]
                self.projections_[idx, zero_idx] = 0.0

                projected = self.projections_[idx].dot(x_arr.T)
                histogram, limits = np.histogram(projected, bins=n_bins, density=False)
                histogram = histogram.astype(np.float64)
                histogram += 1e-12
                histogram /= histogram.sum()

                self.histograms_[idx] = histogram
                self.limits_[idx] = limits

                indices = np.searchsorted(limits, projected, side="right") - 1
                indices = np.clip(indices, 0, n_bins - 1)
                pred_scores += -self.weights[idx] * np.log(histogram[indices])
        else:
            raise ValueError(f"n_bins must be an integer or 'auto', got {self.n_bins!r}")

        self.decision_scores_ = pred_scores / float(self.n_random_cuts)
        self._process_decision_scores()
        return self

    def decision_function(self, x):
        require_fitted(self, ["projections_", "decision_scores_"])
        x_arr = check_array(x, dtype=np.float64)
        pred_scores = np.zeros((x_arr.shape[0],), dtype=np.float64)

        if isinstance(self.n_bins, str) and self.n_bins.lower() == "auto":
            for idx in range(self.n_random_cuts):
                projected = self.projections_[idx].dot(x_arr.T)
                histogram = self.histograms_[idx]
                limits = self.limits_[idx]
                indices = np.searchsorted(limits, projected, side="right") - 1
                indices = np.clip(indices, 0, histogram.size - 1)
                pred_scores += -self.weights[idx] * np.log(histogram[indices])
        elif isinstance(self.n_bins, numbers.Integral):
            n_bins = int(self.n_bins)
            for idx in range(self.n_random_cuts):
                projected = self.projections_[idx].dot(x_arr.T)
                limits = self.limits_[idx]
                histogram = self.histograms_[idx]
                indices = np.searchsorted(limits, projected, side="right") - 1
                indices = np.clip(indices, 0, n_bins - 1)
                pred_scores += -self.weights[idx] * np.log(histogram[indices])
        else:
            raise ValueError(f"n_bins must be an integer or 'auto', got {self.n_bins!r}")

        return pred_scores / float(self.n_random_cuts)


@register_model(
    "vision_loda",
    tags=("vision", "classical"),
    metadata={
        "description": "Vision wrapper around native LODA",
        "paper": "Pevny, Machine Learning 2016",
        "year": 2016,
    },
)
class VisionLODA(BaseVisionDetector):
    """Vision-friendly LODA wrapper."""

    def __init__(
        self,
        contamination: float = 0.1,
        feature_extractor=None,
        n_bins=10,
        n_random_cuts: int = 100,
        feature_method: str = "histogram",
        normalize_features: bool = True,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.n_random_cuts = int(n_random_cuts)
        self.random_state = random_state

        if feature_extractor is None:
            feature_extractor = LODAFeatureExtractor(
                method=feature_method,
                normalize=normalize_features,
            )
            logger.info(
                "LODA: using default feature extractor (method=%s, normalize=%s)",
                feature_method,
                bool(normalize_features),
            )

        super().__init__(contamination=contamination, feature_extractor=feature_extractor)

    def _build_detector(self):
        return CoreLODA(
            contamination=self.contamination,
            n_bins=self.n_bins,
            n_random_cuts=self.n_random_cuts,
            random_state=self.random_state,
        )

    def visualize_projections(self, n_projections: int = 5) -> None:
        from pyimgano.utils.optional_deps import require

        plt = require(
            _MATPLOTLIB_PYPLOT_MODULE,
            extra="viz",
            purpose=_VISION_LODA_VISUALIZATION_PURPOSE,
        )
        if not hasattr(self.detector, "projections_"):
            logger.warning("LODA: visualize_projections called before fit()")
            return

        n_show = min(int(n_projections), self.n_random_cuts)
        _, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
        if n_show == 1:
            axes = [axes]

        for idx in range(n_show):
            projection = self.detector.projections_[idx]
            axes[idx].bar(range(len(projection)), np.abs(projection))
            axes[idx].set_title(f"Projection {idx + 1}")
            axes[idx].set_xlabel("Feature index")
            axes[idx].set_ylabel("Absolute weight")

        plt.tight_layout()
        plt.show()

    def visualize_histograms(self, n_histograms: int = 5) -> None:
        from pyimgano.utils.optional_deps import require

        plt = require(
            _MATPLOTLIB_PYPLOT_MODULE,
            extra="viz",
            purpose=_VISION_LODA_VISUALIZATION_PURPOSE,
        )
        if not hasattr(self.detector, "histograms_"):
            logger.warning("LODA: visualize_histograms called before fit()")
            return

        n_show = min(int(n_histograms), self.n_random_cuts)
        _, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
        if n_show == 1:
            axes = [axes]

        for idx in range(n_show):
            histogram = self.detector.histograms_[idx]
            axes[idx].bar(range(len(histogram)), histogram)
            axes[idx].set_title(f"Histogram {idx + 1}")
            axes[idx].set_xlabel("Bin index")
            axes[idx].set_ylabel("Probability")

        plt.tight_layout()
        plt.show()

    def visualize_scores(self) -> None:
        from pyimgano.utils.optional_deps import require

        plt = require(
            _MATPLOTLIB_PYPLOT_MODULE,
            extra="viz",
            purpose=_VISION_LODA_VISUALIZATION_PURPOSE,
        )
        if not hasattr(self.detector, "decision_scores_"):
            logger.warning("LODA: visualize_scores called before fit()")
            return

        scores = np.asarray(self.detector.decision_scores_, dtype=np.float64)
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=30, edgecolor="black")
        plt.axvline(self.detector.threshold_, color="r", linestyle="--")
        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.title("Score distribution")

        plt.subplot(1, 2, 2)
        sorted_scores = np.sort(scores)
        plt.scatter(range(len(sorted_scores)), sorted_scores, s=10, alpha=0.5)
        plt.axhline(self.detector.threshold_, color="r", linestyle="--")
        plt.xlabel("Sorted sample index")
        plt.ylabel("Anomaly score")
        plt.title("Sorted scores")

        plt.tight_layout()
        plt.show()

    def get_info(self) -> dict[str, object]:
        if not hasattr(self.detector, "projections_"):
            return {"status": "unfitted"}

        info: dict[str, object] = {
            "algorithm": "Vision-LODA",
            "bin_strategy": self.n_bins,
            "n_random_cuts": self.n_random_cuts,
            "feature_dim": int(self.detector.projections_.shape[1]),
            "contamination": float(self.contamination),
            "threshold": float(self.detector.threshold_),
        }
        if getattr(self.detector, "n_bins_", None):
            info["bin_range"] = (
                int(min(self.detector.n_bins_)),
                int(max(self.detector.n_bins_)),
            )
        if hasattr(self.detector, "labels_"):
            labels = np.asarray(self.detector.labels_, dtype=np.int32)
            info["n_anomalies"] = int(labels.sum())
            info["n_samples"] = int(labels.shape[0])
        return info


if __name__ == "__main__":  # pragma: no cover
    logger.info("VisionLODA demo module")
