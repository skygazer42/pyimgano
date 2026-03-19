from __future__ import annotations

from .models.base_detector import BaseDetector


class BaseVisionClassicalDetector(BaseDetector):
    """Lightweight base for classical image detectors operating on image arrays."""

    input_mode = "images"

    def __init__(self, contamination: float = 0.1) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.is_fitted_ = False

    def _check_is_fitted(self) -> None:
        if not bool(getattr(self, "is_fitted_", False)):
            raise RuntimeError("Model must be fitted before calling this method.")
