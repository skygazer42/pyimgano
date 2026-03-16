from __future__ import annotations

from .models.base_deep import BaseDeepLearningDetector
from .models.base_detector import BaseDetector
from .models.baseCv import BaseVisionDeepDetector


class BaseVisionClassicalDetector(BaseDetector):
    """Compatibility base for legacy image-first classical detectors."""

    def __init__(self, contamination: float = 0.1) -> None:
        super().__init__(contamination=contamination)
        self._set_n_classes(None)
        self.is_fitted_ = False

    def _check_is_fitted(self) -> None:
        if not self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} must be fitted before prediction.")


__all__ = [
    "BaseDeepLearningDetector",
    "BaseDetector",
    "BaseVisionClassicalDetector",
    "BaseVisionDeepDetector",
]
