"""Small fitted-state helpers.

scikit-learn's `check_is_fitted` is convenient, but newer sklearn versions
may warn/error when called on non-BaseEstimator objects due to tag retrieval.
We keep a minimal helper for our own detectors.
"""

from __future__ import annotations

from typing import Iterable


def require_fitted(obj: object, attrs: Iterable[str]) -> None:
    missing = [name for name in attrs if not hasattr(obj, name)]
    if missing:
        missing_str = ", ".join(repr(m) for m in missing)
        raise RuntimeError(f"Estimator is not fitted yet. Missing: {missing_str}")

