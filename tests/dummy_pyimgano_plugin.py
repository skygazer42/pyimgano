from __future__ import annotations

import numpy as np


def register() -> None:
    """Dummy plugin used by unit tests.

    This is intentionally lightweight: no optional heavy deps.
    """

    from pyimgano.models.registry import register_model

    @register_model(
        "plugin_dummy_model",
        tags=("plugin", "test"),
        metadata={"description": "Dummy model registered by tests via pyimgano plugins"},
    )
    class PluginDummyModel:
        def __init__(self, *, contamination: float = 0.1) -> None:
            self.contamination = float(contamination)
            self.decision_scores_ = None

        def fit(self, X, _y=None):  # noqa: ANN001, ANN201 - sklearn-like shim for tests
            items = list(X)
            self.decision_scores_ = np.zeros((len(items),), dtype=np.float64)
            return self

        def decision_function(self, X):  # noqa: ANN001, ANN201 - sklearn-like shim for tests
            items = list(X)
            return np.zeros((len(items),), dtype=np.float64)
