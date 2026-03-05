from __future__ import annotations

"""Model serialization helpers (joblib).

Many classical `pyimgano` models are Python objects that can be serialized with
joblib/pickle. Deep models often require special checkpoint handling; those are
out of scope for this helper.
"""

from pathlib import Path
from typing import Any


def save_model(model: Any, path: str | Path) -> Path:
    """Serialize a model via joblib."""

    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(p))
    return p


def load_model(path: str | Path) -> Any:
    """Load a model serialized via :func:`save_model`."""

    import joblib

    return joblib.load(str(path))
