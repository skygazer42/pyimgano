from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch
    except Exception:
        return

    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        return


def micro_finetune(
    detector: Any,
    train_inputs: Sequence[Any],
    *,
    seed: int | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort micro-finetune runner for supported detectors.

    This runner intentionally keeps scope narrow: it sets seeds (best-effort),
    calls `fit(...)`, and returns a small JSON-friendly payload with timing.
    """

    total_start = time.perf_counter()
    if seed is not None:
        _seed_everything(int(seed))

    kwargs = dict(fit_kwargs or {})

    fit_start = time.perf_counter()
    fit_kwargs_used: dict[str, Any]
    try:
        detector.fit(train_inputs, **kwargs)
        fit_kwargs_used = kwargs
    except TypeError:
        detector.fit(train_inputs)
        fit_kwargs_used = {}
    fit_s = float(time.perf_counter() - fit_start)

    total_s = float(time.perf_counter() - total_start)
    return {
        "seed": int(seed) if seed is not None else None,
        "fit_kwargs_used": dict(fit_kwargs_used),
        "timing": {
            "fit_s": fit_s,
            "total_s": total_s,
        },
    }

