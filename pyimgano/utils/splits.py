from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def split_train_calibration(
    paths: Iterable[str],
    *,
    calibration_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[str], list[str]]:
    """Split an iterable of paths into (train, calibration) deterministically.

    Notes
    -----
    - Output lists preserve the original input order.
    - The split is deterministic given the same inputs + seed.
    """

    items = list(paths)
    n = int(len(items))
    if n == 0:
        return [], []
    if n < 2:
        return list(items), []

    frac = float(calibration_fraction)
    if frac <= 0.0:
        return list(items), []
    if frac >= 1.0:
        return [], list(items)

    n_cal = int(math.ceil(n * frac))
    n_cal = max(1, min(n - 1, n_cal))

    rng = np.random.default_rng(int(seed))
    cal_idx = set(rng.choice(n, size=n_cal, replace=False).tolist())

    train: list[str] = []
    cal: list[str] = []
    for i, p in enumerate(items):
        if i in cal_idx:
            cal.append(str(p))
        else:
            train.append(str(p))

    return train, cal
