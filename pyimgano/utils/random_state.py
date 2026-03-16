"""Random-state helpers (NumPy only).

We keep a tiny subset of sklearn's `check_random_state` behavior to avoid
pulling sklearn utilities into places where we want to stay NumPy-only.
"""

from __future__ import annotations

import os

import numpy as np


def check_random_state(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Turn seed into a `np.random.Generator` instance.

    Parameters
    ----------
    seed:
        - `None`: return a new Generator with an unpredictable seed.
        - `int`: return a new Generator seeded with that integer.
        - `Generator`: return it unchanged.
    """

    if seed is None:
        return np.random.default_rng(int.from_bytes(os.urandom(8), "little"))
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, (int, np.integer)):
        return np.random.default_rng(int(seed))
    raise TypeError(f"Invalid random_state type: {type(seed).__name__}")
