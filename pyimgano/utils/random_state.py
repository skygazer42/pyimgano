"""Random-state helpers (NumPy only).

We keep a tiny subset of sklearn's `check_random_state` behavior to avoid
pulling sklearn utilities into places where we want to stay NumPy-only.
"""

from __future__ import annotations

import numpy as np


def check_random_state(seed: int | np.random.RandomState | None) -> np.random.RandomState:
    """Turn seed into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed:
        - `None`: return a new RandomState with unpredictable seed.
        - `int`: return a new RandomState seeded with that integer.
        - `RandomState`: return it unchanged.
    """

    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(int(seed))
    raise TypeError(f"Invalid random_state type: {type(seed).__name__}")
