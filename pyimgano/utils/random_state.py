"""Random-state helpers (NumPy only).

We keep a tiny subset of sklearn's `check_random_state` behavior to avoid
pulling sklearn utilities into places where we want to stay NumPy-only.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar, cast

import numpy as np

_T = TypeVar("_T")


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


@contextmanager
def isolated_random_state(seed: int | None) -> Iterator[None]:
    """Temporarily seed legacy Python/NumPy/Torch RNGs without leaking state.

    New code should still prefer local ``numpy.random.Generator`` and
    ``torch.Generator`` objects. This context exists for legacy training code
    whose parameter initialization or data loading consumes global RNGs.
    """

    if seed is None:
        yield
        return

    seed_value = int(seed)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]

        random.seed(seed_value)
        np.random.seed(seed_value)
        if torch is None:
            yield
            return

        devices: list[int] = []
        if torch.cuda.is_available():
            devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed_value)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def isolated_random_state_method(method: Callable[..., _T]) -> Callable[..., _T]:
    """Wrap an instance method using ``self.random_state`` as an isolated seed."""

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> _T:
        seed = getattr(self, "random_state", None)
        with isolated_random_state(None if seed is None else int(seed)):
            return method(self, *args, **kwargs)

    return cast(Callable[..., _T], wrapped)
