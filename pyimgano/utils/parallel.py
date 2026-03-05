"""Small parallelism helpers (joblib).

We keep this module tiny and explicit to avoid scattering joblib usage patterns
across the codebase.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from joblib import Parallel, cpu_count, delayed

T = TypeVar("T")
U = TypeVar("U")


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Resolve an `n_jobs` value into a positive integer.

    Semantics (sklearn-like):
    - `None` -> 1
    - `-1` -> all CPUs
    - `< -1` -> `cpu_count() + 1 + n_jobs`
    """

    if n_jobs is None:
        return 1
    n = int(n_jobs)
    if n == 0:
        raise ValueError("n_jobs must be != 0")
    if n > 0:
        return n

    total = int(cpu_count())
    if total <= 0:
        return 1
    if n == -1:
        return max(1, total)

    # Example: total=8, n_jobs=-2 -> 7
    return max(1, total + 1 + n)


def parallel_map(
    fn: Callable[[T], U],
    items: Iterable[T],
    *,
    n_jobs: int | None = 1,
    backend: str = "threading",
) -> list[U]:
    """Map `fn` over `items`, optionally in parallel."""

    xs = list(items)
    n = resolve_n_jobs(n_jobs)
    if n <= 1 or len(xs) <= 1:
        return [fn(x) for x in xs]

    return Parallel(n_jobs=n, backend=str(backend))(delayed(fn)(x) for x in xs)
