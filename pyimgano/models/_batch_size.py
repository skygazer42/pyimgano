"""Internal helpers for batch-size compatibility in deep detectors.

Some detectors implement `decision_function(X, batch_size=...)` only to keep
interface compatibility with `BaseDeepLearningDetector`, even though their
scoring is done via a custom `predict()` implementation.

We centralize the small validation + temporary-attribute pattern here to avoid
duplicated code blocks being counted as duplicated new code by SonarCloud.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def validate_batch_size(batch_size: Optional[int]) -> Optional[int]:
    """Validate a `batch_size` optional argument.

    Returns the integer batch size or `None`. Raises `ValueError` when provided
    but not positive.
    """

    if batch_size is None:
        return None

    batch_size_int = int(batch_size)
    if batch_size_int <= 0:
        raise ValueError(f"batch_size must be positive integer, got: {batch_size!r}")
    return batch_size_int


def call_with_temporary_attr(obj: Any, attr: str, value: int, fn: Callable[[], T]) -> T:
    """Call `fn()` with `obj.attr` temporarily set to `value` (restored after)."""

    old = getattr(obj, attr)
    try:
        setattr(obj, attr, int(value))
        return fn()
    finally:
        setattr(obj, attr, old)

