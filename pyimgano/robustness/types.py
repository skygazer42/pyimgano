from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray


class Corruption(Protocol):
    """A deterministic image corruption used for robustness benchmarking."""

    name: str

    def __call__(
        self,
        image: NDArray,
        mask: Optional[NDArray],
        *,
        severity: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray, Optional[NDArray]]: ...

