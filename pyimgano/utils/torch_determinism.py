from __future__ import annotations

"""Torch determinism helpers (best-effort).

Industrial pipelines often need reproducibility for:
- debugging regressions
- benchmark comparisons
- model-card generation

This module provides a small helper without imposing a strict global policy.
"""


def set_torch_determinism(
    seed: int = 42,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> None:
    """Set seeds and (optionally) deterministic flags.

    Notes
    -----
    - Determinism can reduce performance.
    - Some ops remain nondeterministic depending on hardware/backend.
    """

    import os
    import random

    import numpy as np

    from pyimgano.utils.optional_deps import require

    torch = require("torch", extra="torch", purpose="set_torch_determinism")

    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    # cuDNN / CUDA behavior
    try:
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(deterministic)
    except Exception:
        pass

    if deterministic:
        # For some CUDA ops.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
