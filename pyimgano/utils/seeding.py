from __future__ import annotations


def seed_everything(seed: int) -> None:
    """Best-effort deterministic seeding across common libraries.

    This is a lightweight helper used by CLIs to improve reproducibility.
    It intentionally does not force global deterministic modes (which can
    significantly hurt performance).
    """

    import random

    import numpy as np

    s = int(seed)
    random.seed(s)
    np.random.seed(s)

    try:  # pragma: no cover - depends on optional torch install
        import torch
    except Exception:
        return

    try:  # pragma: no cover - best-effort torch seeding
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        return

