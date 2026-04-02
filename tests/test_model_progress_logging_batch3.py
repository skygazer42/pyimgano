from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _make_rgb_batch(*, count: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(45)
    return rng.integers(0, 255, size=(count, size, size, 3), dtype=np.uint8)


def test_fcdd_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.fcdd import FCDD

    det = FCDD(
        objective="occ",
        learning_rate=1e-4,
        batch_size=2,
        epochs=10,
        device="cpu",
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""


def test_memae_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.memae import MemAE

    det = MemAE(
        mem_dim=16,
        shrink_thres=0.0,
        entropy_weight=0.0,
        learning_rate=1e-4,
        batch_size=2,
        epochs=10,
        device="cpu",
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""


def test_intra_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.intra import InTraDetector

    det = InTraDetector(
        img_size=16,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=4,
        epochs=10,
        batch_size=2,
        learning_rate=1e-4,
        device="cpu",
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""
