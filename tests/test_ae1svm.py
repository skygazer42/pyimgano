from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _make_rgb_batch(*, count: int = 4, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(4)
    out: list[np.ndarray] = []
    for _ in range(count):
        out.append(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    return out


def test_vision_ae1svm_contract_fit_and_score() -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)
    test = _make_rgb_batch(count=2)

    det = create_model(
        "vision_ae1svm",
        image_shape=(3, 32, 32),
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        device="cpu",
        verbose=0,
        contamination=0.25,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
