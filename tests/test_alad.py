from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_vision_alad_contract_fit_and_score() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(10)
    train = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(4)]
    test = [rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(2)]

    det = create_model(
        "vision_alad",
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        device="cpu",
        verbose=0,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
