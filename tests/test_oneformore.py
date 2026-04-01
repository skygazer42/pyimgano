from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_vision_oneformore_contract_fit_and_score() -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(32)
    train = rng.integers(0, 255, size=(4, 64, 64, 3), dtype=np.uint8)
    test = rng.integers(0, 255, size=(2, 64, 64, 3), dtype=np.uint8)

    det = create_model(
        "vision_oneformore",
        backbone="resnet18",
        hidden_channels=32,
        num_timesteps=32,
        learning_rate=1e-4,
        batch_size=2,
        epochs=1,
        use_gradient_projection=False,
        device="cpu",
        random_state=0,
    )

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_vision_oneformore_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models import create_model

    rng = np.random.default_rng(33)
    train = rng.integers(0, 255, size=(4, 64, 64, 3), dtype=np.uint8)

    det = create_model(
        "vision_oneformore",
        backbone="resnet18",
        hidden_channels=32,
        num_timesteps=32,
        learning_rate=1e-4,
        batch_size=2,
        epochs=10,
        use_gradient_projection=False,
        device="cpu",
        random_state=0,
    )

    det.fit(train)
    out = capsys.readouterr().out
    assert out == ""
