from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_panda_paper_defaults_and_squared_l2_two_nn_score() -> None:
    from pyimgano.models.panda import VisionPANDA

    detector = VisionPANDA(pretrained=False, device="cpu")

    assert detector.backbone == "resnet152"
    assert detector.batch_size == 32
    assert detector.training_steps == 2300
    assert detector.learning_rate == pytest.approx(1e-2)
    assert detector.momentum == pytest.approx(0.9)
    assert detector.weight_decay == pytest.approx(5e-5)
    assert detector.grad_clip_norm == pytest.approx(1e-3)
    assert detector.n_neighbors == 2

    detector.memory_bank_ = torch.tensor([[0.0], [2.0], [5.0]])
    detector.batch_size = 2
    scores = detector._score_features(torch.tensor([[1.0], [4.0]]))

    np.testing.assert_allclose(scores, [2.0, 5.0])
