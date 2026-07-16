from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_fcdd_matches_the_paper_mvtec_network_loss_and_defaults() -> None:
    from pyimgano.models.fcdd import FCDD, FCDDNetwork, _fcdd_loss, _pseudo_huber

    network = FCDDNetwork(pretrained=False, freeze_features=True)
    assert sum(parameter.numel() for parameter in network.parameters()) == 4_504_833
    assert all(not parameter.requires_grad for parameter in network.features[:15].parameters())
    assert all(parameter.requires_grad for parameter in network.features[15:].parameters())
    assert network(torch.zeros(1, 3, 224, 224)).shape == (1, 1, 28, 28)

    outputs = torch.ones(2, 1, 28, 28)
    paper_score = math.sqrt(2.0) - 1.0
    expected_anomaly_loss = -math.log(1.0 - math.exp(-paper_score))
    assert _fcdd_loss(outputs[:1], torch.tensor([0])).item() == pytest.approx(paper_score)
    assert _fcdd_loss(outputs[:1], torch.tensor([1])).item() == pytest.approx(expected_anomaly_loss)
    assert FCDD.receptive_upsample(_pseudo_huber(outputs)).shape == (2, 1, 224, 224)
    normal = torch.full((3, 240, 240), 0.5)
    confetti = FCDD._confetti_image(normal, np.random.default_rng(3))
    assert confetti.shape == normal.shape
    assert not torch.equal(confetti, normal)

    detector = FCDD(device="cpu")
    assert detector.learning_rate == pytest.approx(1e-3)
    assert detector.weight_decay == pytest.approx(1e-4)
    assert detector.lr_decay == pytest.approx(0.985)
    assert detector.batch_size == 16
    assert detector.accumulate_batches == 8
    assert detector.epoch_size_multiplier == 10
    assert detector.epochs == 200
    assert detector.gaussian_sigma == pytest.approx(12.0)
    assert detector.synthetic_anomalies
