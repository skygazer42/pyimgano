from __future__ import annotations

import warnings

import pytest

pytest.importorskip("torch")


def test_deviation_loss_returns_tensor_for_normal_only_batch() -> None:
    import torch

    from pyimgano.models.devnet import DeviationLoss

    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
    labels = torch.zeros((3,), dtype=torch.long)

    loss = DeviationLoss(margin=5.0)(scores, labels)

    assert isinstance(loss, torch.Tensor)
    loss.backward()
    assert scores.grad is not None


def test_deviation_loss_returns_tensor_for_anomaly_only_batch() -> None:
    import torch

    from pyimgano.models.devnet import DeviationLoss

    scores = torch.tensor([1.0, 1.5], dtype=torch.float32, requires_grad=True)
    labels = torch.ones((2,), dtype=torch.long)

    loss = DeviationLoss(margin=5.0)(scores, labels)

    assert isinstance(loss, torch.Tensor)
    loss.backward()
    assert scores.grad is not None


def test_vision_devnet_contract_fit_and_score() -> None:
    import numpy as np

    from pyimgano.models import create_model

    rng = np.random.default_rng(5)
    normals = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    anomalies = normals.copy()
    anomalies[:, 8:24, 8:24, :] = 255

    x_train = np.concatenate([normals, anomalies], axis=0)
    y_train = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    order = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
    x_train = x_train[order]
    y_train = y_train[order]

    test = rng.integers(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)

    det = create_model(
        "vision_devnet",
        pretrained=False,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    det.fit(x_train, y_train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_vision_devnet_fit_does_not_warn_for_required_labels() -> None:
    import numpy as np

    from pyimgano.models import create_model

    rng = np.random.default_rng(11)
    normals = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    anomalies = normals.copy()
    anomalies[:, 10:22, 10:22, :] = 255

    x_train = np.concatenate([normals, anomalies], axis=0)
    y_train = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    order = np.asarray([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
    x_train = x_train[order]
    y_train = y_train[order]

    det = create_model(
        "vision_devnet",
        pretrained=False,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        det.fit(x_train, y_train)

    messages = [str(item.message) for item in caught]
    assert "y should not be presented in unsupervised learning." not in messages


def test_vision_devnet_fit_does_not_print_progress(capsys) -> None:
    import numpy as np

    from pyimgano.models import create_model

    rng = np.random.default_rng(17)
    normals = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    anomalies = normals.copy()
    anomalies[:, 10:22, 10:22, :] = 255

    x_train = np.concatenate([normals, anomalies], axis=0)
    y_train = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    det = create_model(
        "vision_devnet",
        pretrained=False,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    det.fit(x_train, y_train)
    out = capsys.readouterr().out
    assert out == ""
