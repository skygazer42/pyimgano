from __future__ import annotations

import warnings

import pytest

pytest.importorskip("torch")


def _compact_devnet_kwargs() -> dict[str, object]:
    return {
        "pretrained": False,
        "image_size": 32,
        "n_scales": 1,
        "epochs": 1,
        "batch_size": 2,
        "steps_per_epoch": 1,
        "device": "cpu",
    }


def test_deviation_loss_returns_tensor_for_normal_only_batch() -> None:
    import torch

    from pyimgano.models.devnet import DeviationLoss

    scores = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)
    labels = torch.zeros((3,), dtype=torch.long)

    reference = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    loss = DeviationLoss(margin=5.0)(scores, labels, ref_scores=reference)

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


def test_deviation_loss_matches_gaussian_reference_z_score_objective() -> None:
    import torch

    from pyimgano.models.devnet import DeviationLoss

    scores = torch.tensor([-1.0, 6.0], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)

    reference = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    loss = DeviationLoss(margin=5.0)(scores, labels, ref_scores=reference)

    # |-1| for the normal sample and max(0, 5-6) for the anomaly sample.
    assert float(loss) == pytest.approx(0.5)


def test_deviation_loss_uses_paper_reference_size_by_default() -> None:
    from pyimgano.models.devnet import DeviationLoss

    assert DeviationLoss().reference_size == 5000


def test_balanced_batch_sampler_yields_half_normal_half_anomaly() -> None:
    import numpy as np

    from pyimgano.models.devnet import BalancedBatchSampler

    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    sampler = BalancedBatchSampler(
        labels,
        batch_size=4,
        steps_per_epoch=3,
        random_state=7,
    )

    batches = list(sampler)

    assert len(batches) == 3
    assert all(np.count_nonzero(labels[batch] == 0) == 2 for batch in batches)
    assert all(np.count_nonzero(labels[batch] == 1) == 2 for batch in batches)


def test_devnet_paper_defaults_topology_and_signed_topk() -> None:
    import torch

    from pyimgano.models.devnet import DevNetDetector, DevNetModel

    detector = DevNetDetector(pretrained=False, device="cpu")

    assert detector.image_size == 448
    assert detector.n_scales == 2
    assert detector.topk_ratio == pytest.approx(0.1)
    assert detector.epochs == 50
    assert detector.batch_size == 48
    assert detector.steps_per_epoch == 20
    assert detector.learning_rate == pytest.approx(1e-3)
    assert detector.weight_decay == pytest.approx(1e-2)
    assert detector.scheduler_step_size == 10
    assert detector.scheduler_gamma == pytest.approx(0.1)
    assert detector.model.feature_extractor.feature_dim == 512
    assert detector.model.score_head.kernel_size == (1, 1)
    assert all(parameter.requires_grad for parameter in detector.feature_extractor.parameters())
    params = detector.get_params(deep=False)
    assert params["backbone"] == "resnet18"
    assert params["weight_decay"] == pytest.approx(1e-2)

    patch_scores = torch.tensor([[[[-10.0, 1.0, 2.0, 3.0]]]])
    score = DevNetModel.aggregate_patch_scores(patch_scores, topk_ratio=0.25)
    assert score.item() == pytest.approx(3.0)


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
        **_compact_devnet_kwargs(),
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
        **_compact_devnet_kwargs(),
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
        **_compact_devnet_kwargs(),
    )

    det.fit(x_train, y_train)
    out = capsys.readouterr().out
    assert out == ""
