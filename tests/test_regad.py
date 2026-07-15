from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


class _TinyRegADModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, kernel_size=1)

    def _features(self, images):  # noqa: ANN001
        return torch.nn.functional.avg_pool2d(self.projection(images), 4)

    def registration_loss(self, query, support):  # noqa: ANN001
        query_features = self._features(query)
        batch, shot, channels, height, width = support.shape
        support_features = self._features(
            support.reshape(batch * shot, channels, height, width)
        ).reshape(batch, shot, *query_features.shape[1:])
        return (query_features - support_features.mean(dim=1)).square().mean()

    def aligned_features(self, images):  # noqa: ANN001
        return self._features(images)


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(21)
    source = rng.integers(0, 255, size=(4, 32, 32, 3), dtype=np.uint8)
    labels = np.asarray([0, 0, 1, 1])
    support = rng.integers(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)
    query = rng.integers(0, 255, size=(2, 32, 32, 3), dtype=np.uint8)
    return source, labels, support, query


def _detector(monkeypatch):  # noqa: ANN001
    import pyimgano.models.regad as regad_module
    from pyimgano.models import create_model

    monkeypatch.setattr(regad_module, "RegADModel", _TinyRegADModel)
    return create_model(
        "vision_regad",
        pretrained=False,
        image_size=32,
        epochs=1,
        batch_size=2,
        shot=2,
        gaussian_sigma=0,
        device="cpu",
        random_state=0,
    )


def test_vision_regad_contract_fit_score_and_map(monkeypatch) -> None:  # noqa: ANN001
    source, labels, support, query = _inputs()
    detector = _detector(monkeypatch)

    detector.fit(source, labels, support_images=support)
    scores = np.asarray(detector.decision_function(query), dtype=np.float64)
    maps = np.asarray(detector.predict_anomaly_map(query), dtype=np.float64)

    assert scores.shape == (2,)
    assert maps.shape == (2, 32, 32)
    assert np.isfinite(scores).all()
    assert np.isfinite(maps).all()


def test_vision_regad_fit_does_not_print_progress(monkeypatch, capsys) -> None:  # noqa: ANN001
    source, labels, support, _ = _inputs()
    detector = _detector(monkeypatch)

    detector.fit(source, labels, support_images=support)

    assert capsys.readouterr().out == ""
