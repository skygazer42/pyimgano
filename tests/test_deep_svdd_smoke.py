from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def test_core_deep_svdd_defaults_and_network_follow_paper_constraints() -> None:
    import torch

    from pyimgano.models.deep_svdd import CoreDeepSVDD, InnerDeepSVDD

    detector = CoreDeepSVDD(n_features=8, epochs=0, verbose=0)
    network = InnerDeepSVDD(
        n_features=8,
        use_autoencoder=False,
        hidden_neurons=[16, 8],
        hidden_activation=detector.hidden_activation,
        output_activation=detector.output_activation,
        dropout_rate=detector.dropout_rate,
    )

    assert detector.objective == "one-class"
    assert detector.nu == pytest.approx(0.1)
    assert detector.batch_size == 128
    assert detector.dropout_rate == pytest.approx(0.0)
    assert detector.l2_weight == pytest.approx(1e-6)
    assert detector.hidden_activation == "leaky_relu"
    assert detector.output_activation == "identity"
    assert isinstance(network.encoder[-1], torch.nn.Linear)
    assert not any(isinstance(layer, torch.nn.Dropout) for layer in network.encoder)
    assert all(layer.bias is None for layer in network.encoder if isinstance(layer, torch.nn.Linear))
    leaky_relu = next(layer for layer in network.encoder if isinstance(layer, torch.nn.LeakyReLU))
    assert leaky_relu.negative_slope == pytest.approx(0.1)
    assert detector.get_params(deep=False)["optimizer"] == "adam"


def test_core_deep_svdd_author_code_defaults_without_test_override() -> None:
    from pyimgano.models.deep_svdd import CoreDeepSVDD

    detector = CoreDeepSVDD(n_features=8, verbose=0)

    assert detector.epochs == 50
    assert detector.batch_size == 128
    assert detector.warm_up_epochs == 10
    assert detector.radius_update_interval == 5


def test_core_deep_svdd_soft_boundary_updates_radius_and_signed_scores() -> None:
    import torch

    from pyimgano.models.deep_svdd import CoreDeepSVDD

    rng = np.random.default_rng(19)
    x = rng.normal(size=(24, 8)).astype(np.float32)
    detector = CoreDeepSVDD(
        n_features=8,
        objective="soft-boundary",
        nu=0.25,
        warm_up_epochs=0,
        radius_update_interval=1,
        hidden_neurons=[16, 8],
        epochs=2,
        batch_size=8,
        preprocessing=False,
        verbose=0,
        random_state=3,
    )

    detector.fit(x)

    assert detector.radius_ > 0.0
    assert detector.get_params(deep=False)["center"] is None
    with torch.no_grad():
        representations = detector.model.encode(torch.tensor(x))
        raw_distance = torch.sum((representations - detector.center_) ** 2, dim=-1).numpy()
    assert detector.radius_ == pytest.approx(np.quantile(np.sqrt(raw_distance), 0.75))
    expected = raw_distance - detector.radius_**2
    np.testing.assert_allclose(detector.decision_function(x), expected, rtol=1e-5, atol=1e-6)


def test_core_deep_svdd_runs_on_cuda_when_requested() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from pyimgano.models.deep_svdd import CoreDeepSVDD

    rng = np.random.default_rng(23)
    x = rng.normal(size=(16, 8)).astype(np.float32)
    detector = CoreDeepSVDD(
        n_features=8,
        hidden_neurons=[16, 8],
        epochs=1,
        batch_size=8,
        preprocessing=False,
        device="cuda",
        verbose=0,
        random_state=5,
    )

    detector.fit(x)

    assert detector.center_.device.type == "cuda"
    assert np.isfinite(detector.decision_function(x)).all()


def test_core_deep_svdd_smoke_can_fit_and_score() -> None:
    from pyimgano.models.deep_svdd import CoreDeepSVDD

    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 8)).astype(np.float32)

    det = CoreDeepSVDD(
        n_features=8,
        hidden_neurons=[16, 8],
        use_autoencoder=True,
        epochs=1,
        batch_size=4,
        verbose=0,
        random_state=0,
        contamination=0.2,
    )
    det.fit(x)

    scores = det.decision_function(x[:5])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()

    labels = det.predict(x[:5])
    assert set(np.unique(labels)).issubset({0, 1})


def test_vision_deep_svdd_smoke_can_fit_with_dummy_features() -> None:
    from pyimgano.models import create_model

    class DummyFeatureExtractor:
        def __init__(self, feature_dim: int = 8) -> None:
            self.feature_dim = int(feature_dim)

        def extract(self, inputs):
            inputs = list(inputs)
            rng = np.random.default_rng(123)
            return rng.normal(size=(len(inputs), self.feature_dim)).astype(np.float32)

    det = create_model(
        "vision_deep_svdd",
        feature_extractor=DummyFeatureExtractor(feature_dim=8),
        n_features=8,
        hidden_neurons=[16, 8],
        use_autoencoder=True,
        epochs=1,
        batch_size=2,
        verbose=0,
        random_state=0,
        contamination=0.2,
    )

    train = ["a.png", "b.png", "c.png", "d.png"]
    test = ["e.png", "f.png", "g.png"]
    det.fit(train)
    scores = det.decision_function(test)
    assert scores.shape == (len(test),)
    assert np.isfinite(scores).all()


def test_core_deep_svdd_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.deep_svdd import CoreDeepSVDD

    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 8)).astype(np.float32)

    det = CoreDeepSVDD(
        n_features=8,
        hidden_neurons=[16, 8],
        use_autoencoder=True,
        epochs=2,
        batch_size=4,
        verbose=1,
        random_state=0,
        contamination=0.2,
    )

    det.fit(x)
    out = capsys.readouterr().out
    assert out == ""
