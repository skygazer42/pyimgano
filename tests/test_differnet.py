from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")


def _make_rgb_batch(*, count: int = 4, size: int = 32) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    out: list[np.ndarray] = []
    for _ in range(count):
        out.append(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))
    return out


def _compact_differnet_kwargs() -> dict[str, object]:
    return {
        "pretrained": False,
        "image_size": 64,
        "n_scales": 1,
        "n_flow_steps": 1,
        "n_transforms": 1,
        "n_transforms_test": 2,
        "flow_hidden_dim": 16,
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
        "random_state": 0,
    }


def test_differnet_defaults_match_paper_detection_configuration() -> None:
    from pyimgano.models.differnet import DifferNetDetector

    detector = DifferNetDetector(pretrained=False, device="cpu")

    assert detector.image_size == 448
    assert detector.n_scales == 3
    assert detector.n_flow_steps == 8
    assert detector.flow_hidden_dim == 2048
    assert detector.flow_clamp == pytest.approx(3.0)
    assert detector.flow_dropout == pytest.approx(0.0)
    assert detector.n_transforms == 4
    assert detector.n_transforms_test == 64
    assert detector.epochs == 192
    assert detector.batch_size == 24


def test_differnet_flow_matches_paper_coupling_topology() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.differnet import DifferNetFlow

    flow = DifferNetFlow(8, n_blocks=2, hidden_dim=16, clamp=3.0)
    assert flow.blocks[0].permutation.tolist() == [6, 2, 1, 7, 3, 0, 5, 4]
    for subnet in (flow.blocks[0].s1, flow.blocks[0].s2):
        linear_layers = [layer for layer in subnet if isinstance(layer, torch.nn.Linear)]
        assert len(linear_layers) == 4
        assert [layer.out_features for layer in linear_layers] == [16, 16, 16, 8]

    features = torch.randn(3, 8)
    latent, logdet = flow(features)
    restored, reverse_logdet = flow(latent, reverse=True)
    torch.testing.assert_close(restored, features, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        logdet + reverse_logdet,
        torch.zeros_like(logdet),
        rtol=0,
        atol=1e-5,
    )


def test_differnet_contract_accepts_numpy_image_list() -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)
    test = _make_rgb_batch(count=2)

    det = create_model("vision_differnet", **_compact_differnet_kwargs())

    det.fit(train)
    scores = np.asarray(det.decision_function(test), dtype=np.float64).reshape(-1)
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_differnet_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models import create_model

    train = _make_rgb_batch(count=4)

    det = create_model("vision_differnet", **_compact_differnet_kwargs())

    det.fit(train)
    out = capsys.readouterr().out
    assert out == ""
