from __future__ import annotations

import pytest


def test_flow_blocks_are_invertible() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.cflow import ConditionalFlow
    from pyimgano.models.fastflow import FlowStep

    x = torch.randn(3, 4, 5, 5)
    step = FlowStep(4)
    z, logdet = step(x, torch.zeros(3))
    restored, reverse_logdet = step(z, logdet, reverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reverse_logdet, torch.zeros_like(reverse_logdet), atol=1e-5, rtol=0)

    features = torch.randn(6, 8)
    condition = torch.randn(6, 3)
    flow = ConditionalFlow(8, 3, n_flows=3)
    latent, logdet = flow(features, condition)
    restored, reverse_logdet = flow(latent, condition, reverse=True)
    torch.testing.assert_close(restored, features, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logdet + reverse_logdet, torch.zeros_like(logdet), atol=1e-5, rtol=0)


def test_differnet_inference_uses_latent_energy_not_training_logdet() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.differnet import DifferNetDetector

    class DummyFlow:
        def eval(self):
            return self

        def __call__(self, images):
            assert images.shape[0] == 4
            z = torch.tensor(
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                dtype=torch.float32,
            )
            return z, torch.full((4,), 1_000.0)

    detector = DifferNetDetector.__new__(DifferNetDetector)
    detector.device = torch.device("cpu")
    detector.n_transforms = 2
    detector.model = DummyFlow()

    scores = detector.evaluating_forward(
        (torch.zeros((2, 3, 4, 4), dtype=torch.float32), torch.zeros(2))
    )

    assert scores.tolist() == pytest.approx([5.0, 10.0])
