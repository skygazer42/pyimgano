from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _make_rgb_batch(*, count: int = 4, size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(41)
    return rng.integers(0, 255, size=(count, size, size, 3), dtype=np.uint8)


def test_panda_fit_does_not_print_progress(monkeypatch, capsys) -> None:
    import pyimgano.models.panda as panda_module
    from pyimgano.models.panda import VisionPANDA

    class _DummyEncoder(torch.nn.Module):
        def __init__(
            self,
            backbone: str = "resnet152",
            *,
            pretrained: bool = True,
            weights_name: str = "IMAGENET1K_V1",
        ) -> None:
            del backbone, pretrained, weights_name
            super().__init__()
            self.proj = torch.nn.Linear(3, 8)

        def forward(self, x):  # noqa: ANN001
            return self.proj(x.mean(dim=(-1, -2)))

    monkeypatch.setattr(panda_module, "PANDAEncoder", _DummyEncoder)

    det = VisionPANDA(
        backbone="resnet18",
        pretrained=False,
        batch_size=2,
        training_steps=2,
        device="cpu",
        random_state=0,
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""


def test_glad_fit_does_not_print_progress(monkeypatch, capsys) -> None:
    import torch.nn.functional as F

    from pyimgano.models.glad import VisionGLAD

    class _DummyExtractor(torch.nn.Module):
        def forward(self, x):  # noqa: ANN001
            pooled = F.avg_pool2d(x, kernel_size=2)
            return torch.cat([pooled, pooled[:, :1]], dim=1)

    monkeypatch.setattr(VisionGLAD, "_build_feature_extractor", lambda self: _DummyExtractor())

    det = VisionGLAD(
        backbone="resnet18",
        num_timesteps=20,
        batch_size=2,
        epochs=10,
        device="cpu",
        random_state=0,
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""


def test_inctrl_fit_does_not_print_progress(capsys) -> None:
    from pyimgano.models.inctrl import VisionInCTRL

    class _Backend:
        def fit(self, items, class_name):  # noqa: ANN001, ANN201
            del items, class_name

        def score(self, item):  # noqa: ANN001, ANN201
            return float(np.asarray(item).mean())

    det = VisionInCTRL(
        backend=_Backend(),
        batch_size=2,
        k_shot=2,
        device="cpu",
        random_state=0,
    )

    det.fit(_make_rgb_batch())
    out = capsys.readouterr().out
    assert out == ""
