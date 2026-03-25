from __future__ import annotations

import pytest


def test_fastflow_invconv_logdet_is_finite_for_negative_determinant_seed() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import InvConv2d

    torch.manual_seed(0)
    layer = InvConv2d(4)

    value = layer._log_det()
    assert torch.isfinite(value).item() is True


def test_fastflow_invconv_forward_produces_finite_logdet() -> None:
    torch = pytest.importorskip("torch")

    from pyimgano.models.fastflow import InvConv2d

    torch.manual_seed(0)
    layer = InvConv2d(4)
    x = torch.randn(2, 4, 8, 8)
    logdet = torch.zeros(2)

    _y, updated = layer(x, logdet, reverse=False)
    assert torch.isfinite(updated).all().item() is True
