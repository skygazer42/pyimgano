from __future__ import annotations


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def test_gem_pool2d_constant_is_identity() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch is not installed")

    import torch

    from pyimgano.features.pooling import gem_pool2d

    x = torch.ones((2, 3, 4, 5), dtype=torch.float32)
    y = gem_pool2d(x, p=3.0, eps=1e-6)
    assert tuple(y.shape) == (2, 3)
    assert torch.allclose(y, torch.ones_like(y), atol=1e-6, rtol=0.0)


def test_gem_pool2d_negative_inputs_are_finite() -> None:
    import pytest

    if not _torch_available():
        pytest.skip("torch is not installed")

    import torch

    from pyimgano.features.pooling import gem_pool2d

    x = -torch.ones((1, 2, 8, 8), dtype=torch.float32)
    y = gem_pool2d(x, p=3.0, eps=1e-3)
    assert tuple(y.shape) == (1, 2)
    assert torch.isfinite(y).all()
    assert (y >= 1e-3).all()

