from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _assert_preprocess_delegates(*, module, cls_name: str, monkeypatch) -> None:
    sentinel = object()

    def _fake_helper(x):  # noqa: ANN001, ANN201
        assert isinstance(x, np.ndarray)
        return sentinel

    monkeypatch.setattr(module, "preprocess_imagenet_batch", _fake_helper, raising=False)
    cls = getattr(module, cls_name)
    inst = cls.__new__(cls)
    sample = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    assert cls._preprocess(inst, sample) is sentinel


def test_panda_preprocess_uses_shared_helper(monkeypatch) -> None:
    import torch

    import pyimgano.models.panda as module

    called = False

    def _fake_helper(x):  # noqa: ANN001, ANN202
        nonlocal called
        called = True
        assert isinstance(x, np.ndarray)
        return torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    monkeypatch.setattr(module, "preprocess_imagenet_batch", _fake_helper)
    inst = module.VisionPANDA.__new__(module.VisionPANDA)
    inst.resize_size = 4
    inst.image_size = 4
    output = module.VisionPANDA._preprocess(inst, np.zeros((1, 4, 4, 3), dtype=np.uint8))

    assert called
    assert tuple(output.shape) == (1, 3, 4, 4)


def test_ast_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.ast as module

    _assert_preprocess_delegates(module=module, cls_name="VisionAST", monkeypatch=monkeypatch)


def test_dst_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.dst as module

    _assert_preprocess_delegates(module=module, cls_name="VisionDST", monkeypatch=monkeypatch)


def test_favae_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.favae as module

    _assert_preprocess_delegates(module=module, cls_name="VisionFAVAE", monkeypatch=monkeypatch)
