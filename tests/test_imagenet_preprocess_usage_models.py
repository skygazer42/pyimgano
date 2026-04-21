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


def test_bayesianpf_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.bayesianpf as module

    _assert_preprocess_delegates(
        module=module, cls_name="VisionBayesianPF", monkeypatch=monkeypatch
    )


def test_glad_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.glad as module

    _assert_preprocess_delegates(module=module, cls_name="VisionGLAD", monkeypatch=monkeypatch)


def test_panda_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.panda as module

    _assert_preprocess_delegates(module=module, cls_name="VisionPANDA", monkeypatch=monkeypatch)


def test_inctrl_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.inctrl as module

    _assert_preprocess_delegates(module=module, cls_name="VisionInCTRL", monkeypatch=monkeypatch)


def test_ast_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.ast as module

    _assert_preprocess_delegates(module=module, cls_name="VisionAST", monkeypatch=monkeypatch)


def test_dst_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.dst as module

    _assert_preprocess_delegates(module=module, cls_name="VisionDST", monkeypatch=monkeypatch)


def test_favae_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.favae as module

    _assert_preprocess_delegates(module=module, cls_name="VisionFAVAE", monkeypatch=monkeypatch)


def test_promptad_preprocess_uses_shared_helper(monkeypatch) -> None:
    import pyimgano.models.promptad as module

    _assert_preprocess_delegates(module=module, cls_name="VisionPromptAD", monkeypatch=monkeypatch)
