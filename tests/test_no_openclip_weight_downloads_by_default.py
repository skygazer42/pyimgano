from __future__ import annotations

import sys
import types

import pytest


def test_openclip_extractor_default_pretrained_is_none() -> None:
    from pyimgano.features.openclip_embed import OpenCLIPExtractor

    ex = OpenCLIPExtractor()
    assert ex.pretrained is None, "default must be None to avoid implicit weight downloads"


def test_openclip_extractor_passes_pretrained_none(monkeypatch) -> None:
    """Guardrail: default extractor should not request pretrained weights.

    We simulate an `open_clip` module and assert the extractor forwards
    `pretrained=None` into `create_model_and_transforms(...)`.
    """
    pytest.importorskip("torch")

    calls: list[object] = []

    def create_model_and_transforms(model_name: str, *, pretrained=None, **kwargs):  # noqa: ANN001
        del model_name, kwargs
        calls.append(pretrained)

        class DummyModel:
            def to(self, _dev):  # noqa: ANN001
                return self

            def eval(self):
                return self

        def preprocess(_im):  # noqa: ANN001
            raise RuntimeError("preprocess should not be used in this guardrail test")

        return DummyModel(), preprocess

    mod = types.ModuleType("open_clip")
    mod.create_model_and_transforms = create_model_and_transforms  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "open_clip", mod)

    from pyimgano.features.openclip_embed import OpenCLIPExtractor

    ex = OpenCLIPExtractor()
    ex._ensure_ready()
    assert calls == [None]


def test_openclip_backend_rejects_implicit_pretrained_download() -> None:
    from pyimgano.models.openclip_backend import _load_openclip_model_and_preprocess

    with pytest.raises(RuntimeError, match="allow_download=True"):
        _load_openclip_model_and_preprocess(
            open_clip_module=object(),
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device="cpu",
        )
