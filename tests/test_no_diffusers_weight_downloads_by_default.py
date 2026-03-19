from __future__ import annotations


def test_diffusion_augmentor_defaults_to_local_files_only(monkeypatch) -> None:  # noqa: ANN001
    import pytest

    pytest.importorskip("diffusers")

    from pyimgano.utils.augmentation import (
        DiffusionAugmentor,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionPipeline,
    )

    def _fake_from_pretrained(cls, model, *args, **kwargs):  # noqa: ANN001, ANN201
        del cls, model, args
        assert kwargs.get("local_files_only", None) is True
        raise RuntimeError("stop-before-loading")

    monkeypatch.setattr(
        StableDiffusionPipeline, "from_pretrained", classmethod(_fake_from_pretrained), raising=True
    )
    with pytest.raises(RuntimeError, match="stop-before-loading"):
        DiffusionAugmentor(pipeline="txt2img")

    if StableDiffusionImg2ImgPipeline is not None:
        monkeypatch.setattr(
            StableDiffusionImg2ImgPipeline,
            "from_pretrained",
            classmethod(_fake_from_pretrained),
            raising=True,
        )
        with pytest.raises(RuntimeError, match="stop-before-loading"):
            DiffusionAugmentor(pipeline="img2img")
