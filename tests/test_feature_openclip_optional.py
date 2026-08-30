from __future__ import annotations

import numpy as np
import pytest


def test_openclip_extractor_is_discoverable() -> None:
    import pyimgano.features  # noqa: F401 - registry population side effects
    from pyimgano.features.registry import list_feature_extractors

    assert "openclip_embed" in list_feature_extractors()


def test_openclip_extractor_raises_clean_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyimgano.features import openclip_embed

    def missing_openclip(module_name: str, **_kwargs: object):
        if module_name == "open_clip":
            raise ImportError("simulated missing open_clip")
        raise AssertionError(f"unexpected dependency request: {module_name}")

    monkeypatch.setattr(openclip_embed, "require", missing_openclip)

    ext = openclip_embed.OpenCLIPExtractor(pretrained=None, device="cpu", batch_size=1)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError) as excinfo:
        ext.extract([img])
    assert "open_clip" in str(excinfo.value) or "open_clip_torch" in str(excinfo.value)
