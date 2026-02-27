from __future__ import annotations

import numpy as np
import pytest


def test_openclip_extractor_is_discoverable() -> None:
    import pyimgano.features  # noqa: F401 - registry population side effects

    from pyimgano.features.registry import list_feature_extractors

    assert "openclip_embed" in list_feature_extractors()


def test_openclip_extractor_raises_clean_error_when_missing() -> None:
    from pyimgano.utils.optional_deps import optional_import

    mod, _err = optional_import("open_clip")
    if mod is not None:
        pytest.skip("open_clip is installed; skip missing-dep error path")

    from pyimgano.features.openclip_embed import OpenCLIPExtractor

    ext = OpenCLIPExtractor(pretrained=None, device="cpu", batch_size=1)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ImportError) as excinfo:
        ext.extract([img])
    assert "open_clip" in str(excinfo.value) or "open_clip_torch" in str(excinfo.value)

