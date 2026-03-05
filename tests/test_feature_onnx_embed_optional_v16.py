from __future__ import annotations

import numpy as np
import pytest

import pyimgano.utils.optional_deps as optional_deps


def test_onnx_embed_extractor_raises_importerror_when_onnxruntime_missing(monkeypatch) -> None:
    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "onnxruntime":
            return None, ModuleNotFoundError("No module named 'onnxruntime'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    extractor = ONNXEmbedExtractor(
        checkpoint="missing.onnx", device="cpu", batch_size=1, image_size=224
    )

    with pytest.raises(ImportError) as excinfo:
        extractor.extract([np.zeros((8, 8, 3), dtype=np.uint8)])

    message = str(excinfo.value)
    assert "Optional dependency 'onnxruntime'" in message
    assert "pip install 'pyimgano[onnx]'" in message
