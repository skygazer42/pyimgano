import numpy as np
import pytest

import pyimgano.utils.optional_deps as optional_deps


def test_snarm_model_registered_without_mamba_ssm():
    import pyimgano.models as models

    available = models.list_models()
    assert "vision_snarm" in available


def test_fit_snarm_raises_importerror_when_mamba_ssm_missing(monkeypatch):
    pytest.importorskip("torch")

    import pyimgano.models as models

    class DummyEmbedder:
        def embed(self, image):
            patches = np.zeros((4, 8), dtype=np.float32)
            return patches, (2, 2), (32, 32)

    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "mamba_ssm":
            return None, ModuleNotFoundError("No module named 'mamba_ssm'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    det = models.create_model(
        "vision_snarm",
        embedder=DummyEmbedder(),
        device="cpu",
        epochs=1,
        batch_size=1,
        lr=1e-3,
    )

    imgs = [np.zeros((32, 32, 3), dtype=np.uint8)]

    with pytest.raises(ImportError) as excinfo:
        det.fit(imgs)

    message = str(excinfo.value)
    assert "Optional dependency 'mamba_ssm'" in message
    assert "pip install 'pyimgano[mamba]'" in message
