from pathlib import Path

import numpy as np
import pytest

import pyimgano.utils.optional_deps as optional_deps


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_pyproject_defines_mamba_extra():
    pyproject = _read_repo_file("pyproject.toml")
    assert "[project.optional-dependencies]" in pyproject
    assert "mamba-ssm>=2.0.0" in pyproject
    assert "\nmamba = [" in pyproject or "\r\nmamba = [" in pyproject


def test_require_mamba_has_mamba_install_hint_when_missing(monkeypatch):
    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "mamba_ssm":
            return None, ModuleNotFoundError("No module named 'mamba_ssm'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    with pytest.raises(ImportError) as excinfo:
        optional_deps.require("mamba_ssm", extra="mamba", purpose="MambaAD backend")

    message = str(excinfo.value)
    assert "Optional dependency 'mamba_ssm'" in message
    assert "pip install 'pyimgano[mamba]'" in message


def test_mamba_model_registered_without_mamba_ssm():
    import pyimgano.models as models

    available = models.list_models()
    assert "vision_mambaad" in available


def test_fit_mambaad_raises_importerror_when_mamba_ssm_missing(monkeypatch):
    import pyimgano.models as models

    class DummyEmbedder:
        def embed(self, image):
            patches = np.zeros((4, 8), dtype=np.float32)  # 2x2 grid, dim=8
            return patches, (2, 2), (32, 32)

    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "mamba_ssm":
            return None, ModuleNotFoundError("No module named 'mamba_ssm'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    det = models.create_model(
        "vision_mambaad",
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

