from __future__ import annotations

from pathlib import Path

import pytest

import pyimgano.utils.optional_deps as optional_deps


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_pyproject_defines_torch_onnx_openvino_extras() -> None:
    pyproject = _read_repo_file("pyproject.toml")
    assert "[project.optional-dependencies]" in pyproject

    assert "\ntorch = [" in pyproject or "\r\ntorch = [" in pyproject
    assert "\nonnx = [" in pyproject or "\r\nonnx = [" in pyproject
    assert "\nopenvino = [" in pyproject or "\r\nopenvino = [" in pyproject

    # Sanity: onnx extra should pull onnxruntime and onnxscript (torch.onnx exporter).
    assert "onnxruntime" in pyproject
    assert "onnxscript" in pyproject

    # Sanity: "all" should aggregate the new core extras.
    assert "pyimgano[backends" in pyproject
    assert "torch,onnx,openvino,skimage,numba" in pyproject


def test_require_torch_mentions_pyimgano_torch_extra_when_missing(monkeypatch) -> None:
    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "torch":
            return None, ModuleNotFoundError("No module named 'torch'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    with pytest.raises(ImportError) as excinfo:
        optional_deps.require("torch", extra="torch", purpose="unit-test")

    message = str(excinfo.value)
    assert "Optional dependency 'torch'" in message
    assert "pip install 'pyimgano[torch]'" in message


def test_require_onnxruntime_mentions_pyimgano_onnx_extra_when_missing(monkeypatch) -> None:
    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "onnxruntime":
            return None, ModuleNotFoundError("No module named 'onnxruntime'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    with pytest.raises(ImportError) as excinfo:
        optional_deps.require("onnxruntime", extra="onnx", purpose="unit-test")

    message = str(excinfo.value)
    assert "Optional dependency 'onnxruntime'" in message
    assert "pip install 'pyimgano[onnx]'" in message


def test_models_lazy_ctor_maps_missing_torch_to_pyimgano_torch_extra(monkeypatch) -> None:
    import pyimgano.models as models

    def fake_import_module(_name: str):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(models, "import_module", fake_import_module)

    ctor = models._make_lazy_constructor(model_name="vision_fake_torch", module_name="fake_module")

    with pytest.raises(ImportError) as excinfo:
        ctor()

    message = str(excinfo.value)
    assert "vision_fake_torch" in message
    assert "pip install 'pyimgano[torch]'" in message


def test_models_lazy_ctor_maps_missing_onnxruntime_to_pyimgano_onnx_extra(monkeypatch) -> None:
    import pyimgano.models as models

    def fake_import_module(_name: str):
        raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")

    monkeypatch.setattr(models, "import_module", fake_import_module)

    ctor = models._make_lazy_constructor(model_name="vision_fake_onnx", module_name="fake_module")

    with pytest.raises(ImportError) as excinfo:
        ctor()

    message = str(excinfo.value)
    assert "vision_fake_onnx" in message
    assert "pip install 'pyimgano[onnx]'" in message


def test_models_lazy_ctor_maps_missing_openvino_to_pyimgano_openvino_extra(monkeypatch) -> None:
    import pyimgano.models as models

    def fake_import_module(_name: str):
        raise ModuleNotFoundError("No module named 'openvino'", name="openvino")

    monkeypatch.setattr(models, "import_module", fake_import_module)

    ctor = models._make_lazy_constructor(
        model_name="vision_fake_openvino", module_name="fake_module"
    )

    with pytest.raises(ImportError) as excinfo:
        ctor()

    message = str(excinfo.value)
    assert "vision_fake_openvino" in message
    assert "pip install 'pyimgano[openvino]'" in message
