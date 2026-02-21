from pathlib import Path

import pytest

import pyimgano.utils.optional_deps as optional_deps


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_pyproject_defines_clip_extra():
    pyproject = _read_repo_file("pyproject.toml")
    assert "[project.optional-dependencies]" in pyproject
    assert "open_clip_torch>=2.0.0" in pyproject
    assert "\nclip = [" in pyproject or "\r\nclip = [" in pyproject
    assert "pyimgano[anomalib,faiss,clip]" in pyproject
    assert "pyimgano[backends" in pyproject


def test_readme_mentions_clip_extra_install():
    readme = _read_repo_file("README.md")
    assert 'pip install "pyimgano[clip]"' in readme


def test_require_openclip_has_clip_install_hint_when_missing(monkeypatch):
    original_optional_import = optional_deps.optional_import

    def fake_optional_import(module_name: str):
        if module_name == "open_clip":
            return None, ModuleNotFoundError("No module named 'open_clip'")
        return original_optional_import(module_name)

    monkeypatch.setattr(optional_deps, "optional_import", fake_optional_import)

    with pytest.raises(ImportError) as excinfo:
        optional_deps.require("open_clip", extra="clip", purpose="OpenCLIP backend")

    message = str(excinfo.value)
    assert "Optional dependency 'open_clip'" in message
    assert "pip install 'pyimgano[clip]'" in message
