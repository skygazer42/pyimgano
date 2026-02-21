from pathlib import Path

from pyimgano.utils.optional_deps import require


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_pyproject_defines_clip_extra():
    pyproject = _read_repo_file("pyproject.toml")
    assert "[project.optional-dependencies]" in pyproject
    assert "open_clip_torch>=2.0.0" in pyproject
    assert "\nclip = [" in pyproject or "\r\nclip = [" in pyproject


def test_readme_mentions_clip_extra_install():
    readme = _read_repo_file("README.md")
    assert 'pip install "pyimgano[clip]"' in readme


def test_require_openclip_has_clip_install_hint_when_missing():
    try:
        module = require("open_clip", extra="clip", purpose="OpenCLIP backend")
    except ImportError as exc:
        message = str(exc)
        assert "Optional dependency 'open_clip'" in message
        assert "pip install 'pyimgano[clip]'" in message
    else:
        assert module is not None
