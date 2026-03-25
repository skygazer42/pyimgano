from pathlib import Path


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def test_pyproject_no_longer_defines_mamba_extra() -> None:
    pyproject = _read_repo_file("pyproject.toml")
    assert "\nmamba = [" not in pyproject and "\r\nmamba = [" not in pyproject
    assert "mamba-ssm>=2.0.0" not in pyproject


def test_mamba_models_are_not_discoverable() -> None:
    import pyimgano.models as models

    available = set(models.list_models())
    assert "vision_mambaad" not in available
    assert "vision_snarm" not in available

