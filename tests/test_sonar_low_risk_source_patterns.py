from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_TARGETED_RNG_MODULES = (
    "pyimgano/models/cutpaste.py",
    "pyimgano/models/draem.py",
    "pyimgano/models/loda.py",
    "pyimgano/preprocessing/augmentation.py",
    "pyimgano/utils/augmentation.py",
    "pyimgano/utils/random_state.py",
)


def _read_source(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_targeted_rng_modules_avoid_seedsequence_default_rng_construction() -> None:
    for relative_path in _TARGETED_RNG_MODULES:
        source = _read_source(relative_path)
        assert "SeedSequence()" not in source, relative_path


def test_loda_visualization_literals_are_centralized() -> None:
    source = _read_source("pyimgano/models/loda.py")

    assert '"matplotlib.pyplot", extra="viz", purpose="VisionLODA visualization"' not in source


def test_datasets_image_globs_are_centralized() -> None:
    source = _read_source("pyimgano/utils/datasets.py")

    assert 'for ext in ["*.png", "*.jpg", "*.bmp"]' not in source
    assert 'for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]' not in source


def test_services_package_avoids_redundant_keys_call_for_dunder_all() -> None:
    source = _read_source("pyimgano/services/__init__.py")

    assert "tuple(_SERVICE_EXPORT_SOURCES.keys())" not in source
    assert "tuple(_SERVICE_EXPORT_SOURCES)" not in source


def test_clone_reference_repos_usage_function_returns_explicit_success() -> None:
    source = _read_source("tools/clone_reference_repos.sh")
    match = re.search(r"usage\(\) \{\r?\n(?P<body>.*?)\r?\n\}", source, re.DOTALL)

    assert match is not None
    assert re.search(r"^\s*return 0\s*$", match.group("body"), re.MULTILINE) is not None
