from __future__ import annotations

import re
from pathlib import Path

RELEASE_VERSION = "0.10.0"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def test_package_version_matches_pyproject() -> None:
    import pyimgano

    text = _read("pyproject.toml")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None
    assert pyimgano.__version__ == match.group(1)
    assert pyimgano.__version__ == RELEASE_VERSION


def test_release_version_is_present_in_each_canonical_changelog() -> None:
    assert f"## [{RELEASE_VERSION}] - 2026-08-31" in _read("CHANGELOG.md")
    assert f"## v{RELEASE_VERSION} — 2026-08-31" in _read("docs/site/changelog.md")
    assert f"PyImgAno {RELEASE_VERSION} introduces schema-v1" in _read("docs/source/changelog.rst")
    assert f'<span class="version-badge">v{RELEASE_VERSION}</span>' in _read("docs/site/index.md")


def test_public_artifact_schema_versions_are_release_locked() -> None:
    from pyimgano.artifacts import (
        ARTIFACT_POLICY_SCHEMA_VERSION,
        ARTIFACT_SCHEMA_FAMILY,
        ARTIFACT_SCHEMA_VERSION,
        EXPORT_INDEX_SCHEMA_VERSION,
    )

    assert ARTIFACT_SCHEMA_FAMILY == "pyimgano-artifact"
    assert ARTIFACT_SCHEMA_VERSION == 1
    assert ARTIFACT_POLICY_SCHEMA_VERSION == 1
    assert EXPORT_INDEX_SCHEMA_VERSION == 1


def test_release_artifact_public_imports_remain_available() -> None:
    import pyimgano.artifacts as artifacts
    import pyimgano.exporting as exporting
    import pyimgano.inference as inference

    expected = {
        artifacts: {
            "ARTIFACT_SCHEMA_VERSION",
            "bind_policy",
            "export_run",
            "import_onnx",
            "validate_artifact_manifest",
        },
        exporting: {"ArtifactFormat", "get_export_adapter", "get_export_capability"},
        inference: {
            "ArtifactRuntime",
            "LegacyArtifactWarning",
            "load_artifact",
            "load_legacy_artifact",
        },
    }
    for module, names in expected.items():
        assert names <= set(module.__all__)
        assert all(hasattr(module, name) for name in names)
