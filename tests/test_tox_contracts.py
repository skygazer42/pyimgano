from __future__ import annotations

from pathlib import Path


def _read_repo_file(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / relative_path).read_text(encoding="utf-8")


def _get_section_lines(text: str, header: str) -> list[str]:
    in_section = False
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped == header:
            in_section = True
            continue
        if in_section and stripped.startswith("[") and stripped.endswith("]"):
            break
        if in_section:
            lines.append(raw.rstrip())
    return lines


def _get_multiline_values(section_lines: list[str], key: str) -> list[str]:
    values: list[str] = []
    capture = False
    for raw in section_lines:
        stripped = raw.strip()
        if not capture:
            if stripped == f"{key} =":
                capture = True
            continue

        if stripped and not raw.startswith((" ", "\t")):
            break
        if stripped:
            values.append(stripped)
    return values


def test_default_tox_test_env_includes_skimage_for_classical_model_suite() -> None:
    tox_text = _read_repo_file("tox.ini")
    testenv_lines = _get_section_lines(tox_text, "[testenv:py{39,310,311,312}]")

    extras = _get_multiline_values(testenv_lines, "extras")

    assert "dev" in extras
    assert "skimage" in extras


def test_tox_build_env_checks_only_fresh_artifacts() -> None:
    tox_text = _read_repo_file("tox.ini")
    build_lines = _get_section_lines(tox_text, "[testenv:build]")

    commands = _get_multiline_values(build_lines, "commands")

    assert "python -m build --outdir {envtmpdir}/dist" in commands
    assert "twine check {envtmpdir}/dist/*" in commands
