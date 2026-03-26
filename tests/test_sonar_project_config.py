from __future__ import annotations

from pathlib import Path


def _read_properties() -> dict[str, str]:
    props: dict[str, str] = {}
    for line in Path("sonar-project.properties").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        props[key.strip()] = value.strip()
    return props


def test_sonar_project_focuses_on_runtime_code_only() -> None:
    props = _read_properties()

    assert props["sonar.sources"] == "pyimgano,tools"
    exclusions = {part.strip() for part in props["sonar.exclusions"].split(",") if part.strip()}
    assert "tests/**" in exclusions
    assert "docs/**" in exclusions
    assert "examples/**" in exclusions
    assert ".github/**" in exclusions
    assert "sonar.tests" not in props
    assert "sonar.test.inclusions" not in props


def test_sonar_project_excludes_non_runtime_paths_from_duplication() -> None:
    props = _read_properties()

    exclusions = {part.strip() for part in props["sonar.cpd.exclusions"].split(",") if part.strip()}
    assert "tests/**" in exclusions
    assert "docs/**" in exclusions
    assert "examples/**" in exclusions
    assert ".github/**" in exclusions


def test_sonar_project_ignores_test_and_example_issues() -> None:
    props = _read_properties()

    keys = props["sonar.issue.ignore.multicriteria"].split(",")
    expected = {
        ("tests_float_eq", "python:S1244", "tests/**/*"),
        ("tests_name_style", "python:S117", "tests/**/*"),
        ("tests_rng", "python:S5754", "tests/**/*"),
        ("tests_param_count", "python:S107", "tests/**/*"),
        ("tests_path_taint", "pythonsecurity:S2083", "tests/**/*"),
        ("tests_same_branch", "python:S3923", "tests/**/*"),
        ("tests_regex_complexity", "python:S1542", "tests/**/*"),
        ("tests_runtime_bug", "pythonbugs:S6466", "tests/**/*"),
        ("examples_complexity", "python:S3776", "examples/**/*"),
        ("examples_comment_code", "python:S125", "examples/**/*"),
        ("examples_literals", "python:S1192", "examples/**/*"),
        ("examples_unused", "python:S1481", "examples/**/*"),
    }
    seen = {
        (
            key,
            props[f"sonar.issue.ignore.multicriteria.{key}.ruleKey"],
            props[f"sonar.issue.ignore.multicriteria.{key}.resourceKey"],
        )
        for key in keys
    }
    assert expected.issubset(seen)
