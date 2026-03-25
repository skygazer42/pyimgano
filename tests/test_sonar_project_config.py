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

    exclusions = {
        part.strip() for part in props["sonar.cpd.exclusions"].split(",") if part.strip()
    }
    assert "tests/**" in exclusions
    assert "docs/**" in exclusions
    assert "examples/**" in exclusions
    assert ".github/**" in exclusions


def test_sonar_project_ignores_test_and_example_issues() -> None:
    props = _read_properties()

    assert props["sonar.issue.ignore.multicriteria"] == "tests_all,examples_all"
    assert props["sonar.issue.ignore.multicriteria.tests_all.ruleKey"] == "*"
    assert props["sonar.issue.ignore.multicriteria.tests_all.resourceKey"] == "tests/**/*"
    assert props["sonar.issue.ignore.multicriteria.examples_all.ruleKey"] == "*"
    assert props["sonar.issue.ignore.multicriteria.examples_all.resourceKey"] == "examples/**/*"
