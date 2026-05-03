from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
RC_VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)rc(\d+)$")


def _extract_version(path: Path, pattern: re.Pattern[str], label: str) -> tuple[str | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, f"{path}: missing {label} file"

    match = pattern.search(text)
    if match is None:
        return None, f"{path}: missing {label} version"
    return match.group(1), None


def _tag_candidates_for_version(version: str) -> set[str]:
    candidates = {f"v{version}"}
    rc_match = RC_VERSION_RE.match(version)
    if rc_match is not None:
        base, rc_number = rc_match.groups()
        candidates.add(f"v{base}-rc{rc_number}")
    return candidates


def _validate_versions(
    *,
    pyproject: Path,
    init_file: Path,
    tag: str | None,
) -> list[str]:
    issues: list[str] = []

    project_version, issue = _extract_version(pyproject, PYPROJECT_VERSION_RE, "pyproject")
    if issue is not None:
        issues.append(issue)

    init_version, issue = _extract_version(init_file, INIT_VERSION_RE, "package")
    if issue is not None:
        issues.append(issue)

    if project_version is None or init_version is None:
        return issues

    if project_version != init_version:
        issues.append(
            f"package version mismatch: pyproject has {project_version!r}, "
            f"pyimgano.__version__ has {init_version!r}"
        )

    if tag:
        expected_tags = _tag_candidates_for_version(project_version)
        if tag not in expected_tags:
            expected = " or ".join(sorted(repr(candidate) for candidate in expected_tags))
            issues.append(
                f"release tag {tag!r} does not match project version {project_version!r}; "
                f"expected {expected}"
            )

    return issues


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_release_version",
        description=(
            "Fail when package version metadata and the release tag are not aligned before upload."
        ),
    )
    parser.add_argument("--pyproject", default="pyproject.toml", help="Path to pyproject.toml.")
    parser.add_argument(
        "--init",
        default="pyimgano/__init__.py",
        help="Path to the package __init__.py exposing __version__.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag to validate, for example v0.9.1 or v0.9.1-rc1.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    issues = _validate_versions(
        pyproject=Path(str(args.pyproject)),
        init_file=Path(str(args.init)),
        tag=str(args.tag) if args.tag else None,
    )
    if issues:
        for issue in issues:
            print(issue)
        return 1

    project_version, _ = _extract_version(
        Path(str(args.pyproject)), PYPROJECT_VERSION_RE, "pyproject"
    )
    tag_suffix = f" and release tag {args.tag!r}" if args.tag else ""
    print(f"OK: package version {project_version!r}{tag_suffix} are aligned.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
