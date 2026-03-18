from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_PATHS = (
    "README.md",
    "CONTRIBUTING.md",
    "benchmarks",
    "docs",
)

TEXT_SUFFIXES = {
    ".md",
    ".rst",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
}


@dataclass(frozen=True)
class AuditPattern:
    name: str
    needle: str
    guidance: str


PATTERNS = (
    AuditPattern(
        name="legacy_repo_slug",
        needle="jhlu2019/pyimgano",
        guidance="replace with skygazer42/pyimgano",
    ),
    AuditPattern(
        name="placeholder_contact",
        needle="pyimgano@example.com",
        guidance="replace with a real support channel or remove the placeholder",
    ),
)


def _iter_candidate_files(paths: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for raw in paths:
        path = Path(str(raw))
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
                continue
            yield candidate


def _scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    issues: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern in PATTERNS:
            if pattern.needle in line:
                issues.append(
                    f"{path}:{lineno}: {pattern.name}: found {pattern.needle!r} ({pattern.guidance})"
                )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_repo_links",
        description="Fail if docs still reference legacy repo slugs or placeholder contacts.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional files/directories to scan. Defaults to core docs and benchmark paths.",
    )
    args = parser.parse_args(argv)

    scan_paths = list(args.paths) if args.paths else list(DEFAULT_PATHS)
    issues: list[str] = []
    for path in _iter_candidate_files(scan_paths):
        issues.extend(_scan_file(path))

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: repository links and contacts look clean.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
