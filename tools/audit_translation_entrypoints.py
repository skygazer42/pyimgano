from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_FILES = (
    "README_cn.md",
    "README_ja.md",
    "README_ko.md",
)

REQUIRED_ENTRYPOINTS: tuple[str, ...] = (
    "pyimgano-doctor",
    "pyimgano-demo --smoke",
    "--list-starter-configs",
)


def _scan_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    issues: list[str] = []
    for needle in REQUIRED_ENTRYPOINTS:
        if needle not in text:
            issues.append(f"{path}: missing translation entrypoint: {needle}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_translation_entrypoints",
        description="Fail when translated READMEs drift away from core onboarding entrypoints.",
    )
    parser.add_argument("paths", nargs="*", help="Optional files to scan.")
    args = parser.parse_args(argv)

    scan_paths = [Path(item) for item in (args.paths or DEFAULT_FILES)]
    issues: list[str] = []
    for path in scan_paths:
        if not path.is_file():
            issues.append(f"{path}: missing translation file")
            continue
        issues.extend(_scan_file(path))

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: translated READMEs contain the required onboarding entrypoints.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
