from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditRule:
    path: str
    required: tuple[str, ...]


DEFAULT_RULES = (
    AuditRule(
        path="docs/PUBLISHING.md",
        required=(
            "python3 tools/audit_release_surface.py",
            "python3 tools/audit_release_version.py --tag vX.Y.Z",
            "python3 tools/audit_adoption_docs.py",
            "python3 tools/audit_deploy_smoke_docs.py",
            "pyimgano-doctor --profile deploy-smoke --json",
            "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
            "handoff_report.json",
        ),
    ),
)


def _scan_file_for_required_terms(path: Path, required: Iterable[str]) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [f"{path}: missing file"]

    issues: list[str] = []
    for needle in required:
        if str(needle) not in text:
            issues.append(f"{path}: missing required release checklist entry {needle!r}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_release_checklist",
        description="Fail when release-checklist docs lose required gate commands or deploy-handoff references.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional single file to scan with explicit --require terms.",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=None,
        help="Required phrase for custom single-file checks. Repeatable.",
    )
    args = parser.parse_args(argv)

    issues: list[str] = []
    if args.path is not None:
        required = [str(item) for item in (args.require or []) if str(item)]
        if not required:
            parser.error("--require must be provided when scanning a custom path.")
        issues.extend(_scan_file_for_required_terms(Path(str(args.path)), required))
    else:
        for rule in DEFAULT_RULES:
            issues.extend(_scan_file_for_required_terms(Path(rule.path), rule.required))

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: release checklist docs mention the expected gate commands and deploy handoff artifacts.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
