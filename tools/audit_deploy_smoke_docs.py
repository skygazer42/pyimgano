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
        path="docs/START_HERE.md",
        required=(
            "pyimgano-doctor --profile deploy-smoke --json",
            "deploy_smoke_custom_cpu.json",
            "pyimgano bundle validate",
        ),
    ),
    AuditRule(
        path="docs/STARTER_PATHS.md",
        required=(
            "pyimgano-doctor --profile deploy-smoke --json",
            "deploy_smoke_custom_cpu.json",
            "pyimgano bundle validate",
        ),
    ),
    AuditRule(
        path="docs/QUICKSTART.md",
        required=(
            "deploy_smoke_custom_cpu.json",
            "manifest_industrial_workflow_balanced.json",
            "handoff_report.json",
            "pyimgano bundle validate",
            "pyimgano runs acceptance",
        ),
    ),
    AuditRule(
        path="docs/MANIFEST_DATASET.md",
        required=(
            "manifest_industrial_workflow_balanced.json",
            "handoff_report.json",
            "pyimgano bundle validate",
            "pyimgano runs acceptance",
        ),
    ),
    AuditRule(
        path="docs/ALGORITHM_SELECTION_GUIDE.md",
        required=(
            "pyim --goal deployable --json",
            "deploy_smoke_custom_cpu.json",
        ),
    ),
    AuditRule(
        path="docs/INDUSTRIAL_FASTPATH.md",
        required=(
            "handoff_report.json",
            "bundle_manifest.json",
            "pyimgano-runs acceptance",
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
            issues.append(f"{path}: missing required deploy-smoke reference {needle!r}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_deploy_smoke_docs",
        description="Fail when deploy-smoke docs lose required config, bundle, or acceptance references.",
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

    print("OK: deploy-smoke docs mention the expected configs, bundle artifacts, and gate commands.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
