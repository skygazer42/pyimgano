from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_INDUSTRIAL_FASTPATH_DOC = "docs/INDUSTRIAL_FASTPATH.md"
_AUDITED_CONFIG_JSON = "industrial_adapt_audited.json"


@dataclass(frozen=True)
class AuditRule:
    path: str
    required: tuple[str, ...]


DEFAULT_RULES = (
    AuditRule(
        path="README.md",
        required=(
            _INDUSTRIAL_FASTPATH_DOC,
            _AUDITED_CONFIG_JSON,
        ),
    ),
    AuditRule(
        path="docs/QUICKSTART.md",
        required=(
            _INDUSTRIAL_FASTPATH_DOC,
            _AUDITED_CONFIG_JSON,
        ),
    ),
    AuditRule(
        path=_INDUSTRIAL_FASTPATH_DOC,
        required=(
            "infer_config.json",
            "calibration_card.json",
            "bundle_manifest.json",
            _AUDITED_CONFIG_JSON,
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
            issues.append(f"{path}: missing required reference {needle!r}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_audited_fastpath_docs",
        description="Fail when audited fast-path docs/examples stop mentioning the required artifact set.",
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

    print("OK: audited fast-path docs mention the expected artifact set.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
