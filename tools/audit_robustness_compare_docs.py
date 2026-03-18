from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditRule:
    path: str
    required: tuple[str, ...]


DEFAULT_RULES = (
    AuditRule(
        path="docs/CLI_REFERENCE.md",
        required=(
            "--same-environment-as",
            "--same-target-as",
            "--same-robustness-protocol-as",
            "--require-same-robustness-protocol",
            "robustness_protocol_comparison",
            "trust_comparison",
            "candidate_blocking_reasons",
            "candidate_comparability_gates",
            "candidate_verdict.",
            "candidate_blocking_reasons.",
            "candidate_comparability_gates.",
            "comparison_trust_reason",
            "comparison_trust_ref.",
        ),
    ),
    AuditRule(
        path="docs/RUN_COMPARISON.md",
        required=(
            "--same-environment-as",
            "--same-target-as",
            "--same-robustness-protocol-as",
            "--require-same-robustness-protocol",
            "robustness_protocol_comparison",
            "trust_comparison",
            "candidate_blocking_reasons",
            "candidate_comparability_gates",
            "candidate_verdict.",
            "candidate_blocking_reasons.",
            "candidate_comparability_gates.",
            "comparison_trust_reason",
            "comparison_trust_ref.",
        ),
    ),
    AuditRule(
        path="docs/ROBUSTNESS_BENCHMARK.md",
        required=(
            "--same-robustness-protocol-as",
            "--require-same-robustness-protocol",
            "robustness_protocol_comparison",
        ),
    ),
)


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        argv = list(sys.argv[1:])

    normalized: list[str] = []
    i = 0
    while i < len(argv):
        item = str(argv[i])
        if item == "--require" and i + 1 < len(argv):
            normalized.append(f"--require={argv[i + 1]}")
            i += 2
            continue
        normalized.append(item)
        i += 1
    return normalized


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
        prog="audit_robustness_compare_docs",
        description=(
            "Fail when robustness comparison docs stop mentioning the required CLI gate "
            "and JSON comparison payload."
        ),
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
    args = parser.parse_args(_normalize_argv(argv))

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

    print("OK: robustness comparison docs mention the required CLI and JSON contract.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
