"""Unified discovery shortcut CLI for pyimgano."""

from __future__ import annotations

import argparse

import pyimgano.pyim_app as pyim_app
import pyimgano.pyim_cli_options as pyim_cli_options


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyim",
        description="Unified discovery shortcut for models, families, presets, and preprocessing.",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const="all",
        choices=pyim_cli_options.PYIM_LIST_KIND_CHOICES,
        help=(
            "List available items. Default with no value: all. "
            "Examples: --list, --list models, --list preprocessing"
        ),
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Optional tags filter for model/feature discovery (comma-separated or repeatable). "
            "Example: --tags vision,deep"
        ),
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Optional model family/tag filter used with --list models. Example: --family patchcore",
    )
    parser.add_argument(
        "--type",
        dest="algorithm_type",
        default=None,
        help="Optional high-level model type/tag filter used with --list models. Example: --type deep-vision",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Optional publication year filter used with --list models. Example: --year 2021",
    )
    parser.add_argument(
        "--goal",
        default=None,
        choices=["first-run", "cpu-screening", "pixel-localization", "deployable"],
        help="Task-oriented recommendation goal. Emits models, recipes, and datasets together.",
    )
    parser.add_argument(
        "--objective",
        default=None,
        choices=["balanced", "latency", "localization"],
        help="Optional starter-pick objective used with --list models.",
    )
    parser.add_argument(
        "--selection-profile",
        default=None,
        choices=["balanced", "benchmark-parity", "cpu-screening", "deploy-readiness"],
        help="Optional starter-pick profile used with --list models.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Maximum number of starter picks to emit with --list models.",
    )
    parser.add_argument(
        "--deployable-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When listing preprocessing schemes, include only deployable infer/workbench presets.",
    )
    parser.add_argument(
        "--audit-metadata",
        action="store_true",
        help="Audit registry models against the metadata contract and exit.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list is None and not bool(args.audit_metadata) and args.goal is None:
        parser.print_help()
        return 0

    try:
        return pyim_app.run_pyim_command(
            pyim_app.PyimCommand(
                list_kind=args.list,
                tags=args.tags,
                family=args.family,
                algorithm_type=args.algorithm_type,
                year=args.year,
                deployable_only=bool(args.deployable_only),
                goal=args.goal,
                objective=args.objective,
                selection_profile=args.selection_profile,
                topk=args.topk,
                audit_metadata=bool(args.audit_metadata),
                json_output=bool(args.json),
            )
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
