from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-manifest")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing JSONL manifest (required keys + optional file existence checks)",
    )

    parser.add_argument(
        "--dataset",
        default="custom",
        help="Dataset converter name (default: custom). Use `pyimgano-datasets list` to discover.",
    )

    parser.add_argument(
        "--root",
        required=False,
        help="Dataset root directory (for conversion) or root fallback (for validation).",
    )
    parser.add_argument("--out", required=False, help="Output JSONL path (for conversion).")
    parser.add_argument(
        "--manifest",
        required=False,
        help="Manifest JSONL path to validate (required when --validate).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help=(
            "Category name to stamp into the manifest (conversion) or to validate (validation). "
            "When omitted in conversion, a dataset-specific default is used."
        ),
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of relative paths (conversion only).",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Include mask_path entries when ground-truth masks exist (conversion only).",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip file existence checks when validating a manifest (validation only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON validation report to stdout (validation only).",
    )

    return parser


def generate_manifest_from_custom_layout(
    *,
    root: str | Path,
    out_path: str | Path,
    category: str = "custom",
    absolute_paths: bool = False,
    include_masks: bool = False,
) -> list[dict[str, Any]]:
    """Backwards-compatible helper for the built-in `custom` layout."""

    from pyimgano.datasets.converters import convert_custom_layout_to_manifest

    return convert_custom_layout_to_manifest(
        root=root,
        out_path=out_path,
        category=category,
        absolute_paths=absolute_paths,
        include_masks=include_masks,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if bool(args.validate):
            if args.manifest is None:
                raise ValueError("--manifest is required when --validate is enabled")

            from pyimgano.datasets.manifest_validate import validate_manifest_file

            report = validate_manifest_file(
                manifest_path=str(args.manifest),
                root_fallback=(str(args.root) if args.root is not None else None),
                check_files=(not bool(args.no_check_files)),
                category=(str(args.category) if args.category is not None else None),
            )

            if bool(args.json):
                print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
            else:
                for w in report.warnings:
                    print(f"warning: {w}")
                if report.errors:
                    for e in report.errors:
                        print(f"error: {e}")

            return 0 if report.ok else 1

        # Conversion mode.
        if args.root is None:
            raise ValueError("--root is required for conversion (omit only with --validate)")
        if args.out is None:
            raise ValueError("--out is required for conversion (omit only with --validate)")

        from pyimgano.datasets.converters import convert_dataset_to_manifest, get_dataset_converter

        # Validate converter name early (helpful error).
        _ = get_dataset_converter(str(args.dataset))

        convert_dataset_to_manifest(
            dataset=str(args.dataset),
            root=str(args.root),
            out_path=str(args.out),
            category=(str(args.category) if args.category is not None else None),
            absolute_paths=bool(args.absolute_paths),
            include_masks=bool(args.include_masks),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

