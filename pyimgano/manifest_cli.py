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
    mode.add_argument(
        "--stats",
        action="store_true",
        help="Print basic statistics for a JSONL manifest (requires --manifest)",
    )
    mode.add_argument(
        "--filter",
        action="store_true",
        help="Filter a JSONL manifest and write a new JSONL (requires --manifest and --out)",
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
        help="Manifest JSONL path to validate/stats/filter (required when --validate/--stats/--filter).",
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
        help="Print a JSON report to stdout (validation/stats).",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "val", "test"],
        help="Filter: required split value (filter only).",
    )
    parser.add_argument(
        "--label",
        default=None,
        choices=["0", "1"],
        help="Filter: required label value (0 normal, 1 anomaly; filter only).",
    )
    parser.add_argument(
        "--has-mask",
        default=None,
        choices=["true", "false"],
        help="Filter: require presence/absence of mask_path field (filter only).",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="Filter: keep only the first N matching records (filter only).",
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

        if bool(getattr(args, "stats", False)):
            if args.manifest is None:
                raise ValueError("--manifest is required when --stats is enabled")

            from pyimgano.datasets.manifest_tools import manifest_stats

            payload = manifest_stats(
                manifest_path=str(args.manifest),
                root_fallback=(str(args.root) if args.root is not None else None),
            )

            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"total_records: {payload.get('total_records')}")
                print("category_counts:")
                for k, v in sorted((payload.get("category_counts") or {}).items()):
                    print(f"  - {k}: {v}")
                print("split_counts:")
                for k, v in sorted((payload.get("split_counts") or {}).items()):
                    print(f"  - {k}: {v}")
                print("label_counts:")
                for k, v in sorted((payload.get("label_counts") or {}).items()):
                    print(f"  - {k}: {v}")
                print(
                    "mask_path_present: "
                    f"{payload.get('mask_path_present_count')} "
                    f"({payload.get('mask_path_present_ratio'):.3f})"
                )
                print(
                    "image_path_absolute: "
                    f"{payload.get('image_path_absolute_count')} "
                    f"({payload.get('image_path_absolute_ratio'):.3f})"
                )
                file_checks = payload.get("file_checks", None)
                if isinstance(file_checks, dict) and bool(file_checks.get("enabled", False)):
                    print("file_checks:")
                    print(f"  root_fallback: {file_checks.get('root_fallback')}")
                    print(
                        "  image_exists: "
                        f"{file_checks.get('image_exists_count')} "
                        f"({file_checks.get('image_exists_ratio'):.3f})"
                    )
                    print(
                        "  mask_exists: "
                        f"{file_checks.get('mask_exists_count')} "
                        f"({file_checks.get('mask_exists_ratio_of_present'):.3f} of present)"
                    )

            return 0

        if bool(getattr(args, "filter", False)):
            if args.manifest is None:
                raise ValueError("--manifest is required when --filter is enabled")
            if args.out is None:
                raise ValueError("--out is required when --filter is enabled")

            from pyimgano.datasets.manifest_tools import (
                filter_manifest_records,
                write_manifest_records,
            )

            label = None if args.label is None else int(args.label)
            has_mask = None if args.has_mask is None else (str(args.has_mask) == "true")
            limit = None if args.limit is None else int(args.limit)

            records = filter_manifest_records(
                manifest_path=str(args.manifest),
                category=(str(args.category) if args.category is not None else None),
                split=(str(args.split) if args.split is not None else None),
                label=label,
                has_mask=has_mask,
                limit=limit,
            )
            write_manifest_records(records=records, out_path=str(args.out))
            return 0

        # Conversion mode.
        if args.root is None:
            raise ValueError(
                "--root is required for conversion (omit only with --validate/--stats/--filter)"
            )
        if args.out is None:
            raise ValueError(
                "--out is required for conversion (omit only with --validate/--stats/--filter)"
            )

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
