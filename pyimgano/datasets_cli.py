from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-datasets")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List available dataset→manifest converters")
    p_list.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_info = sub.add_parser("info", help="Show details for one converter")
    p_info.add_argument("name", help="Converter name (e.g. custom, mvtec_ad2)")
    p_info.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_detect = sub.add_parser("detect", help="Best-effort detect a dataset layout or manifest")
    p_detect.add_argument("path", help="Dataset root directory or manifest JSONL file")
    p_detect.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_import = sub.add_parser("import", help="Convert a dataset layout into a JSONL manifest")
    p_import.add_argument("--root", required=True, help="Dataset root directory")
    p_import.add_argument("--out", required=True, help="Output manifest JSONL path")
    p_import.add_argument(
        "--dataset",
        default="auto",
        help="Dataset converter name or 'auto' (default).",
    )
    p_import.add_argument(
        "--category",
        default=None,
        help="Optional category name for category-scoped datasets (auto-selected when unambiguous).",
    )
    p_import.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute image/mask paths into the output manifest.",
    )
    p_import.add_argument(
        "--include-masks",
        action="store_true",
        help="Include ground-truth mask references when the source layout exposes them.",
    )
    p_import.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_lint = sub.add_parser("lint", help="Validate a manifest or dataset root")
    p_lint.add_argument("target", help="Manifest JSONL path or dataset root directory")
    p_lint.add_argument(
        "--dataset",
        default="auto",
        help="Dataset converter name, 'manifest', or 'auto' (default).",
    )
    p_lint.add_argument(
        "--category",
        default=None,
        help="Optional category name for category-scoped datasets.",
    )
    p_lint.add_argument(
        "--root-fallback",
        default=None,
        help="Optional root fallback for manifest file checks.",
    )
    p_lint.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    p_profile = sub.add_parser("profile", help="Summarize dataset readiness for industrial AD")
    p_profile.add_argument("target", help="Manifest JSONL path or dataset root directory")
    p_profile.add_argument(
        "--dataset",
        default="auto",
        help="Dataset converter name, 'manifest', or 'auto' (default).",
    )
    p_profile.add_argument(
        "--category",
        default=None,
        help="Optional category name for category-scoped datasets.",
    )
    p_profile.add_argument(
        "--root-fallback",
        default=None,
        help="Optional root fallback for manifest file checks.",
    )
    p_profile.add_argument("--json", action="store_true", help="Emit JSON payload to stdout")

    return parser


def _converter_to_jsonable(conv) -> dict[str, Any]:  # noqa: ANN001 - CLI boundary
    return {
        "name": str(conv.name),
        "description": str(conv.description),
        "requires_category": bool(conv.requires_category),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from pyimgano.datasets.converters import get_dataset_converter, list_dataset_converters
        from pyimgano.datasets.inspection import (
            detect_dataset_layout,
            import_dataset_to_manifest_payload,
            lint_dataset_target,
            profile_dataset_target,
        )

        if args.cmd == "list":
            payload = [_converter_to_jsonable(c) for c in list_dataset_converters()]
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                for c in list_dataset_converters():
                    req = " (category required)" if bool(c.requires_category) else ""
                    print(f"- {c.name}{req}: {c.description}")
            return 0

        if args.cmd == "info":
            c = get_dataset_converter(str(args.name))
            payload = _converter_to_jsonable(c)
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                req = "yes" if bool(c.requires_category) else "no"
                print(f"name: {c.name}")
                print(f"requires_category: {req}")
                print(f"description: {c.description}")
            return 0

        if args.cmd == "detect":
            payload = detect_dataset_layout(str(args.path))
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"path: {payload.get('path')}")
                print(f"path_kind: {payload.get('path_kind')}")
                print(f"detected: {payload.get('detected')}")
            return 0

        if args.cmd == "import":
            payload = import_dataset_to_manifest_payload(
                root=str(args.root),
                out_path=str(args.out),
                dataset=str(args.dataset),
                category=(str(args.category) if args.category is not None else None),
                absolute_paths=bool(args.absolute_paths),
                include_masks=bool(args.include_masks),
            )
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"dataset: {payload.get('dataset')}")
                print(f"category: {payload.get('category')}")
                print(f"out_path: {payload.get('out_path')}")
                print(f"record_count: {payload.get('record_count')}")
            return 0

        if args.cmd == "lint":
            payload = lint_dataset_target(
                target=str(args.target),
                dataset=str(args.dataset),
                category=(str(args.category) if args.category is not None else None),
                root_fallback=(str(args.root_fallback) if args.root_fallback is not None else None),
            )
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0 if bool(payload.get("ok")) else 1
            print(f"target: {payload.get('target')}")
            print(f"dataset: {payload.get('dataset')}")
            print(f"ok: {payload.get('ok')}")
            return 0 if bool(payload.get("ok")) else 1

        if args.cmd == "profile":
            payload = profile_dataset_target(
                target=str(args.target),
                dataset=str(args.dataset),
                category=(str(args.category) if args.category is not None else None),
                root_fallback=(str(args.root_fallback) if args.root_fallback is not None else None),
            )
            if bool(args.json):
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0

            profile = payload.get("dataset_profile", {}) or {}
            print(f"target: {payload.get('target')}")
            print(f"dataset: {payload.get('dataset')}")
            print(f"category: {payload.get('category')}")
            print(f"total_records: {profile.get('total_records')}")
            print(f"train_count: {profile.get('train_count')}")
            print(f"test_count: {profile.get('test_count')}")
            print(f"pixel_metrics_available: {profile.get('pixel_metrics_available')}")
            return 0

        raise ValueError(f"Unknown command: {args.cmd!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
