from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-validate-infer-config",
        description="Validate a pyimgano workbench infer-config.json for deploy-style inference.",
    )
    parser.add_argument(
        "infer_config",
        help="Path to infer_config.json (exported by pyimgano-train --export-infer-config)",
    )
    parser.add_argument(
        "--infer-category",
        default=None,
        help="Optional category selection when infer-config contains multiple categories",
    )
    parser.add_argument(
        "--no-check-files",
        action="store_true",
        help="Skip file existence checks (e.g. checkpoints) for portability",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the normalized infer-config payload as JSON to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        from pyimgano.inference.validate_infer_config import validate_infer_config_file

        validation = validate_infer_config_file(
            Path(str(args.infer_config)),
            category=(str(args.infer_category) if args.infer_category is not None else None),
            check_files=(not bool(args.no_check_files)),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(str(exc), file=sys.stderr)
        return 1

    for w in validation.warnings:
        print(f"warning: {w}", file=sys.stderr)

    if bool(args.json):
        print(json.dumps(validation.payload, indent=2, sort_keys=True))
        return 0

    model = validation.payload.get("model", {}) or {}
    model_name = model.get("name", None)
    category = validation.payload.get("category", None)
    ckpt = validation.resolved_checkpoint_path
    msg = "ok"
    if model_name is not None:
        msg += f": model={model_name}"
    if category is not None:
        msg += f" category={category}"
    if ckpt is not None:
        msg += f" checkpoint={ckpt}"
    print(msg)
    return 0

