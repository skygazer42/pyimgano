from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-weights",
        description=(
            "Utilities for managing local model weights/checkpoints (manifest validation + hashing). "
            "This tool never downloads weights."
        ),
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hash = sub.add_parser("hash", help="Compute file hash (default: sha256)")
    p_hash.add_argument("path", help="Path to file")
    p_hash.add_argument(
        "--algorithm",
        default="sha256",
        choices=["md5", "sha1", "sha256", "sha512"],
        help="Hash algorithm. Default: sha256",
    )

    p_validate = sub.add_parser("validate", help="Validate a weights manifest JSON file")
    p_validate.add_argument("manifest", help="Path to weights manifest JSON")
    p_validate.add_argument(
        "--base-dir",
        default=None,
        help=(
            "Base directory used to resolve relative entry paths. "
            "Defaults to manifest parent directory."
        ),
    )
    p_validate.add_argument(
        "--check-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check that weight files exist on disk. Default: true",
    )
    p_validate.add_argument(
        "--check-hashes",
        action="store_true",
        default=False,
        help="Verify sha256 for entries that provide it. Default: false",
    )
    p_validate.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON report instead of printing human-readable errors/warnings.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if str(args.cmd) == "hash":
            from pyimgano.utils.security import FileHasher

            p = Path(str(args.path))
            if not p.exists():
                raise FileNotFoundError(f"File not found: {p}")
            digest = FileHasher.compute_hash(str(p), algorithm=str(args.algorithm))
            print(digest)
            return 0

        if str(args.cmd) == "validate":
            from pyimgano.weights.manifest import validate_weights_manifest_file

            report = validate_weights_manifest_file(
                manifest_path=str(args.manifest),
                base_dir=(str(args.base_dir) if args.base_dir is not None else None),
                check_files=bool(args.check_files),
                check_hashes=bool(args.check_hashes),
            )
            if bool(args.json):
                print(json.dumps(report.to_jsonable(), indent=2, sort_keys=True))
            else:
                for w in report.warnings:
                    print(f"warning: {w}")
                for e in report.errors:
                    print(f"error: {e}")
            return 0 if report.ok else 1

        raise RuntimeError(f"Unhandled cmd: {args.cmd!r}")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
