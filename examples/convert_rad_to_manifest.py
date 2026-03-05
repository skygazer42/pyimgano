from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a RAD-like dataset tree to a JSONL manifest (paths-first)."
    )
    parser.add_argument("--root", required=True, help="RAD dataset root directory")
    parser.add_argument("--out", required=True, help="Output manifest.jsonl path")
    parser.add_argument(
        "--category", default="rad", help="Category label to stamp into the manifest"
    )
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute paths")
    parser.add_argument(
        "--include-masks", action="store_true", help="Include mask_path when available"
    )
    args = parser.parse_args(argv)

    from pyimgano.datasets.rad import convert_rad_to_manifest

    records = convert_rad_to_manifest(
        root=Path(str(args.root)),
        out_path=Path(str(args.out)),
        category=str(args.category),
        absolute_paths=bool(args.absolute_paths),
        include_masks=bool(args.include_masks),
    )
    print(f"wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
