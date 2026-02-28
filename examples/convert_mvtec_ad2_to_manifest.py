from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert MVTec AD 2 to a JSONL manifest (paths-first).")
    parser.add_argument("--root", required=True, help="MVTec AD 2 root directory")
    parser.add_argument("--category", required=True, help="Category name (e.g. bottle)")
    parser.add_argument("--out", required=True, help="Output manifest.jsonl path")
    parser.add_argument("--split", default="test_public", help="Split folder name (default: test_public)")
    parser.add_argument("--absolute-paths", action="store_true", help="Write absolute paths")
    parser.add_argument("--include-masks", action="store_true", help="Include mask_path when available")
    args = parser.parse_args(argv)

    from pyimgano.datasets.mvtec_ad2 import convert_mvtec_ad2_to_manifest

    records = convert_mvtec_ad2_to_manifest(
        root=Path(str(args.root)),
        category=str(args.category),
        out_path=Path(str(args.out)),
        split=str(args.split),
        absolute_paths=bool(args.absolute_paths),
        include_masks=bool(args.include_masks),
    )
    print(f"wrote {len(records)} records -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

