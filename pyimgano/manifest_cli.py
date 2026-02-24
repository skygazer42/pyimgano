from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return out


def _relpath(path: Path, *, base: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        # Fall back to a best-effort relative path.
        return str(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-manifest")

    parser.add_argument("--root", required=True, help="Dataset root directory (custom layout)")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument(
        "--category", default="custom", help="Category name to stamp into the manifest"
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths instead of paths relative to the manifest output directory",
    )
    parser.add_argument(
        "--include-masks",
        action="store_true",
        help="Include mask_path entries when ground_truth/anomaly masks exist",
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
    """Generate a JSONL manifest from the built-in `custom` dataset layout.

    Expected directory structure (same as `CustomDataset`):

    root/
      train/normal/*.png|*.jpg|*.jpeg|*.bmp
      test/normal/*.png|*.jpg|*.jpeg|*.bmp
      test/anomaly/*.png|*.jpg|*.jpeg|*.bmp
      ground_truth/anomaly/<stem>_mask.png   (optional)
    """

    root_path = Path(root)
    out_file = Path(out_path)
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cat = str(category).strip() or "custom"

    train_dir = root_path / "train" / "normal"
    test_normal_dir = root_path / "test" / "normal"
    test_anomaly_dir = root_path / "test" / "anomaly"
    gt_anomaly_dir = root_path / "ground_truth" / "anomaly"

    records: list[dict[str, Any]] = []

    for p in _iter_images(train_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "train",
            }
        )

    for p in _iter_images(test_normal_dir):
        records.append(
            {
                "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
                "category": cat,
                "split": "test",
                "label": 0,
            }
        )

    for p in _iter_images(test_anomaly_dir):
        rec: dict[str, Any] = {
            "image_path": _relpath(p, base=out_dir, absolute=absolute_paths),
            "category": cat,
            "split": "test",
            "label": 1,
        }
        if include_masks:
            mask_candidate = gt_anomaly_dir / f"{p.stem}_mask.png"
            if mask_candidate.exists():
                rec["mask_path"] = _relpath(mask_candidate, base=out_dir, absolute=absolute_paths)
        records.append(rec)

    # Stable output order for reproducibility.
    records.sort(key=lambda r: (str(r.get("split", "")), str(r.get("image_path", ""))))

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        generate_manifest_from_custom_layout(
            root=str(args.root),
            out_path=str(args.out),
            category=str(args.category),
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
