"""
End-to-end pixel-first industrial MVP loop (SSIM template-map → defects export).

This script is intentionally dependency-light and runs fully offline:
- synthesizes a tiny dataset + manifest from normal images
- fits a pixel-map baseline (`ssim_template_map`)
- exports industrial defects artifacts (mask + regions + overlays)

Usage:
  python examples/pixel_first_ssim_defects.py --in-dir /path/to/normal_images --out-root ./out_demo

If --in-dir is omitted, the script generates a small set of deterministic
pattern images under <out-root>/input_normals/ for a self-contained demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def _write_pattern_png(path: Path, *, seed: int) -> None:
    """Write a small deterministic RGB image with some structure (SSIM-friendly)."""

    h, w = 128, 128

    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, num=h, dtype=np.float32),
        np.linspace(0.0, 1.0, num=w, dtype=np.float32),
        indexing="ij",
    )
    base = (
        0.55
        + 0.18 * np.sin(2.0 * np.pi * (3.0 * xx + 1.5 * yy))
        + 0.12 * np.cos(2.0 * np.pi * (2.0 * yy))
    ).astype(np.float32)
    shift = (float(int(seed) % 7) - 3.0) * 0.01
    img = np.clip(base + shift, 0.0, 1.0)

    rgb = np.stack(
        [
            img,
            np.clip(img * 0.9 + 0.05, 0.0, 1.0),
            np.clip(np.roll(img, shift=2, axis=1) * 0.95 + 0.02, 0.0, 1.0),
        ],
        axis=-1,
    )
    arr_u8 = (rgb * 255.0).round().astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode="RGB").save(path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pixel-first-ssim-defects")
    p.add_argument(
        "--in-dir",
        default=None,
        help=(
            "Directory of normal images for synthesis. "
            "If omitted, generates a tiny deterministic set under <out-root>/input_normals."
        ),
    )
    p.add_argument("--out-root", default="./out_pixel_first_ssim_defects", help="Output root dir")
    p.add_argument("--category", default="demo", help="Manifest category name")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for synthesis")
    p.add_argument("--n-train", type=int, default=20, help="Number of training normals to synthesize")
    p.add_argument("--n-test-normal", type=int, default=10, help="Number of normal test images to synthesize")
    p.add_argument("--n-test-anomaly", type=int, default=10, help="Number of anomalous test images to synthesize")
    p.add_argument(
        "--pixel-normal-quantile",
        type=float,
        default=0.95,
        help=(
            "Pixel threshold calibration quantile for defects (default: 0.95). "
            "For real datasets you may want 0.999+; for tiny demos 0.95 is more visible."
        ),
    )
    p.add_argument(
        "--defects-preset",
        default="industrial-defects-fp40",
        help="Defects preset name (default: industrial-defects-fp40)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Prepare a tiny "normal image folder".
    if args.in_dir is None:
        in_dir = out_root / "input_normals"
        for i in range(30):
            _write_pattern_png(in_dir / f"{i:04d}.png", seed=1337 + i)
    else:
        in_dir = Path(args.in_dir).expanduser().resolve()
        if not in_dir.exists():
            raise FileNotFoundError(f"--in-dir not found: {in_dir}")

    # 2) Synthesize dataset + manifest (manifest is the stable interchange format).
    manifest_path = out_root / "manifest.jsonl"
    from pyimgano.synthesize_cli import synthesize_dataset

    synthesize_dataset(
        in_dir=in_dir,
        out_root=out_root,
        category=str(args.category),
        preset="tape",
        blend="alpha",
        alpha=0.9,
        seed=int(args.seed),
        n_train=int(args.n_train),
        n_test_normal=int(args.n_test_normal),
        n_test_anomaly=int(args.n_test_anomaly),
        manifest_path=manifest_path,
        absolute_paths=True,
    )

    train_dir = out_root / "train" / "normal"
    test_dir = out_root / "test"

    # 3) Infer with a pixel-map baseline and export defects (mask + regions + overlays).
    from pyimgano.infer_cli import main as infer_main

    out_jsonl = out_root / "infer_out.jsonl"
    masks_dir = out_root / "defects_masks"
    overlays_dir = out_root / "defects_overlays"
    regions_jsonl = out_root / "defects_regions.jsonl"

    rc = infer_main(
        [
            "--model",
            "ssim_template_map",
            "--train-dir",
            str(train_dir),
            "--input",
            str(test_dir),
            "--defects-preset",
            str(args.defects_preset),
            "--pixel-threshold-strategy",
            "normal_pixel_quantile",
            "--pixel-normal-quantile",
            str(float(args.pixel_normal_quantile)),
            "--defects-regions-jsonl",
            str(regions_jsonl),
            "--save-masks",
            str(masks_dir),
            "--save-overlays",
            str(overlays_dir),
            "--save-jsonl",
            str(out_jsonl),
        ]
    )

    print("\nDone.")
    print(f"- manifest:        {manifest_path}")
    print(f"- infer JSONL:     {out_jsonl}")
    print(f"- defects regions: {regions_jsonl}")
    print(f"- masks dir:       {masks_dir}")
    print(f"- overlays dir:    {overlays_dir}")
    return int(rc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

