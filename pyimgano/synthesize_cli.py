from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.manifest_cli import generate_manifest_from_custom_layout
from pyimgano.synthesis.presets import get_preset_names
from pyimgano.synthesis.synthesizer import AnomalySynthesizer, SynthSpec


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out.append(p)
    return out


def _write_u8_bgr(path: Path, image_u8: np.ndarray) -> None:
    import cv2  # local import

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), np.asarray(image_u8, dtype=np.uint8))
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-synthesize")
    parser.add_argument("--in-dir", required=True, help="Directory of normal images (flat folder)")
    parser.add_argument("--out-root", required=True, help="Output dataset root directory")
    parser.add_argument("--category", default="synthetic", help="Category name in manifest")
    parser.add_argument(
        "--preset",
        default="scratch",
        choices=get_preset_names(),
        help="Synthetic anomaly preset name",
    )
    parser.add_argument(
        "--blend",
        default="alpha",
        choices=["alpha", "poisson"],
        help="Blend mode used to inject anomalies",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha for alpha blending (0..1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic output)")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test-normal", type=int, default=4)
    parser.add_argument("--n-test-anomaly", type=int, default=4)
    parser.add_argument(
        "--manifest",
        default=None,
        help="Output manifest JSONL path (default: <out-root>/manifest.jsonl)",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths in manifest instead of relative",
    )
    return parser


def synthesize_dataset(
    *,
    in_dir: str | Path,
    out_root: str | Path,
    category: str = "synthetic",
    preset: str = "scratch",
    blend: str = "alpha",
    alpha: float = 0.9,
    seed: int = 0,
    n_train: int = 8,
    n_test_normal: int = 4,
    n_test_anomaly: int = 4,
    manifest_path: str | Path | None = None,
    absolute_paths: bool = False,
) -> list[dict[str, Any]]:
    """Generate a tiny synthetic dataset in the built-in `custom` layout + manifest."""

    in_path = Path(in_dir)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    items = _iter_images(in_path)
    if not items:
        raise ValueError(f"No images found in --in-dir: {in_path}")

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(items)).tolist()
    items = [items[i] for i in perm]

    n_train = max(0, int(n_train))
    n_test_normal = max(0, int(n_test_normal))
    n_test_anomaly = max(0, int(n_test_anomaly))

    train_items = items[:n_train] if n_train else []
    test_normal_items = items[n_train : n_train + n_test_normal] if n_test_normal else []

    # Fall back: if not enough for test_normal, reuse train items.
    if not test_normal_items and (n_test_normal > 0):
        test_normal_items = list(train_items)
    if not test_normal_items and (n_test_anomaly > 0):
        test_normal_items = list(items)

    # Layout follows `CustomDataset` + `pyimgano-manifest`.
    train_dir = out_root_path / "train" / "normal"
    test_normal_dir = out_root_path / "test" / "normal"
    test_anomaly_dir = out_root_path / "test" / "anomaly"
    gt_anomaly_dir = out_root_path / "ground_truth" / "anomaly"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_normal_dir.mkdir(parents=True, exist_ok=True)
    test_anomaly_dir.mkdir(parents=True, exist_ok=True)
    gt_anomaly_dir.mkdir(parents=True, exist_ok=True)

    # Copy normal images for train/test.
    for i, p in enumerate(train_items):
        dst = train_dir / f"train_{i:05d}{p.suffix.lower()}"
        shutil.copyfile(str(p), str(dst))

    for i, p in enumerate(test_normal_items):
        dst = test_normal_dir / f"good_{i:05d}{p.suffix.lower()}"
        shutil.copyfile(str(p), str(dst))

    syn = AnomalySynthesizer(SynthSpec(preset=str(preset), probability=1.0, blend=str(blend), alpha=float(alpha)))

    # Generate anomalies by synthesizing from test-normal items (with replacement).
    for i in range(n_test_anomaly):
        base_path = test_normal_items[int(rng.integers(0, len(test_normal_items)))]

        import cv2  # local import

        base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if base is None:
            raise ValueError(f"Failed to load image: {base_path}")
        base_u8 = np.asarray(base, dtype=np.uint8)

        res = syn(base_u8, seed=(int(seed) + 1009 * i))
        out_img = res.image_u8
        out_mask = res.mask_u8

        out_name = f"bad_{i:05d}.png"
        img_out = test_anomaly_dir / out_name
        mask_out = gt_anomaly_dir / f"{Path(out_name).stem}_mask.png"

        _write_u8_bgr(img_out, out_img)
        _write_u8_bgr(mask_out, out_mask)

    if manifest_path is None:
        manifest_path = out_root_path / "manifest.jsonl"

    records = generate_manifest_from_custom_layout(
        root=out_root_path,
        out_path=manifest_path,
        category=str(category),
        absolute_paths=bool(absolute_paths),
        include_masks=True,
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        synthesize_dataset(
            in_dir=str(args.in_dir),
            out_root=str(args.out_root),
            category=str(args.category),
            preset=str(args.preset),
            blend=str(args.blend),
            alpha=float(args.alpha),
            seed=int(args.seed),
            n_train=int(args.n_train),
            n_test_normal=int(args.n_test_normal),
            n_test_anomaly=int(args.n_test_anomaly),
            manifest_path=(None if args.manifest is None else str(args.manifest)),
            absolute_paths=bool(args.absolute_paths),
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

