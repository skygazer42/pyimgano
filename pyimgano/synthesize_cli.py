from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.datasets.manifest import iter_manifest_records
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
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-dir", default=None, help="Directory of normal images (flat folder)")
    src.add_argument("--from-manifest", default=None, help="Input manifest JSONL (source normals)")
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
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate a synthesis preview grid and exit (does not write a dataset)",
    )
    parser.add_argument(
        "--preview-out",
        default=None,
        help="Optional output path for preview PNG (default: <out-root>/preview.png)",
    )
    parser.add_argument("--preview-n", type=int, default=16, help="Number of samples in preview grid")
    parser.add_argument("--preview-cols", type=int, default=4, help="Preview grid columns")
    parser.add_argument("--n-train", type=int, default=8)
    parser.add_argument("--n-test-normal", type=int, default=4)
    parser.add_argument("--n-test-anomaly", type=int, default=4)
    parser.add_argument(
        "--from-manifest-category",
        default=None,
        help="Filter source manifest by category (used with --from-manifest)",
    )
    parser.add_argument(
        "--from-manifest-split",
        default=None,
        help="Filter source manifest by split=train|val|test (used with --from-manifest)",
    )
    parser.add_argument(
        "--from-manifest-label",
        default=0,
        type=int,
        help="Filter source manifest by label (default: 0 normals; used with --from-manifest)",
    )
    parser.add_argument(
        "--from-manifest-n",
        default=0,
        type=int,
        help="Number of source images to use (0 = all; used with --from-manifest)",
    )
    parser.add_argument(
        "--from-manifest-root-fallback",
        default=None,
        help="Root directory used to resolve relative manifest paths (used with --from-manifest)",
    )
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


def _resolve_manifest_path(raw: str, *, manifest_path: Path, root_fallback: Path | None) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"Manifest image_path not found: {p}")
        return p

    cand1 = (manifest_path.parent / p).resolve()
    if cand1.exists():
        return cand1
    if root_fallback is not None:
        cand2 = (root_fallback / p).resolve()
        if cand2.exists():
            return cand2

    raise FileNotFoundError(
        f"Manifest image_path not found: {p} (relative to manifest dir or --from-manifest-root-fallback)"
    )


def synthesize_preview_grid(
    *,
    in_dir: str | Path,
    out_root: str | Path,
    preset: str,
    blend: str,
    alpha: float,
    seed: int,
    n: int,
    cols: int,
    preview_out: str | Path | None,
) -> Path:
    """Generate a preview grid image for a synthesis preset (smoke/debug)."""

    in_path = Path(in_dir)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    items = _iter_images(in_path)
    if not items:
        raise ValueError(f"No images found in --in-dir: {in_path}")

    out_path = out_root_path / "preview.png" if preview_out is None else Path(preview_out)

    syn = AnomalySynthesizer(
        SynthSpec(preset=str(preset), probability=1.0, blend=str(blend), alpha=float(alpha))
    )

    rng = np.random.default_rng(int(seed))
    n = max(1, int(n))
    cols = max(1, int(cols))

    imgs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    import cv2  # local import

    for i in range(n):
        base_path = items[int(rng.integers(0, len(items)))]
        base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if base is None:
            raise ValueError(f"Failed to load image: {base_path}")
        base_u8 = np.asarray(base, dtype=np.uint8)

        res = syn(base_u8, seed=(int(seed) + 1009 * i))
        imgs.append(res.image_u8)
        masks.append(res.mask_u8)

    from pyimgano.synthesis.preview import save_preview_grid

    return save_preview_grid(out_path, imgs, masks=masks, cols=cols)


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


def synthesize_dataset_from_manifest(
    *,
    manifest_path: str | Path,
    out_root: str | Path,
    category: str = "synthetic",
    preset: str = "scratch",
    blend: str = "alpha",
    alpha: float = 0.9,
    seed: int = 0,
    source_category: str | None = None,
    source_split: str | None = None,
    source_label: int | None = 0,
    source_n: int = 0,
    source_root_fallback: str | Path | None = None,
    out_manifest_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Generate a synthetic dataset by sampling normal images from an existing manifest.

    Output follows the built-in `custom` dataset layout under `out_root`.

    Manifest output uses absolute paths by default, because manifest-driven pipelines
    often live outside the dataset root and absolute paths are safer.
    """

    mp = Path(manifest_path)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    root_fallback = None if source_root_fallback is None else Path(source_root_fallback)
    if root_fallback is not None and not root_fallback.exists():
        raise FileNotFoundError(f"--from-manifest-root-fallback not found: {root_fallback}")

    cat_filter = None if source_category is None else str(source_category)
    split_filter = None if source_split is None else str(source_split).strip().lower()
    label_filter = None if source_label is None else int(source_label)

    if split_filter is not None and split_filter not in ("train", "val", "test"):
        raise ValueError(f"--from-manifest-split must be train|val|test, got: {source_split!r}")
    if label_filter is not None and label_filter not in (0, 1):
        raise ValueError(f"--from-manifest-label must be 0/1, got: {source_label!r}")

    candidates: list[Path] = []
    for rec in iter_manifest_records(mp):
        if cat_filter is not None and str(rec.category) != cat_filter:
            continue
        if split_filter is not None and (rec.split is None or str(rec.split) != split_filter):
            continue
        if label_filter is not None:
            lab = 0 if rec.label is None else int(rec.label)
            if lab != label_filter:
                continue
        candidates.append(
            _resolve_manifest_path(rec.image_path, manifest_path=mp, root_fallback=root_fallback)
        )

    if not candidates:
        raise ValueError(
            "No manifest records matched the requested filters. "
            f"category={cat_filter!r} split={split_filter!r} label={label_filter!r}"
        )

    rng = np.random.default_rng(int(seed))
    n = int(source_n)
    if n <= 0:
        selected = list(candidates)
    elif n <= len(candidates):
        perm = rng.permutation(len(candidates)).tolist()
        selected = [candidates[i] for i in perm[:n]]
    else:
        idxs = rng.integers(0, len(candidates), size=n, endpoint=False).tolist()
        selected = [candidates[i] for i in idxs]

    # Layout follows `CustomDataset` + `pyimgano-manifest`.
    train_dir = out_root_path / "train" / "normal"
    test_normal_dir = out_root_path / "test" / "normal"
    test_anomaly_dir = out_root_path / "test" / "anomaly"
    gt_anomaly_dir = out_root_path / "ground_truth" / "anomaly"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_normal_dir.mkdir(parents=True, exist_ok=True)
    test_anomaly_dir.mkdir(parents=True, exist_ok=True)
    gt_anomaly_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected normals into output root to make the dataset self-contained.
    for i, p in enumerate(selected):
        dst_train = train_dir / f"train_{i:05d}{p.suffix.lower()}"
        dst_test = test_normal_dir / f"good_{i:05d}{p.suffix.lower()}"
        shutil.copyfile(str(p), str(dst_train))
        shutil.copyfile(str(p), str(dst_test))

    syn = AnomalySynthesizer(
        SynthSpec(preset=str(preset), probability=1.0, blend=str(blend), alpha=float(alpha))
    )

    # Generate one anomaly per selected normal.
    test_normals = _iter_images(test_normal_dir)
    for i, p in enumerate(test_normals):
        import cv2  # local import

        base = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if base is None:
            raise ValueError(f"Failed to load image: {p}")
        base_u8 = np.asarray(base, dtype=np.uint8)

        res = syn(base_u8, seed=(int(seed) + 1009 * i))
        out_img = res.image_u8
        out_mask = res.mask_u8

        out_name = f"bad_{i:05d}.png"
        img_out = test_anomaly_dir / out_name
        mask_out = gt_anomaly_dir / f"{Path(out_name).stem}_mask.png"

        _write_u8_bgr(img_out, out_img)
        _write_u8_bgr(mask_out, out_mask)

    if out_manifest_path is None:
        out_manifest_path = out_root_path / "manifest.jsonl"

    records = generate_manifest_from_custom_layout(
        root=out_root_path,
        out_path=out_manifest_path,
        category=str(category),
        absolute_paths=True,
        include_masks=True,
    )
    return records


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if bool(getattr(args, "preview", False)):
            if args.in_dir is None:
                raise ValueError("--preview is only supported with --in-dir (not --from-manifest)")
            synthesize_preview_grid(
                in_dir=str(args.in_dir),
                out_root=str(args.out_root),
                preset=str(args.preset),
                blend=str(args.blend),
                alpha=float(args.alpha),
                seed=int(args.seed),
                n=int(args.preview_n),
                cols=int(args.preview_cols),
                preview_out=(None if args.preview_out is None else str(args.preview_out)),
            )
            return 0

        if args.from_manifest is not None:
            synthesize_dataset_from_manifest(
                manifest_path=str(args.from_manifest),
                out_root=str(args.out_root),
                category=str(args.category),
                preset=str(args.preset),
                blend=str(args.blend),
                alpha=float(args.alpha),
                seed=int(args.seed),
                source_category=(
                    None if args.from_manifest_category is None else str(args.from_manifest_category)
                ),
                source_split=(None if args.from_manifest_split is None else str(args.from_manifest_split)),
                source_label=(None if args.from_manifest_label is None else int(args.from_manifest_label)),
                source_n=int(args.from_manifest_n),
                source_root_fallback=(
                    None if args.from_manifest_root_fallback is None else str(args.from_manifest_root_fallback)
                ),
                out_manifest_path=(None if args.manifest is None else str(args.manifest)),
            )
        else:
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
