from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.datasets.manifest import iter_manifest_records
from pyimgano.manifest_cli import generate_manifest_from_custom_layout
from pyimgano.synthesis.masks import ensure_u8_mask
from pyimgano.synthesis.presets import get_preset_names, make_preset_mixture
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
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--preset",
        default="scratch",
        choices=get_preset_names(),
        help="Synthetic anomaly preset name",
    )
    preset_group.add_argument(
        "--presets",
        default=None,
        nargs="+",
        choices=get_preset_names(),
        help="Sample presets from a mixture each time (e.g. --presets scratch stain tape)",
    )
    parser.add_argument(
        "--blend",
        default="alpha",
        choices=["alpha", "poisson"],
        help="Blend mode used to inject anomalies",
    )
    parser.add_argument("--alpha", type=float, default=0.9, help="Alpha for alpha blending (0..1)")
    parser.add_argument(
        "--num-defects",
        type=int,
        default=1,
        help="Number of defect injections attempted per anomaly sample (default: 1).",
    )
    parser.add_argument(
        "--severity-range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Severity range in [0,1] sampled per anomaly (default: 1.0 1.0).",
    )
    parser.add_argument(
        "--defect-bank-dir",
        default=None,
        help=(
            "Optional defect bank directory for copy/paste synthesis.\n"
            "When provided, the bank preset is used instead of built-in --preset/--presets.\n"
            "Bank format: images + optional *_mask.png or masks/ subdir (or PNG alpha)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (deterministic output)")
    parser.add_argument(
        "--roi-mask",
        default=None,
        help="Optional ROI mask path (non-zero => allowed anomaly region). Resized to match each image.",
    )
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
    parser.add_argument(
        "--preview-n", type=int, default=16, help="Number of samples in preview grid"
    )
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


def _read_roi_mask(path: str | Path) -> np.ndarray:
    import cv2  # local import

    p = Path(path)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read ROI mask: {p}")
    return ensure_u8_mask(np.asarray(img, dtype=np.uint8))


def _resize_roi_mask(roi_u8: np.ndarray, *, shape_hw: tuple[int, int]) -> np.ndarray:
    import cv2  # local import

    h, w = int(shape_hw[0]), int(shape_hw[1])
    roi = np.asarray(roi_u8, dtype=np.uint8)
    if roi.shape == (h, w):
        return roi
    resized = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    return ensure_u8_mask(np.asarray(resized, dtype=np.uint8), shape_hw=(h, w))


def _make_synthesizer(
    *,
    preset: str,
    presets: list[str] | None,
    blend: str,
    alpha: float,
    defect_bank_dir: str | Path | None = None,
) -> AnomalySynthesizer:
    if defect_bank_dir is not None:
        from pyimgano.synthesis.defect_bank import DefectBank, make_defect_bank_preset

        bank = DefectBank.from_dir(defect_bank_dir)
        preset_fn = make_defect_bank_preset(bank)
        return AnomalySynthesizer(
            SynthSpec(preset="defect_bank", probability=1.0, blend=str(blend), alpha=float(alpha)),
            preset_fn=preset_fn,
        )

    if presets:
        preset_names = [str(p).strip().lower() for p in list(presets)]
        preset_fn = make_preset_mixture(preset_names)
        spec_preset = preset_names[0]
        return AnomalySynthesizer(
            SynthSpec(
                preset=str(spec_preset), probability=1.0, blend=str(blend), alpha=float(alpha)
            ),
            preset_fn=preset_fn,
        )

    return AnomalySynthesizer(
        SynthSpec(preset=str(preset), probability=1.0, blend=str(blend), alpha=float(alpha))
    )


def _normalize_severity_range(
    severity_range: tuple[float, float] | list[float] | None,
) -> tuple[float, float]:
    if severity_range is None:
        return 1.0, 1.0
    if len(severity_range) != 2:
        raise ValueError("--severity-range must provide exactly 2 floats: MIN MAX")
    lo = float(severity_range[0])
    hi = float(severity_range[1])
    if lo > hi:
        lo, hi = hi, lo
    lo = float(np.clip(lo, 0.0, 1.0))
    hi = float(np.clip(hi, 0.0, 1.0))
    return lo, hi


def _sample_severity(*, seed: int, idx: int, severity_range: tuple[float, float]) -> float:
    lo, hi = float(severity_range[0]), float(severity_range[1])
    if hi <= lo:
        return float(lo)
    # Use a per-index RNG to keep determinism stable even if we change retry loops.
    rng = np.random.default_rng(int(seed) + 1618033 * int(idx))
    return float(rng.uniform(lo, hi))


def _attach_meta_to_manifest_records(
    records: list[dict[str, Any]],
    *,
    meta_by_filename: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not meta_by_filename:
        return records

    out: list[dict[str, Any]] = []
    for rec in records:
        r = dict(rec)
        lab = r.get("label", None)
        if lab is not None and int(lab) == 1:
            try:
                fname = Path(str(r.get("image_path", ""))).name
            except Exception:
                fname = ""
            meta = meta_by_filename.get(fname)
            if meta is not None:
                r["meta"] = dict(meta)
        out.append(r)
    return out


def _ensure_synthesis_meta_fields(
    meta: dict[str, Any], *, fallback_preset_id: str
) -> dict[str, Any]:
    """Normalize synthesis meta for stable downstream consumption.

    Requirements (v4):
    - `meta.severity`: float in [0,1] (defaults to 1.0)
    - `meta.preset_id`: stable preset identifier string
    """

    m = dict(meta)

    try:
        sev = float(m.get("severity", 1.0))
    except Exception:
        sev = 1.0
    m["severity"] = float(np.clip(sev, 0.0, 1.0))

    preset_id = m.get("preset_id", None)
    if preset_id is None:
        preset_id = m.get("preset", None)
    if preset_id is None:
        preset_id = fallback_preset_id
    m["preset_id"] = str(preset_id)

    return m


def _rewrite_manifest_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
    presets: list[str] | None = None,
    blend: str,
    alpha: float,
    num_defects: int = 1,
    severity_range: tuple[float, float] | list[float] | None = None,
    defect_bank_dir: str | Path | None = None,
    seed: int,
    n: int,
    cols: int,
    preview_out: str | Path | None,
    roi_mask_path: str | Path | None = None,
) -> Path:
    """Generate a preview grid image for a synthesis preset (smoke/debug)."""

    in_path = Path(in_dir)
    out_root_path = Path(out_root)
    out_root_path.mkdir(parents=True, exist_ok=True)

    items = _iter_images(in_path)
    if not items:
        raise ValueError(f"No images found in --in-dir: {in_path}")

    out_path = out_root_path / "preview.png" if preview_out is None else Path(preview_out)

    sev_rng = _normalize_severity_range(severity_range)
    syn = _make_synthesizer(
        preset=str(preset),
        presets=presets,
        blend=str(blend),
        alpha=float(alpha),
        defect_bank_dir=defect_bank_dir,
    )
    roi_raw = None if roi_mask_path is None else _read_roi_mask(roi_mask_path)

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
        roi = None if roi_raw is None else _resize_roi_mask(roi_raw, shape_hw=base_u8.shape[:2])
        sev = _sample_severity(seed=int(seed), idx=int(i), severity_range=sev_rng)
        res = syn(
            base_u8,
            seed=(int(seed) + 1009 * i),
            roi_mask=roi,
            num_defects=int(num_defects),
            severity=float(sev),
        )
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
    presets: list[str] | None = None,
    blend: str = "alpha",
    alpha: float = 0.9,
    num_defects: int = 1,
    severity_range: tuple[float, float] | list[float] | None = None,
    defect_bank_dir: str | Path | None = None,
    seed: int = 0,
    roi_mask_path: str | Path | None = None,
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

    sev_rng = _normalize_severity_range(severity_range)
    syn = _make_synthesizer(
        preset=str(preset),
        presets=presets,
        blend=str(blend),
        alpha=float(alpha),
        defect_bank_dir=defect_bank_dir,
    )
    roi_raw = None if roi_mask_path is None else _read_roi_mask(roi_mask_path)
    meta_by_filename: dict[str, dict[str, Any]] = {}

    # Generate anomalies by synthesizing from test-normal items (with replacement).
    for i in range(n_test_anomaly):
        base_path = test_normal_items[int(rng.integers(0, len(test_normal_items)))]

        import cv2  # local import

        base = cv2.imread(str(base_path), cv2.IMREAD_COLOR)
        if base is None:
            raise ValueError(f"Failed to load image: {base_path}")
        base_u8 = np.asarray(base, dtype=np.uint8)

        accepted = False
        sev = _sample_severity(seed=int(seed), idx=int(i), severity_range=sev_rng)
        for t in range(10):
            roi = None if roi_raw is None else _resize_roi_mask(roi_raw, shape_hw=base_u8.shape[:2])
            res = syn(
                base_u8,
                seed=(int(seed) + 1009 * i + 7919 * t),
                roi_mask=roi,
                num_defects=int(num_defects),
                severity=float(sev),
            )
            out_img = res.image_u8
            out_mask = res.mask_u8

            if int(res.label) == 1 and int(np.sum(out_mask > 0)) > 0:
                out_name = f"bad_{i:05d}.png"
                img_out = test_anomaly_dir / out_name
                mask_out = gt_anomaly_dir / f"{Path(out_name).stem}_mask.png"

                _write_u8_bgr(img_out, out_img)
                _write_u8_bgr(mask_out, out_mask)
                meta_by_filename[out_name] = _ensure_synthesis_meta_fields(
                    dict(res.meta), fallback_preset_id=str(preset)
                )
                accepted = True
                break

        if not accepted:
            # ROI constraints can prevent anomalies. Fall back to a normal sample.
            out_name = f"good_synth_{i:05d}.png"
            img_out = test_normal_dir / out_name
            _write_u8_bgr(img_out, np.asarray(base_u8, dtype=np.uint8))

    if manifest_path is None:
        manifest_path = out_root_path / "manifest.jsonl"

    records_raw = generate_manifest_from_custom_layout(
        root=out_root_path,
        out_path=manifest_path,
        category=str(category),
        absolute_paths=bool(absolute_paths),
        include_masks=True,
    )
    records = _attach_meta_to_manifest_records(records_raw, meta_by_filename=meta_by_filename)
    _rewrite_manifest_jsonl(manifest_path, records)
    return records


def synthesize_dataset_from_manifest(
    *,
    manifest_path: str | Path,
    out_root: str | Path,
    category: str = "synthetic",
    preset: str = "scratch",
    presets: list[str] | None = None,
    blend: str = "alpha",
    alpha: float = 0.9,
    num_defects: int = 1,
    severity_range: tuple[float, float] | list[float] | None = None,
    defect_bank_dir: str | Path | None = None,
    seed: int = 0,
    roi_mask_path: str | Path | None = None,
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

    sev_rng = _normalize_severity_range(severity_range)
    syn = _make_synthesizer(
        preset=str(preset),
        presets=presets,
        blend=str(blend),
        alpha=float(alpha),
        defect_bank_dir=defect_bank_dir,
    )
    roi_raw = None if roi_mask_path is None else _read_roi_mask(roi_mask_path)
    meta_by_filename: dict[str, dict[str, Any]] = {}

    # Generate one anomaly per selected normal.
    test_normals = _iter_images(test_normal_dir)
    for i, p in enumerate(test_normals):
        import cv2  # local import

        base = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if base is None:
            raise ValueError(f"Failed to load image: {p}")
        base_u8 = np.asarray(base, dtype=np.uint8)

        accepted = False
        sev = _sample_severity(seed=int(seed), idx=int(i), severity_range=sev_rng)
        for t in range(10):
            roi = None if roi_raw is None else _resize_roi_mask(roi_raw, shape_hw=base_u8.shape[:2])
            res = syn(
                base_u8,
                seed=(int(seed) + 1009 * i + 7919 * t),
                roi_mask=roi,
                num_defects=int(num_defects),
                severity=float(sev),
            )
            out_img = res.image_u8
            out_mask = res.mask_u8

            if int(res.label) == 1 and int(np.sum(out_mask > 0)) > 0:
                out_name = f"bad_{i:05d}.png"
                img_out = test_anomaly_dir / out_name
                mask_out = gt_anomaly_dir / f"{Path(out_name).stem}_mask.png"

                _write_u8_bgr(img_out, out_img)
                _write_u8_bgr(mask_out, out_mask)
                meta_by_filename[out_name] = _ensure_synthesis_meta_fields(
                    dict(res.meta), fallback_preset_id=str(preset)
                )
                accepted = True
                break

        if not accepted:
            out_name = f"good_synth_{i:05d}.png"
            img_out = test_normal_dir / out_name
            _write_u8_bgr(img_out, np.asarray(base_u8, dtype=np.uint8))

    if out_manifest_path is None:
        out_manifest_path = out_root_path / "manifest.jsonl"

    records_raw = generate_manifest_from_custom_layout(
        root=out_root_path,
        out_path=out_manifest_path,
        category=str(category),
        absolute_paths=True,
        include_masks=True,
    )
    records = _attach_meta_to_manifest_records(records_raw, meta_by_filename=meta_by_filename)
    _rewrite_manifest_jsonl(out_manifest_path, records)
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
                presets=(None if args.presets is None else [str(x) for x in list(args.presets)]),
                blend=str(args.blend),
                alpha=float(args.alpha),
                num_defects=int(args.num_defects),
                severity_range=(
                    list(args.severity_range) if args.severity_range is not None else None
                ),
                defect_bank_dir=(
                    None if args.defect_bank_dir is None else str(args.defect_bank_dir)
                ),
                seed=int(args.seed),
                n=int(args.preview_n),
                cols=int(args.preview_cols),
                preview_out=(None if args.preview_out is None else str(args.preview_out)),
                roi_mask_path=(None if args.roi_mask is None else str(args.roi_mask)),
            )
            return 0

        if args.from_manifest is not None:
            synthesize_dataset_from_manifest(
                manifest_path=str(args.from_manifest),
                out_root=str(args.out_root),
                category=str(args.category),
                preset=str(args.preset),
                presets=(None if args.presets is None else [str(x) for x in list(args.presets)]),
                blend=str(args.blend),
                alpha=float(args.alpha),
                num_defects=int(args.num_defects),
                severity_range=(
                    list(args.severity_range) if args.severity_range is not None else None
                ),
                defect_bank_dir=(
                    None if args.defect_bank_dir is None else str(args.defect_bank_dir)
                ),
                seed=int(args.seed),
                roi_mask_path=(None if args.roi_mask is None else str(args.roi_mask)),
                source_category=(
                    None
                    if args.from_manifest_category is None
                    else str(args.from_manifest_category)
                ),
                source_split=(
                    None if args.from_manifest_split is None else str(args.from_manifest_split)
                ),
                source_label=(
                    None if args.from_manifest_label is None else int(args.from_manifest_label)
                ),
                source_n=int(args.from_manifest_n),
                source_root_fallback=(
                    None
                    if args.from_manifest_root_fallback is None
                    else str(args.from_manifest_root_fallback)
                ),
                out_manifest_path=(None if args.manifest is None else str(args.manifest)),
            )
        else:
            synthesize_dataset(
                in_dir=str(args.in_dir),
                out_root=str(args.out_root),
                category=str(args.category),
                preset=str(args.preset),
                presets=(None if args.presets is None else [str(x) for x in list(args.presets)]),
                blend=str(args.blend),
                alpha=float(args.alpha),
                num_defects=int(args.num_defects),
                severity_range=(
                    list(args.severity_range) if args.severity_range is not None else None
                ),
                defect_bank_dir=(
                    None if args.defect_bank_dir is None else str(args.defect_bank_dir)
                ),
                seed=int(args.seed),
                roi_mask_path=(None if args.roi_mask is None else str(args.roi_mask)),
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
