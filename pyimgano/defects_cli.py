from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _guess_mask_format(path: str | Path, *, default: str = "png") -> str:
    suf = str(Path(path).suffix).lower().lstrip(".")
    if suf in {"png", "npy", "npz"}:
        return suf
    return str(default)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pyimgano-defects")
    p.add_argument("--map", required=True, help="Path to an anomaly map saved as .npy (HxW float-like)")
    p.add_argument(
        "--pixel-threshold",
        type=float,
        required=True,
        help="Pixels >= threshold are considered defective",
    )
    p.add_argument(
        "--out-mask",
        default=None,
        help="Optional output path for the exported binary mask (png/npy/npz inferred from suffix)",
    )
    p.add_argument(
        "--mask-format",
        default=None,
        choices=["png", "npy", "npz"],
        help="Override mask format for --out-mask (default: inferred from suffix)",
    )
    p.add_argument(
        "--out-jsonl",
        default=None,
        help="Optional output JSONL path (one line). When omitted, prints to stdout.",
    )
    p.add_argument(
        "--roi-xyxy-norm",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional normalized ROI rectangle (x1 y1 x2 y2 in [0,1]) applied to regions.",
    )
    p.add_argument(
        "--mask-space",
        default="roi",
        choices=["roi", "full"],
        help="Whether to export ROI-limited or full-map mask (default: roi).",
    )

    # A small subset of defects knobs (keep this CLI ergonomic; infer_cli covers the full surface).
    p.add_argument("--defect-min-area", type=int, default=0)
    p.add_argument("--defect-open-ksize", type=int, default=0)
    p.add_argument("--defect-close-ksize", type=int, default=0)
    p.add_argument("--defect-fill-holes", action="store_true")
    p.add_argument("--defects-mask-dilate", type=int, default=0)

    p.add_argument("--defect-border-ignore-px", type=int, default=0)
    p.add_argument(
        "--defect-map-smoothing",
        default="none",
        choices=["none", "median", "gaussian", "box"],
    )
    p.add_argument("--defect-map-smoothing-ksize", type=int, default=0)
    p.add_argument("--defect-map-smoothing-sigma", type=float, default=0.0)

    p.add_argument("--defect-hysteresis", action="store_true")
    p.add_argument("--defect-hysteresis-low", type=float, default=None)
    p.add_argument("--defect-hysteresis-high", type=float, default=None)

    p.add_argument("--defect-min-fill-ratio", type=float, default=None)
    p.add_argument("--defect-max-aspect-ratio", type=float, default=None)
    p.add_argument("--defect-min-solidity", type=float, default=None)
    p.add_argument("--defect-min-score-max", type=float, default=None)
    p.add_argument("--defect-min-score-mean", type=float, default=None)

    p.add_argument("--defect-merge-nearby", action="store_true")
    p.add_argument("--defect-merge-nearby-max-gap-px", type=int, default=0)
    p.add_argument("--defect-max-regions", type=int, default=None)
    p.add_argument(
        "--defect-max-regions-sort-by",
        default="score_max",
        choices=["score_max", "score_mean", "area"],
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        amap = np.asarray(np.load(str(args.map)), dtype=np.float32)

        from pyimgano.defects.extract import extract_defects_from_anomaly_map
        from pyimgano.defects.io import save_binary_mask

        defects = extract_defects_from_anomaly_map(
            amap,
            pixel_threshold=float(args.pixel_threshold),
            roi_xyxy_norm=(list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None),
            mask_space=str(args.mask_space),
            border_ignore_px=int(args.defect_border_ignore_px),
            map_smoothing_method=str(args.defect_map_smoothing),
            map_smoothing_ksize=int(args.defect_map_smoothing_ksize),
            map_smoothing_sigma=float(args.defect_map_smoothing_sigma),
            hysteresis_enabled=bool(args.defect_hysteresis),
            hysteresis_low=(float(args.defect_hysteresis_low) if args.defect_hysteresis_low is not None else None),
            hysteresis_high=(float(args.defect_hysteresis_high) if args.defect_hysteresis_high is not None else None),
            mask_dilate_ksize=int(args.defects_mask_dilate),
            open_ksize=int(args.defect_open_ksize),
            close_ksize=int(args.defect_close_ksize),
            fill_holes=bool(args.defect_fill_holes),
            min_area=int(args.defect_min_area),
            min_fill_ratio=(float(args.defect_min_fill_ratio) if args.defect_min_fill_ratio is not None else None),
            max_aspect_ratio=(float(args.defect_max_aspect_ratio) if args.defect_max_aspect_ratio is not None else None),
            min_solidity=(float(args.defect_min_solidity) if args.defect_min_solidity is not None else None),
            min_score_max=(float(args.defect_min_score_max) if args.defect_min_score_max is not None else None),
            min_score_mean=(float(args.defect_min_score_mean) if args.defect_min_score_mean is not None else None),
            merge_nearby_enabled=bool(args.defect_merge_nearby),
            merge_nearby_max_gap_px=int(args.defect_merge_nearby_max_gap_px),
            max_regions_sort_by=str(args.defect_max_regions_sort_by),
            max_regions=(int(args.defect_max_regions) if args.defect_max_regions is not None else None),
        )

        mask_meta: dict[str, Any] = {
            "shape": [int(d) for d in np.asarray(defects["mask"]).shape],
            "dtype": str(np.asarray(defects["mask"]).dtype),
        }
        if args.out_mask is not None:
            fmt = (
                str(args.mask_format)
                if args.mask_format is not None
                else _guess_mask_format(str(args.out_mask))
            )
            written = save_binary_mask(np.asarray(defects["mask"]), str(args.out_mask), format=fmt)
            mask_meta.update({"path": str(written), "encoding": str(fmt)})

        payload = {
            "space": defects["space"],
            "pixel_threshold": float(defects["pixel_threshold"]),
            "mask": mask_meta,
            "regions": list(defects["regions"]),
            "map_stats_roi": defects.get("map_stats_roi", None),
        }

        line = json.dumps(payload, sort_keys=True)
        if args.out_jsonl is not None:
            out_path = Path(str(args.out_jsonl))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(line + "\n", encoding="utf-8")
        else:
            print(line)

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

