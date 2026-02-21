from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import evaluate_split, load_benchmark_split
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess
from pyimgano.reporting.report import save_run_report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-benchmark")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--model", default="vision_patchcore", help="Registered model name")
    parser.add_argument("--device", default="cpu", help="cpu|cuda (model dependent)")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixel", action="store_true", help="Compute pixel-level metrics if possible")
    parser.add_argument(
        "--pixel-postprocess",
        action="store_true",
        help="Apply post-processing to anomaly maps before computing pixel metrics",
    )
    parser.add_argument(
        "--pixel-post-norm",
        default="minmax",
        choices=["minmax", "percentile", "none"],
        help="Normalization method for pixel anomaly maps (only if --pixel-postprocess)",
    )
    parser.add_argument(
        "--pixel-post-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for normalization (only if --pixel-post-norm=percentile)",
    )
    parser.add_argument(
        "--pixel-post-gaussian-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for anomaly maps (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-open-ksize",
        type=int,
        default=0,
        help="Morphological open kernel size (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-close-ksize",
        type=int,
        default=0,
        help="Morphological close kernel size (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-component-threshold",
        type=float,
        default=None,
        help="Threshold for connected-components filtering (requires --pixel-post-min-component-area > 0)",
    )
    parser.add_argument(
        "--pixel-post-min-component-area",
        type=int,
        default=0,
        help="Minimum component area for filtering (0 disables)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    split = load_benchmark_split(
        dataset=args.dataset,
        root=args.root,
        category=args.category,
        resize=(256, 256),
        load_masks=True,
    )

    detector = create_model(
        args.model,
        device=args.device,
        contamination=args.contamination,
        pretrained=args.pretrained,
    )

    postprocess = None
    if args.pixel and args.pixel_postprocess:
        postprocess = AnomalyMapPostprocess(
            normalize=True,
            normalize_method=str(args.pixel_post_norm),
            percentile_range=(
                float(args.pixel_post_percentiles[0]),
                float(args.pixel_post_percentiles[1]),
            ),
            gaussian_sigma=float(args.pixel_post_gaussian_sigma),
            morph_open_ksize=int(args.pixel_post_open_ksize),
            morph_close_ksize=int(args.pixel_post_close_ksize),
            component_threshold=args.pixel_post_component_threshold,
            min_component_area=int(args.pixel_post_min_component_area),
        )

    results = evaluate_split(
        detector,
        split,
        compute_pixel_scores=bool(args.pixel),
        postprocess=postprocess,
    )

    payload = {
        "dataset": args.dataset,
        "category": args.category,
        "model": args.model,
        "results": results,
    }

    if args.output:
        save_run_report(Path(args.output), payload)
    else:
        print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
