"""
Paper-aligned SoftPatch example on MVTec AD / VisA style datasets.

This example uses the built-in pipeline helpers in `pyimgano.pipelines.mvtec_visa`.

Notes:
- Pass `--pretrained` for the paper's ImageNet-pretrained WideResNet50-2 weights.
- For faster kNN on large memory banks, install `pyimgano[faiss]` and pass `--knn-backend faiss`.
"""

from __future__ import annotations

import argparse
import json

from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import evaluate_split, load_benchmark_split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softpatch-mvtec-visa")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument(
        "--pixel", action="store_true", help="Compute pixel-level metrics if possible"
    )

    parser.add_argument(
        "--image-size", type=int, default=224, help="Center-crop size (paper default: 224)"
    )
    parser.add_argument("--knn-backend", default="sklearn", choices=["sklearn", "faiss"])
    parser.add_argument("--n-neighbors", type=int, default=1)
    parser.add_argument(
        "--coreset", type=float, default=0.1, help="Greedy memory-bank ratio (paper default: 0.1)"
    )
    parser.add_argument(
        "--train-outlier-quantile",
        type=float,
        default=0.15,
        help="Remove the top-q positional outlier patches (paper default: 0.15)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    split = load_benchmark_split(
        dataset=args.dataset,
        root=args.root,
        category=args.category,
        resize=(256, 256),
        load_masks=True,
    )

    detector = create_model(
        "vision_softpatch",
        device=args.device,
        contamination=args.contamination,
        pretrained=bool(args.pretrained),
        image_size=args.image_size,
        resize_size=max(256, args.image_size),
        knn_backend=args.knn_backend,
        n_neighbors=args.n_neighbors,
        coreset_sampling_ratio=args.coreset,
        train_patch_outlier_quantile=args.train_outlier_quantile,
    )

    results = evaluate_split(
        detector,
        split,
        compute_pixel_scores=bool(args.pixel),
    )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
