"""
AnomalyDINO (DINOv2 patch-kNN) example on MVTec AD / VisA style datasets.

This example uses the built-in pipeline helpers in `pyimgano.pipelines.mvtec_visa`.

Notes:
- The default DINOv2 embedder uses `torch.hub` and may download weights on first run.
- For faster kNN on large memory banks, install `pyimgano[faiss]` and pass `--knn-backend faiss`.
"""

from __future__ import annotations

import argparse
import json

from pyimgano.pipelines.mvtec_visa import build_default_detector, evaluate_split, load_benchmark_split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anomalydino-mvtec-visa")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--pixel", action="store_true", help="Compute pixel-level metrics if possible")

    parser.add_argument("--image-size", type=int, default=518, help="DINOv2 square resize size")
    parser.add_argument("--knn-backend", default="sklearn", choices=["sklearn", "faiss"])
    parser.add_argument("--n-neighbors", type=int, default=1)
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

    detector = build_default_detector(
        model="vision_anomalydino",
        device=args.device,
        contamination=args.contamination,
        image_size=args.image_size,
        knn_backend=args.knn_backend,
        n_neighbors=args.n_neighbors,
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

