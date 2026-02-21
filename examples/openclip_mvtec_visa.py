"""
OpenCLIP-based detectors example on MVTec AD / VisA style datasets.

Detectors:
- vision_openclip_promptscore: prompt-based scoring + anomaly maps
- vision_openclip_patchknn: OpenCLIP patch embeddings + kNN (AnomalyDINO-style)

Notes:
- Requires: `pip install "pyimgano[clip]"`
- OpenCLIP may download weights on first run (cached by torch).
"""

from __future__ import annotations

import argparse
import json

from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import evaluate_split, load_benchmark_split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclip-mvtec-visa")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--pixel", action="store_true", help="Compute pixel-level metrics if possible")

    parser.add_argument(
        "--model",
        default="vision_openclip_promptscore",
        choices=["vision_openclip_promptscore", "vision_openclip_patchknn"],
    )
    parser.add_argument("--openclip-model", default="ViT-B-32")
    parser.add_argument("--openclip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument(
        "--class-name",
        default=None,
        help="Class name used for prompt templates (promptscore only). Defaults to --category.",
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

    model_kwargs = dict(
        contamination=args.contamination,
        device=args.device,
        openclip_model_name=args.openclip_model,
        openclip_pretrained=args.openclip_pretrained,
    )

    if args.model == "vision_openclip_promptscore":
        model_kwargs["class_name"] = args.class_name or args.category

    detector = create_model(args.model, **model_kwargs)

    results = evaluate_split(
        detector,
        split,
        compute_pixel_scores=bool(args.pixel),
    )

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

