"""
Quick pixel-first comparison script for industrial AD detectors.

Evaluates multiple detectors on a single MVTec/VisA category and prints a compact summary.

Detectors included by default:
- PatchCore (TorchVision backbone)
- AnomalyDINO (DINOv2 patch-kNN, torch.hub)
- SoftPatch (robust patch-memory, torch.hub)
- OpenCLIP PatchKNN (optional, requires `pyimgano[clip]`)
"""

from __future__ import annotations

import argparse
import json

from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import evaluate_split, load_benchmark_split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixel-first-compare")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--pixel", action="store_true", help="Compute pixel-level metrics if possible")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _summarize(results: dict) -> dict:
    out = {
        "auroc": results.get("auroc", None),
        "average_precision": results.get("average_precision", None),
    }
    pixel = results.get("pixel_metrics")
    if isinstance(pixel, dict):
        out["pixel_auroc"] = pixel.get("pixel_auroc")
        out["pixel_average_precision"] = pixel.get("pixel_average_precision")
        out["aupro"] = pixel.get("aupro")
    return out


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    split = load_benchmark_split(
        dataset=args.dataset,
        root=args.root,
        category=args.category,
        resize=(256, 256),
        load_masks=True,
    )

    model_specs: list[tuple[str, dict]] = [
        (
            "vision_patchcore",
            {
                "device": args.device,
                "contamination": args.contamination,
                "pretrained": bool(args.pretrained),
                "coreset_sampling_ratio": 0.1,
            },
        ),
        (
            "vision_anomalydino",
            {
                "device": args.device,
                "contamination": args.contamination,
            },
        ),
        (
            "vision_softpatch",
            {
                "device": args.device,
                "contamination": args.contamination,
                "train_patch_outlier_quantile": 0.1,
                "coreset_sampling_ratio": 0.5,
            },
        ),
        (
            "vision_openclip_patchknn",
            {
                "device": args.device,
                "contamination": args.contamination,
            },
        ),
    ]

    summaries: dict[str, dict] = {}
    for model_name, kwargs in model_specs:
        try:
            detector = create_model(model_name, **kwargs)
        except Exception as exc:
            summaries[model_name] = {"skipped": True, "reason": str(exc)}
            continue

        try:
            results = evaluate_split(detector, split, compute_pixel_scores=bool(args.pixel))
        except Exception as exc:
            summaries[model_name] = {"error": str(exc)}
            continue

        summaries[model_name] = _summarize(results)

    payload = {
        "dataset": args.dataset,
        "category": args.category,
        "summaries": summaries,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

