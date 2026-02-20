from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.evaluation import evaluate_detector
from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import load_benchmark_split
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

    detector.fit(split.train_paths)
    scores = detector.decision_function(split.test_paths)

    pixel_scores = None
    if args.pixel and split.test_masks is not None and hasattr(detector, "get_anomaly_map"):
        maps = [detector.get_anomaly_map(path) for path in split.test_paths]
        pixel_scores = np.stack([np.asarray(m, dtype=np.float32) for m in maps])

    results = evaluate_detector(
        split.test_labels,
        scores,
        pixel_labels=split.test_masks,
        pixel_scores=pixel_scores,
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

