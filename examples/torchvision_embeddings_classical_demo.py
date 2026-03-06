# -*- coding: utf-8 -*-
"""Example: torchvision embeddings + configurable classical core detector.

This demonstrates a common industrial path:

  image paths -> torchvision embeddings -> core_* detector -> anomaly scores

Notes
-----
- Defaults stay offline-safe (`pretrained=False`).
- Works with the built-in `custom` dataset layout:
    root/
      train/normal/*.png
      test/normal/*.png
      test/anomaly/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyimgano.models import create_model

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_images(d: Path) -> list[str]:
    if not d.exists():
        return []
    return [
        str(p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]


def _parse_json_object(text: str, *, name: str) -> dict[str, object]:
    try:
        payload = json.loads(str(text))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} must be valid JSON. Original error: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{name} must decode to a JSON object/dict, got {type(payload).__name__}")
    return dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="torchvision_embeddings_classical_demo")
    parser.add_argument("--root", required=True, help="Dataset root using the built-in custom layout")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--backbone", default="resnet18", help="torchvision backbone name")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pool", default="avg", choices=["avg", "max", "gem", "cls"])
    parser.add_argument("--core-detector", default="core_ecod", help="Registered core_* detector name")
    parser.add_argument(
        "--core-kwargs-json",
        default="{}",
        help='JSON object passed to the selected core detector, e.g. \'{"n_neighbors": 5}\'',
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    train_paths = _iter_images(root / "train" / "normal")
    test_normal = _iter_images(root / "test" / "normal")
    test_anom = _iter_images(root / "test" / "anomaly")
    test_paths = test_normal + test_anom
    test_labels = np.asarray([0] * len(test_normal) + [1] * len(test_anom), dtype=np.int64)

    if not train_paths or not test_paths:
        raise SystemExit("Dataset appears empty or missing required custom-layout folders.")

    core_kwargs = _parse_json_object(str(args.core_kwargs_json), name="--core-kwargs-json")

    det = create_model(
        "vision_embedding_core",
        contamination=float(args.contamination),
        embedding_extractor="torchvision_backbone",
        embedding_kwargs={
            "backbone": str(args.backbone),
            "pretrained": bool(args.pretrained),
            "pool": str(args.pool),
            "device": str(args.device),
        },
        core_detector=str(args.core_detector),
        core_kwargs=core_kwargs,
    )

    det.fit(train_paths)
    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)

    print(
        {
            "core_detector": str(args.core_detector),
            "core_kwargs": core_kwargs,
            "train_n": len(train_paths),
            "test_n": len(test_paths),
            "scores_min": float(scores.min()),
            "scores_max": float(scores.max()),
            "scores_mean": float(scores.mean()),
            "test_labels": test_labels.tolist(),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
