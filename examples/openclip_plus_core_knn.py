# -*- coding: utf-8 -*-
"""Example: OpenCLIP embeddings + core_knn (optional).

This example requires the optional dependency:
  pip install 'open_clip_torch'

Important:
- By default `openclip_embed` uses `pretrained=None` to avoid implicit downloads.
- If you set `--pretrained <tag>`, OpenCLIP may download weights (network required).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pyimgano.models.registry import create_model

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_images(d: Path) -> list[str]:
    if not d.exists():
        return []
    return [
        str(p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openclip_plus_core_knn")
    parser.add_argument("--root", required=True, help="Dataset root (custom layout)")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--model-name", default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument(
        "--pretrained",
        default=None,
        help="OpenCLIP pretrained tag (default: none; avoids weight downloads)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    train_paths = _iter_images(root / "train" / "normal")
    test_normal = _iter_images(root / "test" / "normal")
    test_anom = _iter_images(root / "test" / "anomaly")
    test_paths = test_normal + test_anom

    if not train_paths or not test_paths:
        raise SystemExit("Dataset appears empty or missing required folders.")

    det = create_model(
        "vision_embedding_core",
        contamination=float(args.contamination),
        embedding_extractor="openclip_embed",
        embedding_kwargs={
            "model_name": str(args.model_name),
            "pretrained": (None if args.pretrained is None else str(args.pretrained)),
            "device": str(args.device),
        },
        core_detector="core_knn",
        core_kwargs={"method": "largest", "n_neighbors": 5},
    )

    det.fit(train_paths)
    scores = np.asarray(det.decision_function(test_paths), dtype=np.float64).reshape(-1)
    print(
        {
            "train_n": len(train_paths),
            "test_n": len(test_paths),
            "scores_min": float(scores.min()),
            "scores_max": float(scores.max()),
            "scores_mean": float(scores.mean()),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
