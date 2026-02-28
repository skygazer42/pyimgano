# -*- coding: utf-8 -*-
"""Example: Torchvision embeddings + core_mahalanobis_shrinkage.

This demonstrates a practical industrial baseline:

  images -> embedding extractor -> Mahalanobis distance (shrinkage) -> scores

Notes
-----
- By default this example uses `pretrained=False` to avoid implicit weight downloads.
- For a stable score scale across runs/backbones, consider wrapping the core with
  `core_score_standardizer(method="rank")` (see `docs/CORE_SELECTION_ON_EMBEDDINGS.md`).
- Works with datasets in the built-in `custom` layout:
    root/
      train/normal/*.png
      test/normal/*.png
      test/anomaly/*.png
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
    return [str(p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="embedding_plus_core_mahalanobis_shrinkage")
    parser.add_argument("--root", required=True, help="Dataset root (custom layout)")
    parser.add_argument("--device", default="cpu", help="cpu|cuda")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--backbone", default="resnet18", help="torchvision backbone name")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pool", default="avg", choices=["avg", "max", "gem", "cls"])
    args = parser.parse_args(argv)

    root = Path(args.root)
    train_paths = _iter_images(root / "train" / "normal")
    test_normal = _iter_images(root / "test" / "normal")
    test_anom = _iter_images(root / "test" / "anomaly")
    test_paths = test_normal + test_anom
    test_labels = np.asarray([0] * len(test_normal) + [1] * len(test_anom), dtype=np.int64)

    if not train_paths or not test_paths:
        raise SystemExit("Dataset appears empty or missing required folders.")

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
        core_detector="core_mahalanobis_shrinkage",
        core_kwargs={"assume_centered": False},
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
            "test_labels": test_labels.tolist(),
        }
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

