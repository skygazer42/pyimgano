#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _parse_kwargs(text: str | None) -> dict[str, Any]:
    if text is None:
        return {}
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("--kwargs must be a JSON object")
    return dict(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="feature_extractors_demo")
    parser.add_argument("--list", action="store_true", help="List available feature extractors and exit")
    parser.add_argument("--name", default="hog", help="Feature extractor name (registry)")
    parser.add_argument(
        "--kwargs",
        default=None,
        help="Optional JSON object of extractor kwargs (e.g. '{\"resize_hw\": [64, 64]}')",
    )
    parser.add_argument("--n", type=int, default=4, help="Number of synthetic images to generate")
    parser.add_argument("--h", type=int, default=64, help="Synthetic image height")
    parser.add_argument("--w", type=int, default=64, help="Synthetic image width")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data")
    args = parser.parse_args(argv)

    from pyimgano.features import create_feature_extractor, list_feature_extractors

    if bool(args.list):
        for name in list_feature_extractors():
            print(name)
        return 0

    ext = create_feature_extractor(str(args.name), **_parse_kwargs(args.kwargs))

    rng = np.random.RandomState(int(args.seed))
    imgs = [(rng.rand(int(args.h), int(args.w), 3) * 255).astype(np.uint8) for _ in range(int(args.n))]

    fit = getattr(ext, "fit", None)
    if callable(fit):
        try:
            fit(imgs)
        except Exception:
            # Some extractors are stateless; ignore.
            pass

    feats = np.asarray(ext.extract(imgs))
    print(f"Extractor: {type(ext).__module__}.{type(ext).__qualname__}")
    print(f"Features shape: {feats.shape}")
    print(f"Features dtype: {feats.dtype}")
    print(f"Finite: {bool(np.all(np.isfinite(feats)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

