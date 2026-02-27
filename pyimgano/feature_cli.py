from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parse_kwargs(text: str | None) -> dict[str, Any]:
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--extractor-kwargs must be valid JSON. Original error: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--extractor-kwargs must be a JSON object")
    return dict(parsed)


def _collect_paths(root: Path, *, pattern: str) -> list[str]:
    paths = sorted(str(p) for p in root.glob(pattern) if p.is_file())
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyimgano-features")
    parser.add_argument("--root", required=True, help="Root directory containing images")
    parser.add_argument(
        "--pattern",
        default="**/*.*",
        help="Glob pattern relative to --root. Default: **/*.*",
    )
    parser.add_argument("--output", required=True, help="Output .npy path for feature matrix")
    parser.add_argument(
        "--paths-json",
        default=None,
        help="Optional output .json path to store the ordered input paths",
    )
    parser.add_argument("--extractor", default="hog", help="Registered feature extractor name")
    parser.add_argument(
        "--extractor-kwargs",
        default=None,
        help="Optional JSON object of extractor kwargs",
    )
    parser.add_argument(
        "--fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call extractor.fit() before extract() when available. Default: true",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Optional device hint for extractors that support it (e.g. torch extractors). "
            "Example: cpu|cuda. When provided, sets extractor.device if present."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional extraction batch size. When provided and the extractor supports it, "
            "uses extractor.extract_batched(..., batch_size=N)."
        ),
    )
    args = parser.parse_args(argv)

    from pyimgano.features import create_feature_extractor

    root = Path(str(args.root))
    if not root.is_dir():
        raise ValueError(f"--root must be a directory, got {root}")

    paths = _collect_paths(root, pattern=str(args.pattern))
    if not paths:
        raise ValueError("No files matched --pattern under --root")

    extractor = create_feature_extractor(str(args.extractor), **_parse_kwargs(args.extractor_kwargs))

    # Best-effort: allow setting a common `.device` attribute without forcing every
    # extractor to implement a full config protocol.
    if args.device is not None and hasattr(extractor, "device"):
        try:
            setattr(extractor, "device", str(args.device))
        except Exception:
            pass

    if bool(args.fit):
        fit = getattr(extractor, "fit", None)
        if callable(fit):
            fit(paths)

    if args.batch_size is not None:
        extract_batched = getattr(extractor, "extract_batched", None)
        if callable(extract_batched):
            feats = np.asarray(extract_batched(paths, batch_size=int(args.batch_size)))
        else:
            feats = np.asarray(extractor.extract(paths))
    else:
        feats = np.asarray(extractor.extract(paths))
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    if feats.shape[0] != len(paths):
        raise ValueError("extractor.extract must return one row per input path")

    out_path = Path(str(args.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), feats, allow_pickle=False)

    if args.paths_json is not None:
        pj = Path(str(args.paths_json))
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text(json.dumps(paths, indent=2), encoding="utf-8")

    print(f"Wrote features: {out_path} shape={feats.shape}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
