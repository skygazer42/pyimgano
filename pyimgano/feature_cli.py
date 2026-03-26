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


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    out: list[str] = []
    for item in tags:
        for piece in str(item).split(","):
            tag = piece.strip()
            if tag:
                out.append(tag)
    return out


def _resolve_manifest_paths(
    *,
    manifest_path: Path,
    category: str | None,
    split: str | None,
    label: int | None,
    root_fallback: Path | None,
) -> list[str]:
    from pyimgano.datasets.manifest import iter_manifest_records

    cat = None if category is None else str(category)
    split_f = None if split is None else str(split).strip().lower()
    label_f = None if label is None else int(label)

    out: list[str] = []
    for rec in iter_manifest_records(manifest_path):
        if cat is not None and str(rec.category) != cat:
            continue
        if split_f is not None:
            if rec.split is None or str(rec.split) != split_f:
                continue
        if label_f is not None:
            if rec.label is None or int(rec.label) != label_f:
                continue

        raw = str(rec.image_path)
        p = Path(raw)
        if p.is_absolute():
            out.append(str(p))
            continue

        # Resolve relative paths relative to the manifest file, then optionally
        # fall back to an explicit root.
        cand1 = (manifest_path.parent / p).resolve()
        if cand1.exists():
            out.append(str(cand1))
            continue

        if root_fallback is not None:
            cand2 = (root_fallback / p).resolve()
            if cand2.exists():
                out.append(str(cand2))
                continue

        raise FileNotFoundError(
            f"Manifest image_path not found: {raw!r}. Tried {cand1} and {cand2 if root_fallback else '<no root>'}."
        )

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pyimgano-features")

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--list-extractors",
        action="store_true",
        help="List available registered feature extractors and exit",
    )
    mode.add_argument(
        "--extractor-info",
        default=None,
        metavar="NAME",
        help="Show signature/metadata for one feature extractor and exit",
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Optional tags filter for --list-extractors (comma-separated or repeatable). "
            "Example: --tags texture"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON payload for --list-extractors/--extractor-info",
    )

    parser.add_argument("--root", required=False, help="Root directory containing images")
    parser.add_argument(
        "--pattern",
        default="**/*.*",
        help="Glob pattern relative to --root. Default: **/*.*",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSONL manifest path (enables manifest input mode; overrides --root/--pattern)",
    )
    parser.add_argument(
        "--manifest-category", default=None, help="Optional category filter for --manifest"
    )
    parser.add_argument(
        "--manifest-split",
        default=None,
        choices=["train", "val", "test"],
        help="Optional split filter for --manifest",
    )
    parser.add_argument(
        "--manifest-label",
        default=None,
        choices=["0", "1"],
        help="Optional label filter for --manifest (0 normal, 1 anomaly)",
    )
    parser.add_argument(
        "--manifest-root-fallback",
        default=None,
        help="Optional root fallback for resolving relative manifest paths",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Output .npy path for feature matrix (required unless using --list-extractors/--extractor-info)",
    )
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
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional directory for disk caching extracted features when inputs are paths. "
            "This speeds up repeated runs with the same extractor config."
        ),
    )
    try:
        args = parser.parse_args(argv)

        if bool(getattr(args, "list_extractors", False)):
            from pyimgano.features import list_feature_extractors

            tags = _normalize_tags(list(getattr(args, "tags", None) or [])) or None
            payload = list_feature_extractors(tags=tags)
            if bool(getattr(args, "json", False)):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                for name in payload:
                    print(name)
            return 0

        if getattr(args, "extractor_info", None) is not None:
            from pyimgano.features import feature_info

            payload = feature_info(str(getattr(args, "extractor_info")))
            if bool(getattr(args, "json", False)):
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"name: {payload.get('name')}")
                tags = payload.get("tags", None)
                if tags:
                    print(f"tags: {', '.join(str(t) for t in tags)}")
                meta = payload.get("metadata", None)
                if isinstance(meta, dict) and meta:
                    for k in sorted(meta):
                        print(f"{k}: {meta[k]}")
                if payload.get("signature", None) is not None:
                    print(f"signature: {payload.get('signature')}")
                accepted = payload.get("accepted_kwargs", None)
                if accepted:
                    print(f"accepted_kwargs: {', '.join(str(x) for x in accepted)}")
            return 0

        if args.output is None:
            raise ValueError(
                "--output is required unless using --list-extractors/--extractor-info."
            )

        from pyimgano.features import create_feature_extractor

        manifest_path = None if args.manifest is None else Path(str(args.manifest))
        if manifest_path is None:
            if args.root is None:
                raise ValueError("Provide either --root or --manifest.")
            root = Path(str(args.root))
            if not root.is_dir():
                raise ValueError(f"--root must be a directory, got {root}")
            paths = _collect_paths(root, pattern=str(args.pattern))
            if not paths:
                raise ValueError("No files matched --pattern under --root")
        else:
            if not manifest_path.exists():
                raise FileNotFoundError(f"--manifest not found: {manifest_path}")
            root_fb = (
                None
                if args.manifest_root_fallback is None
                else Path(str(args.manifest_root_fallback))
            )
            label = None if args.manifest_label is None else int(args.manifest_label)
            paths = _resolve_manifest_paths(
                manifest_path=manifest_path,
                category=args.manifest_category,
                split=args.manifest_split,
                label=label,
                root_fallback=root_fb,
            )
            if not paths:
                raise ValueError("No manifest records matched the provided filters.")

        extractor = create_feature_extractor(
            str(args.extractor), **_parse_kwargs(args.extractor_kwargs)
        )

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

        if args.cache_dir is not None:
            from pyimgano.cache.features import (
                CachedFeatureExtractor,
                FeatureCache,
                fingerprint_feature_extractor,
            )

            cache_root = Path(str(args.cache_dir))
            fp = fingerprint_feature_extractor(extractor)
            cache = FeatureCache(cache_dir=cache_root, extractor_fingerprint=fp)
            extractor = CachedFeatureExtractor(base_extractor=extractor, cache=cache)

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
    except (ValueError, FileNotFoundError) as exc:
        # CLI-friendly error reporting: show message and return a non-zero exit code.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
