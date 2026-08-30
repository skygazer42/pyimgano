from __future__ import annotations

"""Run the repository's reproducible MVTec AD golden benchmark slice."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyimgano.models import create_model
from pyimgano.pipelines.mvtec_visa import BenchmarkSplit, evaluate_split, load_benchmark_split

DEFAULT_CATEGORIES = ("bottle", "carpet")
DEFAULT_MODELS = ("patchcore", "padim", "openclip")
MODEL_NAMES = {
    "patchcore": "vision_patchcore",
    "padim": "vision_padim",
    "openclip": "vision_openclip_promptscore",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run-golden-mvtec-benchmark")
    parser.add_argument("--root", required=True, help="MVTec AD dataset root")
    parser.add_argument("--output-dir", default="runs/golden_mvtec")
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--models", nargs="+", choices=sorted(MODEL_NAMES), default=list(DEFAULT_MODELS)
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, nargs=2, default=(224, 224), metavar=("H", "W"))
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Acknowledge that official pretrained weights may be downloaded.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate dataset layout and write provenance without loading models.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use four train/test samples per category; results are not publication-ready.",
    )
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_split(
    split: BenchmarkSplit,
    *,
    root: Path,
    category: str,
) -> dict[str, Any]:
    mask_root = root / str(category) / "ground_truth"
    mask_paths = list(mask_root.rglob("*")) if mask_root.is_dir() else []
    paths = sorted(
        {
            *(Path(path).resolve() for path in split.train_paths),
            *(Path(path).resolve() for path in split.test_paths),
            *(path.resolve() for path in mask_paths if path.is_file()),
        },
        key=lambda path: str(path),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        file_digest = _sha256_file(path)
        size = int(path.stat().st_size)
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256(relative_path\\0size\\0file_sha256\\n)",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "train_count": len(split.train_paths),
        "test_count": len(split.test_paths),
    }


def _package_versions() -> dict[str, str | None]:
    names = (
        "pyimgano",
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "open-clip-torch",
    )
    resolved: dict[str, str | None] = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def _model_kwargs(name: str, *, category: str, device: str, seed: int) -> dict[str, Any]:
    common = {"contamination": 0.1, "device": device}
    if name == "patchcore":
        return {
            **common,
            "backbone": "wide_resnet50_2",
            "pretrained": True,
            "coreset_sampling_ratio": 0.1,
            "n_neighbors": 1,
            "knn_backend": "sklearn",
            "random_seed": int(seed),
        }
    if name == "padim":
        return {
            **common,
            "backbone": "resnet18",
            "pretrained": True,
            "d_reduced": 100,
            "image_size": 224,
            "random_state": int(seed),
        }
    if name == "openclip":
        return {
            **common,
            "class_name": category,
            "openclip_model_name": "ViT-B-32",
            "openclip_pretrained": "laion2b_s34b_b79k",
            "allow_download": True,
        }
    raise ValueError(f"Unsupported golden benchmark model: {name}")


def _limit_split(split: BenchmarkSplit, limit: int) -> BenchmarkSplit:
    labels = split.test_labels
    normal = [index for index, label in enumerate(labels) if int(label) == 0]
    anomaly = [index for index, label in enumerate(labels) if int(label) == 1]
    half = max(1, int(limit) // 2)
    selected = sorted((normal[:half] + anomaly[:half])[: int(limit)])
    test_masks = split.test_masks
    if test_masks is not None:
        test_masks = test_masks[selected]
    return BenchmarkSplit(
        train_paths=list(split.train_paths)[:limit],
        test_paths=[split.test_paths[index] for index in selected],
        test_labels=split.test_labels[selected],
        test_masks=test_masks,
    )


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _metric_row(
    category: str, model: str, results: dict[str, Any], elapsed: float
) -> dict[str, Any]:
    pixel = results.get("pixel_metrics") or {}
    metrics = results.get("metrics") or {}
    return {
        "category": category,
        "model": model,
        "image_auroc": results.get("auroc"),
        "image_average_precision": results.get("average_precision"),
        "image_f1": metrics.get("f1"),
        "pixel_auroc": pixel.get("pixel_auroc"),
        "pixel_average_precision": pixel.get("pixel_average_precision"),
        "aupro": pixel.get("aupro"),
        "elapsed_seconds": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"MVTec AD root not found: {root}")
    if not args.check_only and not args.allow_download:
        raise ValueError(
            "Golden benchmark uses official pretrained weights. Pass --allow-download "
            "after accepting the models' download/license terms."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resize = (int(args.resize[0]), int(args.resize[1]))

    splits: dict[str, BenchmarkSplit] = {}
    fingerprints: dict[str, Any] = {}
    for category in args.categories:
        split = load_benchmark_split(
            dataset="mvtec",
            root=str(root),
            category=str(category),
            resize=resize,
            load_masks=True,
        )
        fingerprints[str(category)] = _fingerprint_split(
            split,
            root=root,
            category=str(category),
        )
        splits[str(category)] = _limit_split(split, 4) if args.smoke else split

    provenance = {
        "schema_version": 1,
        "dataset": "MVTec AD",
        "dataset_root": str(root),
        "categories": [str(value) for value in args.categories],
        "models": [str(value) for value in args.models],
        "resize": list(resize),
        "seed": int(args.seed),
        "device": str(args.device),
        "smoke": bool(args.smoke),
        "dataset_fingerprints": fingerprints,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    (output_dir / "golden_manifest.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.check_only:
        print(json.dumps({"status": "ready", "manifest": str(output_dir / "golden_manifest.json")}))
        return 0

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for category, split in splits.items():
        reports[category] = {}
        for model_key in args.models:
            model_name = MODEL_NAMES[str(model_key)]
            _seed_everything(int(args.seed))
            kwargs = _model_kwargs(
                str(model_key),
                category=category,
                device=str(args.device),
                seed=int(args.seed),
            )
            started = time.perf_counter()
            detector = create_model(model_name, **kwargs)
            results = evaluate_split(
                detector,
                split,
                compute_pixel_scores=True,
                pro_integration_limit=0.3,
                pro_num_thresholds=200,
                score_calibration_quantile=0.9,
            )
            elapsed = float(time.perf_counter() - started)
            reports[category][str(model_key)] = _jsonable(results)
            rows.append(_metric_row(category, str(model_key), results, elapsed))

    summary = {
        **provenance,
        "publication_ready": not bool(args.smoke),
        "rows": rows,
        "reports": reports,
    }
    summary_path = output_dir / "golden_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_dir / "golden_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "summary": str(summary_path), "csv": str(csv_path)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
