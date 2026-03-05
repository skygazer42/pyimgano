from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyimgano.utils.jsonable import to_jsonable


def _ensure_unique_output_dir(base: Path) -> Path:
    base = Path(base)
    if base.exists() and not base.is_dir():
        raise ValueError(f"--output-dir must be a directory path. Got: {base}")
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    # Avoid mixing artifacts across runs.
    for i in range(1, 1000):
        cand = base.parent / f"{base.name}_{i:03d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand
    raise RuntimeError(f"Failed to create a unique output dir under: {base}")


def _write_demo_custom_dataset(root: Path, *, size_hw: tuple[int, int]) -> None:
    # Local imports keep module import light; these are core deps.
    import cv2
    import numpy as np

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    h, w = int(size_hw[0]), int(size_hw[1])
    for rel, value in [
        ("train/normal/train_0.png", 120),
        ("train/normal/train_1.png", 120),
        ("test/normal/good_0.png", 120),
        ("test/anomaly/bad_0.png", 240),
    ]:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        img = np.ones((h, w, 3), dtype=np.uint8) * int(value)
        ok = cv2.imwrite(str(p), img)
        if not ok:
            raise RuntimeError(f"Failed to write demo image: {p}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-demo")
    parser.add_argument(
        "--dataset-root",
        default="./_demo_custom_dataset",
        help="Where to write a tiny demo custom dataset (default: ./_demo_custom_dataset).",
    )
    parser.add_argument(
        "--suite",
        default="industrial-ci",
        help="Suite name to run (default: industrial-ci).",
    )
    parser.add_argument(
        "--sweep",
        default="industrial-small",
        help="Sweep profile name (default: industrial-small).",
    )
    parser.add_argument(
        "--no-sweep",
        action="store_true",
        help="Disable sweep (run suite baselines only).",
    )
    parser.add_argument(
        "--sweep-max-variants",
        type=int,
        default=1,
        help="Cap sweep variants per baseline (excluding base). Default: 1.",
    )
    parser.add_argument(
        "--output-dir",
        default="./_demo_suite_run",
        help=(
            "Where to write suite run artifacts (report.json, leaderboard.*). "
            "If the directory exists, a numeric suffix is added to avoid mixing runs. "
            "Default: ./_demo_suite_run"
        ),
    )
    parser.add_argument(
        "--export",
        default="csv",
        choices=["csv", "md", "both", "none"],
        help="Export suite tables into output dir (default: csv).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for models that support it (default: cpu).",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(64, 64),
        metavar=("H", "W"),
        help="Resize used for demo dataset + suite run. Default: 64 64.",
    )
    parser.add_argument(
        "--limit-train", type=int, default=2, help="Limit number of train images. Default: 2."
    )
    parser.add_argument(
        "--limit-test", type=int, default=2, help="Limit number of test images. Default: 2."
    )
    parser.add_argument(
        "--save-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write artifacts to --output-dir (default: true).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full suite JSON payload to stdout (default: false).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dataset_root = Path(str(args.dataset_root))
    resize = (int(args.resize[0]), int(args.resize[1]))
    _write_demo_custom_dataset(dataset_root, size_hw=resize)

    output_dir: Path | None
    if bool(args.save_run):
        output_dir = _ensure_unique_output_dir(Path(str(args.output_dir)))
    else:
        output_dir = None
        if str(args.export).lower().strip() != "none":
            raise ValueError("--export requires --save-run.")

    sweep: str | None = None if bool(args.no_sweep) else str(args.sweep)

    from pyimgano.pipelines.run_suite import run_baseline_suite

    payload = run_baseline_suite(
        suite=str(args.suite),
        dataset="custom",
        root=str(dataset_root),
        manifest_path=None,
        category="custom",
        input_mode="paths",
        seed=0,
        device=str(args.device),
        pretrained=bool(args.pretrained),
        contamination=0.1,
        resize=resize,
        limit_train=int(args.limit_train),
        limit_test=int(args.limit_test),
        save_run=bool(args.save_run),
        per_image_jsonl=True,
        cache_dir=None,
        output_dir=(output_dir if output_dir is not None else None),
        continue_on_error=True,
        sweep=sweep,
        sweep_max_variants=int(args.sweep_max_variants),
    )

    exported: dict[str, str] | None = None
    if bool(args.save_run) and str(args.export).lower().strip() != "none":
        from pyimgano.reporting.suite_export import export_suite_tables

        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            raise RuntimeError("internal error: expected suite run_dir when --save-run is enabled.")
        fmt = str(args.export).lower().strip()
        formats = ["csv", "md"] if fmt == "both" else [fmt]
        exported = export_suite_tables(payload, Path(run_dir), formats=formats)

    if bool(args.json):
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        return 0

    run_dir = payload.get("run_dir")
    print(f"Demo dataset: {dataset_root.resolve()}")
    if isinstance(run_dir, str) and run_dir:
        print(f"Suite run dir: {Path(run_dir).resolve()}")
    if exported:
        for k in sorted(exported):
            print(f"{k}: {exported[k]}")
    else:
        print("Export: <disabled>")

    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        by_auroc = summary.get("by_auroc", None)
        if isinstance(by_auroc, list) and by_auroc:
            top = by_auroc[0]
            if isinstance(top, dict):
                name = top.get("name")
                auroc = top.get("auroc")
                print(f"Top by AUROC: {name} (auroc={auroc})")

    print("Tip: remove ./_demo_custom_dataset and ./_demo_suite_run* when done.")
    return 0


__all__ = [
    "main",
]
