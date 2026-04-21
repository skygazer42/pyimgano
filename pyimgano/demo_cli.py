from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyimgano.utils.jsonable import to_jsonable

_DEMO_SCENARIOS = {"smoke", "benchmark", "infer-defects"}


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
        "--scenario",
        default=None,
        choices=sorted(_DEMO_SCENARIOS),
        help="Named demo scenario preset: smoke, benchmark, or infer-defects.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Force a lightweight CPU-friendly smoke path with bounded defaults.",
    )
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
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write a compact demo summary JSON payload.",
    )
    parser.add_argument(
        "--emit-next-steps",
        action="store_true",
        help="Print a small copy-pasteable next-step command block after the demo completes.",
    )
    parser.add_argument(
        "--infer-defects",
        action="store_true",
        help=(
            "After the suite run, run a one-command inference + defects loop and write artifacts under "
            "<suite-run-dir>/infer/ (results.jsonl + masks/ + overlays/ + regions.jsonl)."
        ),
    )
    return parser


def _resolve_demo_scenario(args: argparse.Namespace) -> dict[str, Any]:
    scenario = getattr(args, "scenario", None)
    if scenario is None:
        if bool(getattr(args, "smoke", False)):
            scenario = "smoke"
        elif bool(getattr(args, "infer_defects", False)):
            scenario = "infer-defects"
        else:
            scenario = "custom"

    scenario_key = str(scenario)
    if scenario_key == "smoke":
        return {
            "scenario": scenario_key,
            "smoke": True,
            "resize": (32, 32),
            "suite": "industrial-ci",
            "sweep": None,
            "sweep_max_variants": 0,
            "export_mode": "csv",
            "limit_train": 2,
            "limit_test": 2,
            "infer_defects": False,
        }
    if scenario_key == "benchmark":
        return {
            "scenario": scenario_key,
            "smoke": False,
            "resize": (32, 32),
            "suite": "industrial-ci",
            "sweep": None,
            "sweep_max_variants": 0,
            "export_mode": "csv",
            "limit_train": 2,
            "limit_test": 2,
            "infer_defects": False,
        }
    if scenario_key == "infer-defects":
        return {
            "scenario": scenario_key,
            "smoke": False,
            "resize": (32, 32),
            "suite": "industrial-ci",
            "sweep": None,
            "sweep_max_variants": 0,
            "export_mode": "none",
            "limit_train": 2,
            "limit_test": 2,
            "infer_defects": True,
        }
    return {
        "scenario": scenario_key,
        "smoke": bool(getattr(args, "smoke", False)),
        "resize": (int(args.resize[0]), int(args.resize[1])),
        "suite": str(args.suite),
        "sweep": None if bool(getattr(args, "no_sweep", False)) else str(args.sweep),
        "sweep_max_variants": int(args.sweep_max_variants),
        "export_mode": str(args.export),
        "limit_train": int(args.limit_train),
        "limit_test": int(args.limit_test),
        "infer_defects": bool(getattr(args, "infer_defects", False)),
    }


def _build_next_steps(
    *,
    dataset_root: Path,
    run_dir: Path | None,
    scenario: str,
    infer_dir: Path | None,
) -> list[str]:
    if str(scenario) == "benchmark":
        steps = [
            "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
        ]
        if run_dir is not None:
            steps.extend(
                [
                    f"pyimgano runs quality {run_dir} --json",
                    f"pyimgano runs publication {run_dir} --json",
                ]
            )
        return steps

    if str(scenario) == "infer-defects":
        steps = []
        if run_dir is not None:
            rerun_path = (
                (infer_dir / "rerun_results.jsonl")
                if infer_dir is not None
                else (run_dir / "rerun_results.jsonl")
            )
            steps.append(
                "pyimgano-infer "
                f"--from-run {run_dir} --input {dataset_root / 'test'} --save-jsonl {rerun_path}"
            )
            steps.extend(
                [
                    f"pyimgano runs quality {run_dir} --json",
                    f"pyimgano runs acceptance {run_dir} --require-status audited --json",
                ]
            )
        return steps

    steps = [
        (
            "pyimgano-infer "
            f"--model-preset industrial-template-ncc-map --train-dir {dataset_root / 'train' / 'normal'} "
            f"--input {dataset_root / 'test'} --defects-preset industrial-defects-fp40 "
            f"--save-jsonl {dataset_root.parent / '_demo_results.jsonl'}"
        ),
        "pyimgano benchmark --list-starter-configs",
    ]
    if run_dir is not None:
        steps.append(f"pyimgano runs quality {run_dir} --json")
    return steps


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    dataset_root = Path(str(args.dataset_root))
    scenario_config = _resolve_demo_scenario(args)
    scenario = str(scenario_config["scenario"])
    smoke = bool(scenario_config["smoke"])
    resize = tuple(int(v) for v in scenario_config["resize"])
    _write_demo_custom_dataset(dataset_root, size_hw=resize)

    output_dir: Path | None
    if bool(args.save_run):
        output_dir = _ensure_unique_output_dir(Path(str(args.output_dir)))
    else:
        output_dir = None
        if str(args.export).lower().strip() != "none":
            raise ValueError("--export requires --save-run.")

    suite = str(scenario_config["suite"])
    sweep = scenario_config["sweep"]
    sweep_max_variants = int(scenario_config["sweep_max_variants"])
    export_mode = str(scenario_config["export_mode"])
    limit_train = int(scenario_config["limit_train"])
    limit_test = int(scenario_config["limit_test"])

    from pyimgano.pipelines.run_suite import run_baseline_suite

    payload = run_baseline_suite(
        suite=suite,
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
        limit_train=limit_train,
        limit_test=limit_test,
        save_run=bool(args.save_run),
        per_image_jsonl=True,
        cache_dir=None,
        output_dir=(output_dir if output_dir is not None else None),
        continue_on_error=True,
        sweep=sweep,
        sweep_max_variants=sweep_max_variants,
    )

    infer_defects_payload: dict[str, Any] | None = None
    if bool(scenario_config["infer_defects"]):
        if not bool(args.save_run):
            raise ValueError("--infer-defects requires --save-run.")

        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            raise RuntimeError("internal error: expected suite run_dir for --infer-defects.")

        infer_dir = Path(run_dir) / "infer"
        masks_dir = infer_dir / "masks"
        overlays_dir = infer_dir / "overlays"
        regions_jsonl = infer_dir / "regions.jsonl"
        infer_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        from pyimgano.infer_cli import main as infer_main

        infer_model_kwargs = {"resize_hw": [int(resize[0]), int(resize[1])]}
        infer_argv = [
            "--model-preset",
            "industrial-template-ncc-map",
            "--model-kwargs",
            json.dumps(infer_model_kwargs, sort_keys=True),
            "--device",
            str(args.device),
            "--train-dir",
            str(dataset_root / "train" / "normal"),
            "--input",
            str(dataset_root / "test"),
            "--defects-preset",
            "industrial-defects-fp40",
            "--save-jsonl",
            str(infer_dir / "results.jsonl"),
            "--save-masks",
            str(masks_dir),
            "--defects-regions-jsonl",
            str(regions_jsonl),
            "--save-overlays",
            str(overlays_dir),
        ]
        infer_argv.append("--pretrained" if bool(args.pretrained) else "--no-pretrained")

        infer_rc = int(infer_main(infer_argv))
        infer_defects_payload = {
            "ok": bool(infer_rc == 0),
            "rc": int(infer_rc),
            "infer_dir": str(infer_dir),
            "results_jsonl": str(infer_dir / "results.jsonl"),
            "regions_jsonl": str(regions_jsonl),
            "masks_dir": str(masks_dir),
            "overlays_dir": str(overlays_dir),
            "model_preset": "industrial-template-ncc-map",
            "defects_preset": "industrial-defects-fp40",
        }
        payload["infer_defects"] = infer_defects_payload
        if infer_rc != 0:
            return int(infer_rc)

    exported: dict[str, str] | None = None
    if bool(args.save_run) and str(export_mode).lower().strip() != "none":
        from pyimgano.reporting.suite_export import export_suite_tables

        run_dir = payload.get("run_dir")
        if not isinstance(run_dir, str) or not run_dir:
            raise RuntimeError("internal error: expected suite run_dir when --save-run is enabled.")
        fmt = str(export_mode).lower().strip()
        formats = ["csv", "md"] if fmt == "both" else [fmt]
        exported = export_suite_tables(payload, Path(run_dir), formats=formats)

    run_dir = payload.get("run_dir")
    run_dir_path = Path(run_dir) if isinstance(run_dir, str) and run_dir else None
    infer_dir_path = (
        Path(str(infer_defects_payload["infer_dir"]))
        if isinstance(infer_defects_payload, dict) and infer_defects_payload.get("infer_dir")
        else None
    )
    next_steps = _build_next_steps(
        dataset_root=dataset_root,
        run_dir=run_dir_path,
        scenario=scenario,
        infer_dir=infer_dir_path,
    )
    summary_payload = {
        "scenario": scenario,
        "smoke": smoke,
        "dataset_root": str(dataset_root),
        "run_dir": (str(run_dir_path) if run_dir_path is not None else None),
        "exported": (dict(exported) if exported is not None else {}),
        "next_steps": next_steps,
    }

    summary_json = getattr(args, "summary_json", None)
    if summary_json is not None:
        Path(str(summary_json)).write_text(
            json.dumps(to_jsonable(summary_payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if bool(args.json):
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
        return 0

    print(f"Demo dataset: {dataset_root.resolve()}")
    if isinstance(run_dir, str) and run_dir:
        print(f"Suite run dir: {Path(run_dir).resolve()}")
    if exported:
        for k in sorted(exported):
            print(f"{k}: {exported[k]}")
    else:
        print("Export: <disabled>")

    if infer_defects_payload is not None:
        print(f"Infer+defects artifacts: {Path(str(infer_defects_payload['infer_dir'])).resolve()}")

    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        by_auroc = summary.get("by_auroc", None)
        if isinstance(by_auroc, list) and by_auroc:
            top = by_auroc[0]
            if isinstance(top, dict):
                name = top.get("name")
                auroc = top.get("auroc")
                print(f"Top by AUROC: {name} (auroc={auroc})")

    if bool(getattr(args, "emit_next_steps", False)):
        print("Next steps:")
        for step in next_steps:
            print(step)

    print("Tip: remove ./_demo_custom_dataset and ./_demo_suite_run* when done.")
    return 0


__all__ = [
    "main",
]
