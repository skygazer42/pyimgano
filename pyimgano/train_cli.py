from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pyimgano.config import load_config
from pyimgano.recipes.registry import list_recipes, recipe_info
from pyimgano.reporting.report import save_run_report
from pyimgano.workbench.config import WorkbenchConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-train")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-recipes", action="store_true", help="List available recipe names")
    mode.add_argument("--recipe-info", default=None, help="Show recipe metadata for a recipe name")
    mode.add_argument(
        "--config", default=None, help="Path to a workbench config (JSON, optional YAML)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --list-recipes/--recipe-info, output JSON instead of text",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the effective config JSON without running the recipe",
    )
    parser.add_argument(
        "--export-infer-config",
        action="store_true",
        help="Write artifacts/infer_config.json to the run directory (requires output.save_run=true)",
    )

    # Optional overrides (applied after loading --config).
    parser.add_argument("--dataset", default=None, help="Override dataset.name from config")
    parser.add_argument("--root", default=None, help="Override dataset.root from config")
    parser.add_argument("--category", default=None, help="Override dataset.category from config")
    parser.add_argument("--model", default=None, help="Override model.name from config")
    parser.add_argument("--device", default=None, help="Override model.device from config")

    return parser


def _apply_overrides(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(raw)

    dataset = dict(out.get("dataset", {}) or {})
    model = dict(out.get("model", {}) or {})

    if args.dataset is not None:
        dataset["name"] = str(args.dataset)
    if args.root is not None:
        dataset["root"] = str(args.root)
    if args.category is not None:
        dataset["category"] = str(args.category)

    if args.model is not None:
        model["name"] = str(args.model)
    if args.device is not None:
        model["device"] = str(args.device)

    if dataset:
        out["dataset"] = dataset
    if model:
        out["model"] = model

    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # Import builtin recipes for side effects (registration).
        import pyimgano.recipes  # noqa: F401

        if bool(args.list_recipes):
            names = list_recipes()
            if bool(args.json):
                print(json.dumps(names, sort_keys=True))
            else:
                for name in names:
                    print(name)
            return 0

        if args.recipe_info is not None:
            info = recipe_info(str(args.recipe_info))
            if bool(args.json):
                print(json.dumps(info, sort_keys=True))
            else:
                print(f"Recipe: {info.get('name')}")
                tags = info.get("tags", [])
                if tags:
                    print(f"Tags: {', '.join(str(t) for t in tags)}")
                meta = info.get("metadata", {})
                if meta:
                    print("Metadata:")
                    for k in sorted(meta):
                        print(f"  {k}: {meta[k]}")
            return 0

        if args.config is None:
            raise ValueError("--config is required to run a recipe")

        config_path = Path(str(args.config))
        raw = load_config(config_path)
        raw = _apply_overrides(raw, args)
        cfg = WorkbenchConfig.from_dict(raw)

        from pyimgano.recipes.registry import RECIPE_REGISTRY

        recipe = RECIPE_REGISTRY.get(cfg.recipe)
        if bool(args.dry_run):
            if str(cfg.dataset.name).lower() == "manifest":
                mp_raw = cfg.dataset.manifest_path
                mp = Path(str(mp_raw)) if mp_raw is not None else None
                if mp is None:
                    raise ValueError(
                        "dataset.manifest_path is required when dataset.name='manifest'."
                    )
                if not mp.exists():
                    raise ValueError(f"dataset.manifest_path not found: {mp}")
                if not mp.is_file():
                    raise ValueError(f"dataset.manifest_path must be a file: {mp}")
                try:
                    with mp.open("r", encoding="utf-8") as f:
                        f.read(1)
                except Exception as exc:  # noqa: BLE001 - validation boundary
                    raise ValueError(f"dataset.manifest_path not readable: {mp}") from exc

            # Emit the canonical config payload shape used by workbench artifacts.
            print(json.dumps({"config": asdict(cfg)}, sort_keys=True))
            return 0
        report = recipe(cfg)

        if bool(args.export_infer_config):
            if not bool(cfg.output.save_run):
                raise ValueError("--export-infer-config requires output.save_run=true.")
            run_dir_raw = report.get("run_dir", None)
            if run_dir_raw is None:
                raise ValueError("--export-infer-config requires recipe output to include run_dir.")
            run_dir = Path(str(run_dir_raw))
            infer_config_path = run_dir / "artifacts" / "infer_config.json"

            from pyimgano.workbench.runner import build_infer_config_payload

            payload = build_infer_config_payload(config=cfg, report=report)
            save_run_report(infer_config_path, payload)

        print(json.dumps(report, sort_keys=True))
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
