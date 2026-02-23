from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyimgano.config import load_config
from pyimgano.recipes.registry import list_recipes, recipe_info
from pyimgano.workbench.config import WorkbenchConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-train")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list-recipes", action="store_true", help="List available recipe names")
    mode.add_argument("--recipe-info", default=None, help="Show recipe metadata for a recipe name")
    mode.add_argument("--config", default=None, help="Path to a workbench config (JSON, optional YAML)")

    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --list-recipes/--recipe-info, output JSON instead of text",
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
        report = recipe(cfg)
        print(json.dumps(report, sort_keys=True))
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

