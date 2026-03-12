from __future__ import annotations

import argparse
import pyimgano.cli_listing as cli_listing
import pyimgano.cli_output as cli_output

from pyimgano.recipes.registry import list_recipes, recipe_info
import pyimgano.services.train_service as train_service


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
        "--preflight",
        action="store_true",
        help="Validate dataset health and print a preflight JSON report without running the recipe",
    )
    parser.add_argument(
        "--export-infer-config",
        action="store_true",
        help="Write artifacts/infer_config.json to the run directory (requires output.save_run=true)",
    )
    parser.add_argument(
        "--export-deploy-bundle",
        action="store_true",
        help=(
            "Create a deploy_bundle/ directory under the run directory containing "
            "infer_config.json + referenced checkpoints + (best-effort) metadata. "
            "Requires output.save_run=true."
        ),
    )

    # Optional overrides (applied after loading --config).
    parser.add_argument("--dataset", default=None, help="Override dataset.name from config")
    parser.add_argument("--root", default=None, help="Override dataset.root from config")
    parser.add_argument("--category", default=None, help="Override dataset.category from config")
    parser.add_argument("--model", default=None, help="Override model.name from config")
    parser.add_argument("--device", default=None, help="Override model.device from config")
    parser.add_argument(
        "--preprocessing-preset",
        default=None,
        help=(
            "Override preprocessing with a deployable preset. "
            "Use `pyim --list preprocessing --deployable-only` to discover names."
        ),
    )

    return parser

def _build_train_request(args: argparse.Namespace) -> train_service.TrainRunRequest:
    if args.config is None:
        raise ValueError("--config is required to run a recipe")

    return train_service.TrainRunRequest(
        config_path=str(args.config),
        dataset_name=(str(args.dataset) if args.dataset is not None else None),
        root=(str(args.root) if args.root is not None else None),
        category=(str(args.category) if args.category is not None else None),
        model_name=(str(args.model) if args.model is not None else None),
        device=(str(args.device) if args.device is not None else None),
        preprocessing_preset=(
            str(args.preprocessing_preset) if args.preprocessing_preset is not None else None
        ),
        export_infer_config=bool(args.export_infer_config),
        export_deploy_bundle=bool(args.export_deploy_bundle),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # Import builtin recipes for side effects (registration).
        import pyimgano.recipes  # noqa: F401

        if bool(args.list_recipes):
            names = list_recipes()
            return cli_listing.emit_listing(names, json_output=bool(args.json))

        if args.recipe_info is not None:
            info = recipe_info(str(args.recipe_info))
            if bool(args.json):
                return cli_output.emit_json(info)
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

        request = _build_train_request(args)
        if bool(args.preflight):
            if bool(args.dry_run):
                raise ValueError("--preflight and --dry-run are mutually exclusive.")
            payload = train_service.run_train_preflight_payload(request)
            cli_output.emit_json(payload)
            issues = payload.get("preflight", {}).get("issues", [])
            has_error = any(str(issue.get("severity")) == "error" for issue in issues)
            return 2 if has_error else 0

        if bool(args.dry_run):
            payload = train_service.build_train_dry_run_payload(request)
            return cli_output.emit_json(payload)
        report = train_service.run_train_request(request)

        return cli_output.emit_json(report)

    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cli_output.print_cli_error(exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
