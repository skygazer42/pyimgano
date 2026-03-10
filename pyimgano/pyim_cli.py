"""Unified discovery shortcut CLI for pyimgano."""

from __future__ import annotations

import argparse
import json
from typing import Any

from pyimgano.cli_presets import (
    list_defects_presets,
    list_model_preset_infos,
    list_model_presets,
)
from pyimgano.discovery import (
    list_feature_names,
    list_model_families,
    list_model_names,
    list_model_types,
    list_model_years,
    list_preprocessing_schemes,
    resolve_family_tags,
    resolve_type_tags,
    resolve_year_filter,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyim",
        description="Unified discovery shortcut for models, families, presets, and preprocessing.",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const="all",
        choices=(
            "all",
            "models",
            "families",
            "types",
            "years",
            "metadata-contract",
            "features",
            "model-presets",
            "defects-presets",
            "preprocessing",
            "recipes",
            "datasets",
        ),
        help=(
            "List available items. Default with no value: all. "
            "Examples: --list, --list models, --list preprocessing"
        ),
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Optional tags filter for model/feature discovery (comma-separated or repeatable). "
            "Example: --tags vision,deep"
        ),
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Optional model family/tag filter used with --list models. Example: --family patchcore",
    )
    parser.add_argument(
        "--type",
        dest="algorithm_type",
        default=None,
        help="Optional high-level model type/tag filter used with --list models. Example: --type deep-vision",
    )
    parser.add_argument(
        "--year",
        default=None,
        help="Optional publication year filter used with --list models. Example: --year 2021",
    )
    parser.add_argument(
        "--deployable-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When listing preprocessing schemes, include only deployable infer/workbench presets.",
    )
    parser.add_argument(
        "--audit-metadata",
        action="store_true",
        help="Audit registry models against the metadata contract and exit.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser


def _emit_json(payload: Any) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _print_named_block(title: str, lines: list[str]) -> None:
    print(title)
    for line in lines:
        print(line)
    print()


def _render_families_text() -> None:
    families = list_model_families()
    lines = [
        f"{item['name']} ({item['model_count']}): {item['description']}" for item in families
    ]
    _print_named_block("Families", lines)


def _render_types_text() -> None:
    types = list_model_types()
    lines = [f"{item['name']} ({item['model_count']}): {item['description']}" for item in types]
    _print_named_block("Types", lines)


def _render_years_text() -> None:
    years = list_model_years()
    lines = [f"{item['name']} ({item['model_count']}): {item['description']}" for item in years]
    _print_named_block("Years", lines)


def _render_metadata_contract_text(contract: list[dict[str, Any]]) -> None:
    lines = []
    for item in contract:
        rule = str(item["requirement"])
        if item.get("required_when"):
            rule = f"{rule}; {item['required_when']}"
        lines.append(f"{item['name']} [{item['source']} / {rule}]: {item['description']}")
    _print_named_block("Metadata Contract", lines)


def _render_preprocessing_text(*, deployable_only: bool) -> None:
    schemes = list_preprocessing_schemes(deployable_only=deployable_only)
    lines = []
    for item in schemes:
        suffix = " [deployable]" if bool(item.get("deployable", False)) else ""
        lines.append(f"{item['name']}{suffix}: {item['description']}")
    _print_named_block("Preprocessing Schemes", lines)


def _render_names_text(title: str, items: list[str]) -> None:
    _print_named_block(title, items)


def _normalize_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    out: list[str] = []
    for item in tags:
        for piece in str(item).split(","):
            t = piece.strip()
            if t:
                out.append(t)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list is None and not bool(args.audit_metadata):
        parser.print_help()
        return 0

    from pyimgano.models.registry import audit_model_metadata, model_metadata_contract

    if bool(args.audit_metadata):
        payload = audit_model_metadata()
        has_issues = bool(
            payload["summary"]["models_with_required_issues"]
            or payload["summary"]["models_with_recommended_issues"]
            or payload["summary"]["models_with_invalid_fields"]
        )
        if args.json:
            return _emit_json(payload) if not has_issues else (_emit_json(payload) or 1)

        print("Metadata Audit")
        print(
            f"required={payload['summary']['models_with_required_issues']} "
            f"recommended={payload['summary']['models_with_recommended_issues']} "
            f"invalid={payload['summary']['models_with_invalid_fields']}"
        )
        return 1 if has_issues else 0

    list_kind = str(args.list)
    if args.family is not None and list_kind not in {"all", "models", "model-presets"}:
        parser.error("--family is supported only with --list models, --list model-presets, or --list.")
    if args.algorithm_type is not None and list_kind != "models":
        parser.error("--type is supported only with --list models.")
    if args.year is not None and list_kind != "models":
        parser.error("--year is supported only with --list models.")
    if bool(args.deployable_only) and list_kind not in {"all", "preprocessing"}:
        parser.error("--deployable-only is supported only with --list preprocessing or --list.")

    if args.family is not None:
        try:
            resolve_family_tags(str(args.family))
        except KeyError as exc:
            parser.error(str(exc))
    if args.algorithm_type is not None:
        try:
            resolve_type_tags(str(args.algorithm_type))
        except KeyError as exc:
            parser.error(str(exc))
    if args.year is not None:
        try:
            resolve_year_filter(str(args.year))
        except KeyError as exc:
            parser.error(str(exc))

    if list_kind == "recipes":
        # Builtin recipes are registered on import.
        import pyimgano.recipes  # noqa: F401

        from pyimgano.recipes.registry import list_recipes as _list_recipes
        from pyimgano.recipes.registry import recipe_info as _recipe_info

        tag_filter = _normalize_tags(list(args.tags or [])) or None
        names = _list_recipes(tags=tag_filter)
        infos = [_recipe_info(name) for name in names]
        if args.json:
            return _emit_json(infos)

        lines = []
        for info in infos:
            meta = info.get("metadata", {}) or {}
            desc = str(meta.get("description", "")).strip()
            suffix = f": {desc}" if desc else ""
            lines.append(f"{info.get('name')}{suffix}")
        _print_named_block("Recipes", lines)
        return 0

    if list_kind == "datasets":
        from pyimgano.datasets.converters import list_dataset_converters

        payload = [
            {
                "name": str(c.name),
                "description": str(c.description),
                "requires_category": bool(c.requires_category),
            }
            for c in list_dataset_converters()
        ]
        if args.json:
            return _emit_json(payload)

        lines = []
        for c in payload:
            req = " (category required)" if bool(c.get("requires_category", False)) else ""
            lines.append(f"{c['name']}{req}: {c['description']}")
        _print_named_block("Datasets", lines)
        return 0

    preset_tags = list(args.tags or [])
    if args.family is not None:
        preset_tags.extend(resolve_family_tags(str(args.family)))

    models = list_model_names(
        tags=args.tags,
        family=args.family,
        algorithm_type=args.algorithm_type,
        year=args.year,
    )
    families = list_model_families()
    types = list_model_types()
    years = list_model_years()
    contract = model_metadata_contract()
    preprocessing = list_preprocessing_schemes(deployable_only=bool(args.deployable_only))
    features = list_feature_names(tags=args.tags)
    model_presets = list_model_presets(tags=preset_tags or None)
    model_preset_infos = list_model_preset_infos(tags=preset_tags or None)
    defects_presets = list_defects_presets()

    # Recipes/datasets are lightweight but still imported lazily to keep
    # non-all list operations fast and dependency-safe.
    recipes: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    if list_kind == "all":
        import pyimgano.recipes  # noqa: F401

        from pyimgano.recipes.registry import list_recipes as _list_recipes
        from pyimgano.recipes.registry import recipe_info as _recipe_info

        tag_filter = _normalize_tags(list(args.tags or [])) or None
        recipe_names = _list_recipes(tags=tag_filter)
        recipes = [_recipe_info(name) for name in recipe_names]

        from pyimgano.datasets.converters import list_dataset_converters

        datasets = [
            {
                "name": str(c.name),
                "description": str(c.description),
                "requires_category": bool(c.requires_category),
            }
            for c in list_dataset_converters()
        ]

    if list_kind == "models":
        if args.json:
            return _emit_json(models)
        _render_names_text("Models", models)
        return 0

    if list_kind == "families":
        if args.json:
            return _emit_json(families)
        _render_families_text()
        return 0

    if list_kind == "types":
        if args.json:
            return _emit_json(types)
        _render_types_text()
        return 0

    if list_kind == "years":
        if args.json:
            return _emit_json(years)
        _render_years_text()
        return 0

    if list_kind == "metadata-contract":
        if args.json:
            return _emit_json(contract)
        _render_metadata_contract_text(contract)
        return 0

    if list_kind == "features":
        if args.json:
            return _emit_json(features)
        _render_names_text("Feature Extractors", features)
        return 0

    if list_kind == "model-presets":
        if args.json:
            return _emit_json(model_preset_infos)
        _render_names_text("Model Presets", model_presets)
        return 0

    if list_kind == "defects-presets":
        if args.json:
            return _emit_json(defects_presets)
        _render_names_text("Defects Presets", defects_presets)
        return 0

    if list_kind == "preprocessing":
        if args.json:
            return _emit_json(preprocessing)
        _render_preprocessing_text(deployable_only=bool(args.deployable_only))
        return 0

    payload = {
        "models": models,
        "families": families,
        "types": types,
        "years": years,
        "metadata_contract": contract,
        "preprocessing": preprocessing,
        "features": features,
        "model_presets": model_presets,
        "defects_presets": defects_presets,
        "recipes": recipes,
        "datasets": datasets,
    }
    if args.json:
        return _emit_json(payload)

    _render_names_text("Models", models)
    _render_families_text()
    _render_types_text()
    _render_years_text()
    _render_metadata_contract_text(contract)
    _render_preprocessing_text(deployable_only=bool(args.deployable_only))
    _render_names_text("Feature Extractors", features)
    _render_names_text("Model Presets", model_presets)
    _render_names_text("Defects Presets", defects_presets)
    if recipes:
        lines = []
        for info in recipes:
            meta = info.get("metadata", {}) or {}
            desc = str(meta.get("description", "")).strip()
            suffix = f": {desc}" if desc else ""
            lines.append(f"{info.get('name')}{suffix}")
        _print_named_block("Recipes", lines)
    if datasets:
        lines = []
        for c in datasets:
            req = " (category required)" if bool(c.get("requires_category", False)) else ""
            lines.append(f"{c['name']}{req}: {c['description']}")
        _print_named_block("Datasets", lines)
    return 0


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
