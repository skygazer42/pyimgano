from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pyimgano.config import load_config


def _validate_recipe_config_path(
    *,
    recipe_name: str,
    config_path: Path,
) -> list[str]:
    issues: list[str] = []
    if not config_path.exists():
        return [f"{recipe_name}: missing starter config {config_path}"]

    try:
        payload = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - CLI/tool boundary
        return [f"{recipe_name}: failed to load {config_path}: {exc}"]

    actual_recipe = str(payload.get("recipe", "")).strip()
    if actual_recipe != str(recipe_name):
        issues.append(
            f"{recipe_name}: config {config_path} declares recipe {actual_recipe!r}"
        )
    return issues


def _validate_recipe_metadata(
    *,
    recipe_name: str,
    metadata: dict[str, object],
    repo_root: Path,
) -> list[str]:
    issues: list[str] = []
    config_paths: list[str] = []

    default_config = str(metadata.get("default_config", "")).strip()
    if default_config:
        config_paths.append(default_config)
    config_paths.extend(
        str(item) for item in metadata.get("starter_configs", []) or [] if str(item).strip()
    )

    if config_paths:
        seen: set[str] = set()
        for rel_path in config_paths:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            issues.extend(
                _validate_recipe_config_path(
                    recipe_name=str(recipe_name),
                    config_path=repo_root / rel_path,
                )
            )
        return issues

    starter_status = str(metadata.get("starter_status", "")).strip()
    starter_reason = str(metadata.get("starter_reason", "")).strip()
    if not starter_status:
        issues.append(f"{recipe_name}: missing starter_status for recipe without checked-in starter configs")
    if not starter_reason:
        issues.append(f"{recipe_name}: missing starter_reason for recipe without checked-in starter configs")
    return issues


def _audit_recipe_metadata(*, repo_root: Path) -> list[str]:
    import pyimgano.recipes  # noqa: F401
    from pyimgano.recipes.registry import list_recipes, recipe_info

    issues: list[str] = []
    for recipe_name in list_recipes():
        info = recipe_info(recipe_name)
        metadata = dict(info.get("metadata", {}) or {})
        issues.extend(
            _validate_recipe_metadata(
                recipe_name=str(recipe_name),
                metadata=metadata,
                repo_root=repo_root,
            )
        )
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="audit_recipe_starters",
        description=(
            "Fail when builtin recipe starter metadata points at missing configs "
            "or configs whose top-level recipe does not match the owning recipe."
        ),
    )
    parser.add_argument("--recipe-name", default=None, help="Optional single recipe name to validate.")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional single config path to validate against --recipe-name.",
    )
    args = parser.parse_args(argv)

    repo_root = _REPO_ROOT
    issues: list[str] = []

    if args.recipe_name is not None or args.config_path is not None:
        if args.recipe_name is None or args.config_path is None:
            parser.error("--recipe-name and --config-path must be provided together.")
        issues.extend(
            _validate_recipe_config_path(
                recipe_name=str(args.recipe_name),
                config_path=Path(str(args.config_path)),
            )
        )
    else:
        issues.extend(_audit_recipe_metadata(repo_root=repo_root))

    if issues:
        for issue in issues:
            print(issue)
        return 1

    print("OK: recipe starter metadata points at existing configs with matching recipe names.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
