from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
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

    return parser


def _gather_checkpoint_paths_from_infer_config(payload: dict[str, Any]) -> list[str]:
    """Collect checkpoint paths referenced by an infer-config payload."""

    out: list[str] = []

    def _add(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        ckpt = obj.get("checkpoint", None)
        if not isinstance(ckpt, dict):
            return
        raw = ckpt.get("path", None)
        if raw is None:
            return
        text = str(raw).strip()
        if text:
            out.append(text)

    _add(payload)
    per_category = payload.get("per_category", None)
    if isinstance(per_category, dict):
        for _cat, cat_payload in per_category.items():
            _add(cat_payload)

    seen: set[str] = set()
    uniq: list[str] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _export_deploy_bundle(*, run_dir: Path, infer_config_payload: dict[str, Any]) -> Path:
    """Export a deploy-friendly bundle directory under `run_dir`."""

    bundle_dir = run_dir / "deploy_bundle"
    if bundle_dir.exists():
        raise FileExistsError(f"deploy bundle already exists: {bundle_dir}")
    bundle_dir.mkdir(parents=True, exist_ok=False)

    # Use the on-disk infer-config as the anchor for path resolution.
    infer_src = run_dir / "artifacts" / "infer_config.json"
    if not infer_src.exists():
        raise FileNotFoundError(f"infer_config.json not found: {infer_src}")

    # Best-effort: include run-level metadata when present.
    for name in ("report.json", "config.json", "environment.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, bundle_dir / name)

    def _resolve_path(raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            if not p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {p}")
            return p

        base = infer_src.parent
        candidates: list[Path] = [
            (base / p).resolve(),
            (base.parent / p).resolve(),
        ]

        from_run = infer_config_payload.get("from_run", None)
        if from_run is not None:
            try:
                candidates.append((Path(str(from_run)) / p).resolve())
            except Exception:
                pass

        for cand in candidates:
            if cand.exists():
                return cand

        tried = "\n".join(f"- {c}" for c in candidates)
        raise FileNotFoundError(
            "Artifact referenced by infer-config not found.\n"
            f"path={raw!r}\n"
            "Tried:\n"
            f"{tried}"
        )

    def _iter_path_slots(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
        out: list[tuple[str, dict[str, Any], str]] = []

        def _add_checkpoint(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            ckpt = obj.get("checkpoint", None)
            if isinstance(ckpt, dict):
                out.append(("trained_checkpoint", ckpt, "path"))

        def _add_model_checkpoint(obj: Any) -> None:
            if not isinstance(obj, dict):
                return
            model = obj.get("model", None)
            if isinstance(model, dict):
                out.append(("model_checkpoint", model, "checkpoint_path"))

        _add_checkpoint(payload)
        _add_model_checkpoint(payload)

        per_category = payload.get("per_category", None)
        if isinstance(per_category, dict):
            for _cat, cat_payload in per_category.items():
                _add_checkpoint(cat_payload)
                _add_model_checkpoint(cat_payload)
        return out

    # Copy referenced artifacts and rewrite absolute paths into bundle-relative ones.
    # This keeps deploy bundles self-contained and portable.
    bundle_payload = deepcopy(infer_config_payload)
    used_dst: set[Path] = set()
    bundle_root = bundle_dir.resolve()

    for kind, container, key in _iter_path_slots(bundle_payload):
        raw_any = container.get(key, None)
        if raw_any is None:
            continue
        raw = str(raw_any).strip()
        if not raw:
            continue

        p = Path(raw)
        src = _resolve_path(raw)

        if p.is_absolute():
            prefix = "checkpoints_abs" if kind == "trained_checkpoint" else "artifacts_abs"
            base = Path(prefix)
            dst_rel = base / p.name
            if dst_rel in used_dst:
                stem = p.stem
                suffix = p.suffix
                i = 2
                while True:
                    cand = base / f"{stem}_{i}{suffix}"
                    if cand not in used_dst:
                        dst_rel = cand
                        break
                    i += 1
            container[key] = str(dst_rel)
        else:
            dst_rel = Path(raw)

        used_dst.add(dst_rel)
        dst = (bundle_dir / dst_rel).resolve()
        if bundle_root not in dst.parents and dst != bundle_root:
            raise ValueError(f"infer-config path escapes deploy bundle: {raw!r} -> {dst_rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Write the (potentially rewritten) infer-config into the bundle.
    save_run_report(bundle_dir / "infer_config.json", bundle_payload)

    return bundle_dir


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
        if bool(args.preflight):
            if bool(args.dry_run):
                raise ValueError("--preflight and --dry-run are mutually exclusive.")

            from pyimgano.workbench.preflight import run_preflight

            report = run_preflight(config=cfg)
            payload = {"preflight": asdict(report)}
            print(json.dumps(payload, sort_keys=True))
            has_error = any(str(i.severity) == "error" for i in report.issues)
            return 2 if has_error else 0

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

        infer_config_payload: dict[str, Any] | None = None
        if bool(args.export_infer_config) or bool(args.export_deploy_bundle):
            if (
                bool(args.export_deploy_bundle)
                and bool(cfg.defects.enabled)
                and cfg.defects.pixel_threshold is None
            ):
                raise ValueError(
                    "--export-deploy-bundle with defects.enabled=true requires defects.pixel_threshold to be set.\n"
                    "Deploy bundles are intended to be self-contained for `pyimgano-infer --infer-config ... --defects`."
                )
            if not bool(cfg.output.save_run):
                raise ValueError(
                    "--export-infer-config/--export-deploy-bundle require output.save_run=true."
                )
            run_dir_raw = report.get("run_dir", None)
            if run_dir_raw is None:
                raise ValueError(
                    "--export-infer-config/--export-deploy-bundle require recipe output to include run_dir."
                )
            run_dir = Path(str(run_dir_raw))
            infer_config_path = run_dir / "artifacts" / "infer_config.json"

            from pyimgano.workbench.runner import build_infer_config_payload

            infer_config_payload = build_infer_config_payload(config=cfg, report=report)
            save_run_report(infer_config_path, infer_config_payload)

        if bool(args.export_deploy_bundle):
            if infer_config_payload is None:
                raise RuntimeError(
                    "Internal error: infer-config payload was not built for deploy bundle."
                )
            run_dir_raw = report.get("run_dir", None)
            if run_dir_raw is None:
                raise ValueError(
                    "--export-deploy-bundle requires recipe output to include run_dir."
                )
            run_dir = Path(str(run_dir_raw))
            bundle_dir = _export_deploy_bundle(
                run_dir=run_dir, infer_config_payload=infer_config_payload
            )
            report = dict(report)
            report["deploy_bundle_dir"] = str(bundle_dir)

        print(json.dumps(report, sort_keys=True))
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
