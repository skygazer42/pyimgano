from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyimgano.cli_discovery_options as cli_discovery_options
import pyimgano.cli_discovery_rendering as cli_discovery_rendering
import pyimgano.cli_listing as cli_listing
import pyimgano.cli_output as cli_output
from pyimgano.models.registry import create_model
from pyimgano.utils.jsonable import to_jsonable
from pyimgano.utils.optional_deps import optional_import


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-benchmark")
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Optional benchmark config (JSON). "
            "When provided, config values are applied first and explicit CLI flags override them. "
            "Accepts a file path, '@path.json', or inline JSON (object or list)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=[
            "mvtec",
            "mvtec_ad",
            "mvtec_loco",
            "mvtec_ad2",
            "visa",
            "btad",
            "custom",
            "manifest",
        ],
    )
    parser.add_argument("--root", default=None, help="Dataset root path")
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Path to a JSONL manifest file when --dataset manifest. "
            "If omitted, --root is treated as the manifest path for backwards compatibility."
        ),
    )
    parser.add_argument(
        "--manifest-test-normal-fraction",
        type=float,
        default=0.2,
        help=(
            "When --dataset manifest, fraction of normal samples assigned to test during auto-split. "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--manifest-split-seed",
        type=int,
        default=None,
        help="When --dataset manifest, seed for deterministic auto-split. Defaults to --seed or 0.",
    )
    parser.add_argument("--category", default=None, help="Dataset category name")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="Resize images/masks during dataset loading. Default: 256 256",
    )
    parser.add_argument(
        "--input-mode",
        default="paths",
        choices=["paths", "numpy"],
        help=(
            "How to provide inputs to detectors. "
            "paths=pass file paths to detectors; numpy=decode images into memory first. "
            "Default: paths"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible runs (best-effort).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional run directory for saved artifacts. "
            "When omitted and --save-run is enabled, writes to runs/<timestamp>_<dataset>_<model>/"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Optional directory for caching extracted classical features when input_mode=paths. "
            "This can speed up repeated decision_function() calls on the same images."
        ),
    )
    parser.add_argument(
        "--save-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write run artifacts (report.json, per_image.jsonl) to disk. Default: true",
    )
    parser.add_argument(
        "--save-detector",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Save the fitted detector to disk (pickle; classical detectors only). "
            "Optionally provide PATH. When omitted, writes <output-dir>/detector.pkl. "
            "Warning: never load pickle files from untrusted sources."
        ),
    )
    parser.add_argument(
        "--load-detector",
        default=None,
        help=(
            "Load a previously saved detector (pickle; classical detectors only) and skip fitting. "
            "Warning: never load pickle files from untrusted sources."
        ),
    )
    parser.add_argument(
        "--per-image-jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write categories/<cat>/per_image.jsonl when saving a run. Default: true",
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=None,
        help=(
            "Score threshold quantile calibrated from train scores. "
            "Default: 1-contamination when available, else 0.995."
        ),
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional limit for number of train images (debug/smoke).",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional limit for number of test images (debug/smoke).",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List dataset categories for --dataset/--root and exit",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit (default output: text, one per line)",
    )
    parser.add_argument(
        "--list-feature-extractors",
        action="store_true",
        help="List available feature extractor names and exit (default output: text, one per line)",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List curated industrial baseline suites and exit",
    )
    parser.add_argument(
        "--suite-info",
        default=None,
        help="Show suite contents for a suite name and exit",
    )
    parser.add_argument(
        "--list-sweeps",
        action="store_true",
        help="List curated suite sweep profiles (small grid searches) and exit",
    )
    parser.add_argument(
        "--sweep-info",
        default=None,
        help="Show sweep profile contents for a sweep name and exit",
    )
    parser.add_argument(
        "--model-info",
        default=None,
        help="Show tags/metadata/signature/accepted kwargs for a model name and exit",
    )
    parser.add_argument(
        "--feature-info",
        default=None,
        help="Show tags/metadata/signature/accepted kwargs for a feature extractor name and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "When used with discovery flags (--list-models/--model-info/--list-suites/--suite-info/"
            "--list-sweeps/--sweep-info/--list-feature-extractors/--feature-info/--list-categories), "
            "output JSON instead of text"
        ),
    )
    parser.add_argument(
        "--plugins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load third-party pyimgano plugins via Python entry points (group 'pyimgano.plugins'). "
            "This is opt-in because plugins may import optional heavy dependencies. Default: false"
        ),
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Filter --list-models by required tags (comma-separated or repeatable). "
            "Example: --tags vision,deep"
        ),
    )
    parser.add_argument(
        "--family",
        default=None,
        help=(
            "Optional algorithm family/tag filter for --list-models. "
            "Example: --family patchcore"
        ),
    )
    parser.add_argument(
        "--type",
        dest="algorithm_type",
        default=None,
        help=(
            "Optional high-level algorithm type/tag filter for --list-models. "
            "Example: --type one-class-svm"
        ),
    )
    parser.add_argument(
        "--year",
        default=None,
        help=(
            "Optional publication year filter for --list-models. "
            "Example: --year 2021"
        ),
    )
    parser.add_argument(
        "--feature-tags",
        action="append",
        default=None,
        help=(
            "Filter --list-feature-extractors by required tags (comma-separated or repeatable). "
            "Example: --feature-tags embeddings"
        ),
    )
    parser.add_argument("--model", default="vision_patchcore", help="Registered model name")
    parser.add_argument(
        "--suite",
        default=None,
        help=(
            "Run a curated baseline suite (multiple presets) instead of a single model. "
            "Example: --suite industrial-v1"
        ),
    )
    parser.add_argument(
        "--suite-max-models",
        type=int,
        default=None,
        help="Optional limit for number of suite baselines (debug/smoke). Default: none",
    )
    parser.add_argument(
        "--suite-include",
        action="append",
        default=None,
        help=(
            "Optional allowlist of suite baseline names to run (comma-separated or repeatable). "
            "Example: --suite-include industrial-template-ncc-map,industrial-pixel-mad-map"
        ),
    )
    parser.add_argument(
        "--suite-exclude",
        action="append",
        default=None,
        help=(
            "Optional blocklist of suite baseline names to skip (comma-separated or repeatable). "
            "Example: --suite-exclude industrial-embed-knn-cosine"
        ),
    )
    parser.add_argument(
        "--suite-continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue the suite when a baseline fails or is missing optional deps. Default: true",
    )
    parser.add_argument(
        "--suite-export",
        default=None,
        choices=["csv", "md", "both"],
        help=(
            "When running --suite, export leaderboard/skipped/best_by_baseline tables to files in the suite output dir "
            "(requires --save-run). Choices: csv, md, both."
        ),
    )
    parser.add_argument(
        "--suite-export-best-metric",
        default="auroc",
        choices=[
            "auroc",
            "average_precision",
            "pixel_auroc",
            "pixel_average_precision",
            "aupro",
            "pixel_segf1",
        ],
        help=(
            "Metric used to select best variant per baseline when exporting best_by_baseline.* tables. "
            "Default: auroc."
        ),
    )

    parser.add_argument(
        "--suite-sweep",
        default=None,
        help=(
            "Optional sweep spec to run per-suite baseline (grid search). "
            "Use --list-sweeps for built-in profiles. "
            "Also accepts a JSON file path (or '@path.json') or inline JSON starting with '{'."
        ),
    )
    parser.add_argument(
        "--suite-sweep-max-variants",
        type=int,
        default=None,
        help=(
            "Optional cap for number of sweep variants per baseline (excluding base). "
            "Example: --suite-sweep-max-variants 1"
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["industrial-fast", "industrial-balanced", "industrial-accurate"],
        help="Optional model preset (applied before --model-kwargs). Default: none",
    )
    parser.add_argument("--device", default="cpu", help="cpu|cuda (model dependent)")
    parser.add_argument("--contamination", type=float, default=0.1)
    # Industrial default: keep CLIs offline-safe and avoid implicit weight downloads.
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object of extra model constructor kwargs, e.g. '{\"k\": 1}' (advanced)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for checkpoint-backed models; sets model kwarg checkpoint_path",
    )
    parser.add_argument(
        "--pixel", action="store_true", help="Compute pixel-level metrics if possible"
    )
    parser.add_argument(
        "--pixel-segf1",
        action="store_true",
        help=(
            "Compute pixel SegF1/bg-FPR under a single calibrated pixel threshold "
            "(VAND-style). Requires --pixel."
        ),
    )
    parser.add_argument(
        "--pixel-threshold-strategy",
        default=None,
        choices=["normal_pixel_quantile", "supervised_segf1"],
        help=(
            "Pixel threshold calibration strategy used for --pixel-segf1. "
            "Default: normal_pixel_quantile"
        ),
    )
    parser.add_argument(
        "--pixel-normal-quantile",
        type=float,
        default=0.999,
        help=(
            "Quantile used for normal_pixel_quantile calibration (only if --pixel-segf1). "
            "Default: 0.999"
        ),
    )
    parser.add_argument(
        "--pixel-calibration-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction of train/good held out for pixel threshold calibration (only if --pixel-segf1). "
            "Default: 0.2"
        ),
    )
    parser.add_argument(
        "--pixel-calibration-seed",
        type=int,
        default=0,
        help="RNG seed for calibration hold-out split (only if --pixel-segf1). Default: 0",
    )
    parser.add_argument(
        "--pixel-aupro-limit",
        type=float,
        default=0.3,
        help="FPR integration limit for AUPRO (only if --pixel). Default: 0.3",
    )
    parser.add_argument(
        "--pixel-aupro-thresholds",
        type=int,
        default=200,
        help="Number of thresholds used to approximate AUPRO (only if --pixel). Default: 200",
    )
    parser.add_argument(
        "--pixel-postprocess",
        action="store_true",
        help="Apply post-processing to anomaly maps before computing pixel metrics",
    )
    parser.add_argument(
        "--pixel-post-norm",
        default="minmax",
        choices=["minmax", "percentile", "none"],
        help="Normalization method for pixel anomaly maps (only if --pixel-postprocess)",
    )
    parser.add_argument(
        "--pixel-post-percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentile range for normalization (only if --pixel-post-norm=percentile)",
    )
    parser.add_argument(
        "--pixel-post-gaussian-sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for anomaly maps (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-open-ksize",
        type=int,
        default=0,
        help="Morphological open kernel size (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-close-ksize",
        type=int,
        default=0,
        help="Morphological close kernel size (0 disables)",
    )
    parser.add_argument(
        "--pixel-post-component-threshold",
        type=float,
        default=None,
        help="Threshold for connected-components filtering (requires --pixel-post-min-component-area > 0)",
    )
    parser.add_argument(
        "--pixel-post-min-component-area",
        type=int,
        default=0,
        help="Minimum component area for filtering (0 disables)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def load_benchmark_split(*args, **kwargs):
    # Lazy wrapper to keep CLI import light; also makes it easy to monkeypatch
    # in unit tests without importing cv2-heavy pipeline modules.
    from pyimgano.pipelines.mvtec_visa import load_benchmark_split as _load_benchmark_split

    return _load_benchmark_split(*args, **kwargs)


def evaluate_split(*args, **kwargs):
    # Lazy wrapper to keep CLI import light; also makes it easy to monkeypatch
    # in unit tests without importing cv2-heavy pipeline modules.
    from pyimgano.pipelines.mvtec_visa import evaluate_split as _evaluate_split

    return _evaluate_split(*args, **kwargs)


def _parse_model_kwargs(text: str | None) -> dict[str, Any]:
    from pyimgano import cli_common

    return cli_common.parse_model_kwargs(text)


def _faiss_available() -> bool:
    module, _error = optional_import("faiss")
    return module is not None


def _default_knn_backend() -> str:
    return "faiss" if _faiss_available() else "sklearn"


def _resolve_preset_kwargs(preset: str | None, model_name: str) -> dict[str, Any]:
    from pyimgano.services.model_options import resolve_preset_kwargs

    return resolve_preset_kwargs(
        preset,
        model_name,
        default_knn_backend=_default_knn_backend,
    )


def _merge_checkpoint_path(
    model_kwargs: dict[str, Any],
    *,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    from pyimgano import cli_common

    return cli_common.merge_checkpoint_path(model_kwargs, checkpoint_path=checkpoint_path)


def _get_model_signature_info(model_name: str) -> tuple[set[str], bool]:
    from pyimgano.cli_common import _get_model_signature_info as _shared_get_model_signature_info

    return _shared_get_model_signature_info(model_name)


def _validate_user_model_kwargs(model_name: str, user_kwargs: dict[str, Any]) -> None:
    from pyimgano import cli_common

    cli_common.validate_user_model_kwargs(model_name, user_kwargs)


def _build_model_kwargs(
    model_name: str,
    *,
    user_kwargs: dict[str, Any],
    preset_kwargs: dict[str, Any] | None = None,
    auto_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Combine preset, user, and auto kwargs with stable precedence.

    Precedence (highest wins):
    - user kwargs (explicit --model-kwargs)
    - preset kwargs (--preset)
    - auto kwargs (device/contamination/pretrained inferred from CLI flags)

    Preset/auto kwargs are filtered to only include keys accepted by the target
    model constructor (unless it accepts **kwargs).
    """

    from pyimgano import cli_common

    return cli_common.build_model_kwargs(
        model_name,
        user_kwargs=user_kwargs,
        preset_kwargs=preset_kwargs,
        auto_kwargs=auto_kwargs,
    )


def _extract_config_spec(argv: list[str]) -> tuple[str | None, list[str]]:
    """Extract --config from argv and return (spec, argv_without_config)."""

    spec = None
    cleaned: list[str] = []
    i = 0
    while i < len(argv):
        item = str(argv[i])
        if item == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config requires a value")
            spec = str(argv[i + 1])
            i += 2
            continue
        if item.startswith("--config="):
            spec = item.split("=", 1)[1]
            i += 1
            continue
        cleaned.append(item)
        i += 1
    return spec, cleaned


def _load_config_spec(spec: str) -> Any:
    text = str(spec).strip()
    if not text:
        raise ValueError("--config must not be empty")

    if text.startswith("@"):
        text = text[1:].strip()

    if text.startswith("{") or text.startswith("["):
        try:
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Failed to parse inline --config JSON") from exc

    path = Path(text)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to parse benchmark config JSON: {path}") from exc


def _primary_option_string(action: argparse.Action) -> str | None:
    for s in getattr(action, "option_strings", ()) or ():
        if str(s).startswith("--") and not str(s).startswith("--no-"):
            return str(s)
    opts = list(getattr(action, "option_strings", ()) or ())
    # Avoid index access: some static analyzers conservatively flag `opts[0]`
    # even when guarded by `if opts`.
    first = next(iter(opts), None)
    return str(first) if first is not None else None


def _argv_from_config_obj(parser: argparse.ArgumentParser, obj: Any) -> list[str]:
    """Convert a config payload to argv tokens.

    Supported config forms:
    - list[str]: treated as a raw argv token list (exact, explicit)
    - dict: keys map to argparse dest names; values map to option values
    """

    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            if not isinstance(item, str):
                raise ValueError("--config JSON list must contain only strings")
            if item.strip():
                out.append(item)
        return out

    if not isinstance(obj, dict):
        raise ValueError("--config JSON must be an object (dict) or a list of argv strings")

    actions_by_dest: dict[str, argparse.Action] = {}
    for act in parser._actions:  # noqa: SLF001 - argparse introspection for config expansion
        dest = getattr(act, "dest", None)
        if isinstance(dest, str) and dest:
            actions_by_dest[dest] = act

    out: list[str] = []
    for raw_key, value in obj.items():
        key = str(raw_key)
        if key == "config":
            continue

        action = actions_by_dest.get(key)
        if action is None:
            raise ValueError(f"Unknown --config key: {key!r}")

        if value is None:
            continue

        # BooleanOptionalAction supports --flag / --no-flag style toggles.
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise ValueError(f"--config key {key!r} must be boolean for this flag")
            opts = [str(s) for s in action.option_strings]
            pos = next((o for o in opts if o.startswith("--") and not o.startswith("--no-")), None)
            neg = next((o for o in opts if o.startswith("--no-")), None)
            if pos is None or neg is None:
                raise ValueError(
                    f"Internal error: expected --flag/--no-flag option strings for {key!r}"
                )
            out.append(pos if value else neg)
            continue

        # store_true flags
        if isinstance(action, argparse._StoreTrueAction):  # noqa: SLF001 - stable stdlib type
            if bool(value):
                opt = _primary_option_string(action)
                if opt is None:
                    raise ValueError(f"Internal error: missing option string for {key!r}")
                out.append(opt)
            continue

        # append flags (repeatable)
        if isinstance(action, argparse._AppendAction):  # noqa: SLF001 - stable stdlib type
            opt = _primary_option_string(action)
            if opt is None:
                raise ValueError(f"Internal error: missing option string for {key!r}")
            items = value if isinstance(value, list) else [value]
            for item in items:
                if item is None:
                    continue
                out.extend([opt, str(item)])
            continue

        opt = _primary_option_string(action)
        if opt is None:
            raise ValueError(f"Internal error: missing option string for {key!r}")

        nargs = getattr(action, "nargs", None)
        if nargs is None:
            out.extend([opt, str(value)])
            continue

        # nargs > 1 or fixed-size tuples/lists
        if isinstance(value, (list, tuple)):
            out.append(opt)
            out.extend([str(v) for v in value])
            continue

        raise ValueError(f"--config key {key!r} must be a list/tuple for nargs={nargs!r}")

    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_spec, cleaned_argv = _extract_config_spec(raw_argv)

    if config_spec is not None:
        cfg_obj = _load_config_spec(str(config_spec))
        cfg_argv = _argv_from_config_obj(parser, cfg_obj)
        args = parser.parse_args(cfg_argv + cleaned_argv)
    else:
        args = parser.parse_args(cleaned_argv)

    try:
        # Import model implementations for side effects (registry population).
        # Keep `pyimgano.cli` importable without heavy deps; discovery/benchmarking
        # happens inside `main()`.
        import pyimgano.features  # noqa: F401
        import pyimgano.models  # noqa: F401

        if bool(getattr(args, "plugins", False)):
            from pyimgano.plugins import load_plugins

            # Best-effort: keep going even if a plugin fails to load.
            load_plugins(groups=("pyimgano.plugins",), on_error="warn")

        import pyimgano.services.benchmark_service as benchmark_service
        import pyimgano.services.discovery_service as discovery_service

        tags_raw = getattr(args, "tags", None)
        feature_tags_raw = getattr(args, "feature_tags", None)

        cli_discovery_options.validate_mutually_exclusive_flags(
            [
                ("--list-models", bool(args.list_models)),
                ("--model-info", args.model_info is not None),
                ("--list-categories", bool(args.list_categories)),
                ("--list-suites", bool(args.list_suites)),
                ("--suite-info", args.suite_info is not None),
                ("--list-sweeps", bool(args.list_sweeps)),
                ("--sweep-info", args.sweep_info is not None),
                ("--list-feature-extractors", bool(args.list_feature_extractors)),
                ("--feature-info", args.feature_info is not None),
            ]
        )
        model_list_options = cli_discovery_options.resolve_model_list_discovery_options(
            list_models=bool(args.list_models),
            tags=tags_raw,
            family=getattr(args, "family", None),
            algorithm_type=getattr(args, "algorithm_type", None),
            year=getattr(args, "year", None),
            allow_family_without_list_models=False,
        )

        if bool(args.list_models):
            names = discovery_service.list_discovery_model_names(
                tags=model_list_options.tags,
                family=model_list_options.family,
                algorithm_type=model_list_options.algorithm_type,
                year=model_list_options.year,
            )
            return cli_listing.emit_listing(
                names,
                json_output=bool(args.json),
                sort_keys=False,
            )

        if bool(args.list_feature_extractors):
            names = discovery_service.list_discovery_feature_names(tags=feature_tags_raw)
            return cli_listing.emit_listing(
                names,
                json_output=bool(args.json),
                sort_keys=False,
            )

        if bool(args.list_categories):
            missing: list[str] = []
            if args.dataset is None:
                missing.append("--dataset")
            ds = "" if args.dataset is None else str(args.dataset)
            if ds.lower() == "manifest":
                if args.manifest_path is None and args.root is None:
                    missing.append("--manifest-path (or legacy --root=MANIFEST.jsonl)")
            else:
                if args.root is None:
                    missing.append("--root")
            if missing:
                raise ValueError(
                    "Missing required arguments for --list-categories: " f"{', '.join(missing)}."
                )

            categories = discovery_service.list_dataset_categories_payload(
                dataset=str(args.dataset),
                root=str(args.root) if args.root is not None else "",
                manifest_path=(str(args.manifest_path) if args.manifest_path is not None else None),
            )
            return cli_listing.emit_listing(
                categories,
                json_output=bool(args.json),
                sort_keys=False,
            )

        if bool(args.list_suites):
            suites = discovery_service.list_baseline_suites_payload()
            return cli_listing.emit_listing(
                suites,
                json_output=bool(args.json),
                sort_keys=False,
            )

        if bool(args.list_sweeps):
            sweeps = discovery_service.list_sweeps_payload()
            return cli_listing.emit_listing(
                sweeps,
                json_output=bool(args.json),
                sort_keys=False,
            )

        if args.suite_info is not None:
            payload = discovery_service.build_suite_info_payload(str(args.suite_info))

            if bool(args.json):
                return cli_output.emit_jsonable(payload)
            print(f"Name: {payload['name']}")
            print(f"Description: {payload['description']}")
            print("Entries:")
            for ref in payload["entries"]:
                print(f"  - {ref}")
            print("Baselines:")
            for b in payload["baselines"]:
                opt = "yes" if b.get("optional") else "no"
                print(f"  - {b['name']} -> {b['model']} (optional={opt})")
            return 0

        if args.sweep_info is not None:
            payload = discovery_service.build_sweep_info_payload(str(args.sweep_info))

            if bool(args.json):
                return cli_output.emit_jsonable(payload)
            print(f"Name: {payload['name']}")
            print(f"Description: {payload['description']}")
            print("Entries:")
            for name in payload["entries"]:
                variants = payload["variants_by_entry"].get(str(name), [])
                print(f"  - {name} (variants={len(variants)})")
            print("Variants:")
            for name in payload["entries"]:
                print(f"Entry: {name}")
                variants = payload["variants_by_entry"].get(str(name), [])
                for v in variants:
                    desc = str(v.get("description") or "").strip()
                    suffix = f" — {desc}" if desc else ""
                    print(f"  - {v.get('name')}{suffix}")
                    print(f"    override: {json.dumps(v.get('override', {}), sort_keys=True)}")
            return 0

        if args.model_info is not None:
            payload = discovery_service.build_model_info_payload(str(args.model_info))
            return cli_discovery_rendering.emit_signature_payload(
                payload,
                json_output=bool(args.json),
            )

        if args.feature_info is not None:
            payload = discovery_service.build_feature_info_payload(str(args.feature_info))
            return cli_discovery_rendering.emit_signature_payload(
                payload,
                json_output=bool(args.json),
            )

        missing: list[str] = []
        if args.dataset is None:
            missing.append("--dataset")
        if args.root is None:
            missing.append("--root")
        if args.category is None and str(args.dataset).lower() != "custom":
            missing.append("--category")
        if missing:
            raise ValueError(
                "Missing required arguments for benchmarking mode: "
                f"{', '.join(missing)}. "
                "Provide them or use --list-models/--model-info."
            )

        dataset = str(args.dataset)
        category = str(args.category) if args.category is not None else "custom"
        resize = (int(args.resize[0]), int(args.resize[1]))

        if dataset.lower() == "custom":
            from pyimgano.utils.datasets import CustomDataset

            CustomDataset(
                root=str(args.root),
                resize=resize,
                load_masks=bool(args.pixel),
            ).validate_structure()

        postprocess = None
        if args.pixel and args.pixel_postprocess:
            postprocess = benchmark_service.PixelPostprocessConfig(
                normalize_method=str(args.pixel_post_norm),
                percentile_range=(
                    float(args.pixel_post_percentiles[0]),
                    float(args.pixel_post_percentiles[1]),
                ),
                gaussian_sigma=float(args.pixel_post_gaussian_sigma),
                morph_open_ksize=int(args.pixel_post_open_ksize),
                morph_close_ksize=int(args.pixel_post_close_ksize),
                component_threshold=args.pixel_post_component_threshold,
                min_component_area=int(args.pixel_post_min_component_area),
            )

        if bool(args.pixel_segf1) and not bool(args.pixel):
            raise ValueError("--pixel-segf1 requires --pixel.")

        if args.suite is not None:
            if args.suite_export is not None and not bool(args.save_run):
                raise ValueError("--suite-export requires --save-run.")
            if args.model_kwargs is not None:
                raise ValueError("--model-kwargs is not supported with --suite.")
            if args.checkpoint_path is not None:
                raise ValueError("--checkpoint-path is not supported with --suite.")
            if args.preset is not None:
                raise ValueError("--preset is not supported with --suite.")
            if args.save_detector is not None:
                raise ValueError("--save-detector is not supported with --suite.")
            if args.load_detector is not None:
                raise ValueError("--load-detector is not supported with --suite.")

            best_metric = str(args.suite_export_best_metric)
            if best_metric in {
                "pixel_auroc",
                "pixel_average_precision",
                "aupro",
                "pixel_segf1",
            } and not bool(args.pixel):
                raise ValueError("--suite-export-best-metric pixel_* requires --pixel.")

            payload = benchmark_service.run_suite_request(
                benchmark_service.SuiteRunRequest(
                    suite=str(args.suite),
                    dataset=str(dataset),
                    root=str(args.root),
                    manifest_path=(
                        str(args.manifest_path) if args.manifest_path is not None else None
                    ),
                    category=str(category),
                    input_mode=str(args.input_mode),
                    seed=(int(args.seed) if args.seed is not None else None),
                    device=str(args.device),
                    pretrained=bool(args.pretrained),
                    contamination=float(args.contamination),
                    resize=resize,
                    calibration_quantile=(
                        float(args.calibration_quantile)
                        if args.calibration_quantile is not None
                        else None
                    ),
                    limit_train=(int(args.limit_train) if args.limit_train is not None else None),
                    limit_test=(int(args.limit_test) if args.limit_test is not None else None),
                    manifest_split_seed=(
                        int(args.manifest_split_seed)
                        if args.manifest_split_seed is not None
                        else None
                    ),
                    manifest_test_normal_fraction=float(args.manifest_test_normal_fraction),
                    pixel=bool(args.pixel),
                    pixel_segf1=bool(args.pixel_segf1),
                    pixel_threshold_strategy=(
                        str(args.pixel_threshold_strategy)
                        if args.pixel_threshold_strategy is not None
                        else None
                    ),
                    pixel_normal_quantile=float(args.pixel_normal_quantile),
                    pixel_calibration_fraction=float(args.pixel_calibration_fraction),
                    pixel_calibration_seed=int(args.pixel_calibration_seed),
                    pixel_postprocess=postprocess,
                    pixel_aupro_limit=float(args.pixel_aupro_limit),
                    pixel_aupro_thresholds=int(args.pixel_aupro_thresholds),
                    save_run=bool(args.save_run),
                    per_image_jsonl=bool(args.per_image_jsonl),
                    cache_dir=(str(args.cache_dir) if args.cache_dir is not None else None),
                    output_dir=(str(args.output_dir) if args.output_dir is not None else None),
                    max_models=(
                        int(args.suite_max_models) if args.suite_max_models is not None else None
                    ),
                    include_baselines=args.suite_include,
                    exclude_baselines=args.suite_exclude,
                    continue_on_error=bool(args.suite_continue_on_error),
                    sweep=(str(args.suite_sweep) if args.suite_sweep is not None else None),
                    sweep_max_variants=(
                        int(args.suite_sweep_max_variants)
                        if args.suite_sweep_max_variants is not None
                        else None
                    ),
                )
            )

            if args.suite_export is not None:
                run_dir = payload.get("run_dir")
                if not isinstance(run_dir, str) or not run_dir:
                    raise RuntimeError("internal error: expected suite run_dir for --suite-export.")

                from pyimgano.reporting.suite_export import export_suite_tables

                fmt = str(args.suite_export)
                formats = ["csv", "md"] if fmt == "both" else [fmt]
                export_suite_tables(
                    payload, Path(run_dir), formats=formats, best_metric=best_metric
                )

            if args.output:
                from pyimgano.reporting.report import save_run_report

                save_run_report(Path(args.output), payload)
            else:
                return cli_output.emit_jsonable(payload)

            return 0

        payload = benchmark_service.run_benchmark_request(
            benchmark_service.BenchmarkRunRequest(
                dataset=dataset,
                root=str(args.root),
                manifest_path=(str(args.manifest_path) if args.manifest_path is not None else None),
                category=str(category),
                model=str(args.model),
                input_mode=str(args.input_mode),
                seed=(int(args.seed) if args.seed is not None else None),
                device=str(args.device),
                preset=(str(args.preset) if args.preset is not None else None),
                pretrained=bool(args.pretrained),
                contamination=float(args.contamination),
                resize=resize,
                model_kwargs=_parse_model_kwargs(args.model_kwargs),
                checkpoint_path=(str(args.checkpoint_path) if args.checkpoint_path is not None else None),
                calibration_quantile=(
                    float(args.calibration_quantile) if args.calibration_quantile is not None else None
                ),
                limit_train=(int(args.limit_train) if args.limit_train is not None else None),
                limit_test=(int(args.limit_test) if args.limit_test is not None else None),
                manifest_split_seed=(
                    int(args.manifest_split_seed) if args.manifest_split_seed is not None else None
                ),
                manifest_test_normal_fraction=float(args.manifest_test_normal_fraction),
                save_run=bool(args.save_run),
                per_image_jsonl=bool(args.per_image_jsonl),
                cache_dir=(str(args.cache_dir) if args.cache_dir is not None else None),
                load_detector_path=(str(args.load_detector) if args.load_detector is not None else None),
                save_detector_path=(str(args.save_detector) if args.save_detector is not None else None),
                output_dir=(str(args.output_dir) if args.output_dir is not None else None),
                pixel=bool(args.pixel),
                pixel_segf1=bool(args.pixel_segf1),
                pixel_threshold_strategy=(
                    str(args.pixel_threshold_strategy)
                    if args.pixel_threshold_strategy is not None
                    else None
                ),
                pixel_normal_quantile=float(args.pixel_normal_quantile),
                pixel_calibration_fraction=float(args.pixel_calibration_fraction),
                pixel_calibration_seed=int(args.pixel_calibration_seed),
                pixel_postprocess=postprocess,
                pixel_aupro_limit=float(args.pixel_aupro_limit),
                pixel_aupro_thresholds=int(args.pixel_aupro_thresholds),
                default_knn_backend=_default_knn_backend,
            )
        )

        if args.output:
            from pyimgano.reporting.report import save_run_report

            save_run_report(Path(args.output), payload)
        else:
            return cli_output.emit_jsonable(payload)

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI surface error
        context_lines = None
        if isinstance(exc, ImportError):
            model_name = getattr(args, "model", None)
            if model_name:
                context_lines = [f"context: model={model_name!r}"]
        cli_output.print_cli_error(exc, context_lines=context_lines)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
