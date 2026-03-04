from __future__ import annotations

import argparse
import inspect
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from pyimgano.models.registry import MODEL_REGISTRY, create_model, materialize_model_constructor
from pyimgano.utils.jsonable import to_jsonable
from pyimgano.utils.optional_deps import optional_import


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-benchmark")
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
    if text is None:
        return {}

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--model-kwargs must be valid JSON. Original error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--model-kwargs must be a JSON object (e.g. '{\"k\": 1}').")

    return dict(parsed)


def _faiss_available() -> bool:
    module, _error = optional_import("faiss")
    return module is not None


def _default_knn_backend() -> str:
    return "faiss" if _faiss_available() else "sklearn"


def _resolve_preset_kwargs(preset: str | None, model_name: str) -> dict[str, Any]:
    if preset is None:
        return {}

    if preset == "industrial-fast":
        if model_name == "vision_patchcore":
            return {
                "backbone": "resnet50",
                "coreset_sampling_ratio": 0.02,
                "feature_projection_dim": 256,
                "n_neighbors": 3,
                "knn_backend": _default_knn_backend(),
                "memory_bank_dtype": "float16",
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet18",
                "d_reduced": 32,
                "image_size": 192,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "resnet18",
                "image_size": 192,
                "k_neighbors": 20,
                "feature_levels": ["layer2", "layer3"],
                "gaussian_sigma": 2.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.1,
                "image_size": 336,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.1,
                "train_patch_outlier_quantile": 0.1,
                "image_size": 336,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "resnet50",
                "epochs": 5,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 5,
                "n_flow_steps": 4,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 10,
                "n_flows": 2,
                "batch_size": 32,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 10,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 5,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 20,
                "batch_size": 16,
            }
        return {}

    if preset == "industrial-balanced":
        if model_name == "vision_patchcore":
            return {
                "backbone": "resnet50",
                "coreset_sampling_ratio": 0.05,
                "feature_projection_dim": 512,
                "n_neighbors": 5,
                "knn_backend": _default_knn_backend(),
                "memory_bank_dtype": "float16",
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet18",
                "d_reduced": 64,
                "image_size": 224,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "resnet50",
                "image_size": 256,
                "k_neighbors": 50,
                "feature_levels": ["layer1", "layer2", "layer3"],
                "gaussian_sigma": 4.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.2,
                "image_size": 448,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.2,
                "train_patch_outlier_quantile": 0.1,
                "image_size": 448,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "resnet50",
                "epochs": 10,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 10,
                "n_flow_steps": 6,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 15,
                "n_flows": 4,
                "batch_size": 32,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 50,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 10,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 50,
                "batch_size": 16,
            }
        return {}

    if preset == "industrial-accurate":
        if model_name == "vision_patchcore":
            return {
                "backbone": "wide_resnet50",
                "coreset_sampling_ratio": 0.1,
                "n_neighbors": 9,
                "knn_backend": _default_knn_backend(),
            }
        if model_name == "vision_padim":
            return {
                "backbone": "resnet50",
                "d_reduced": 128,
                "image_size": 224,
            }
        if model_name == "vision_spade":
            return {
                "backbone": "wide_resnet50",
                "image_size": 256,
                "k_neighbors": 50,
                "feature_levels": ["layer1", "layer2", "layer3"],
                "gaussian_sigma": 4.0,
            }
        if model_name == "vision_anomalydino":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.5,
                "image_size": 518,
            }
        if model_name == "vision_softpatch":
            return {
                "knn_backend": _default_knn_backend(),
                "coreset_sampling_ratio": 0.5,
                "train_patch_outlier_quantile": 0.05,
                "image_size": 518,
            }
        if model_name == "vision_simplenet":
            return {
                "backbone": "wide_resnet50",
                "epochs": 20,
                "batch_size": 16,
            }
        if model_name == "vision_fastflow":
            return {
                "epoch_num": 20,
                "n_flow_steps": 8,
                "batch_size": 32,
            }
        if model_name == "vision_cflow":
            return {
                "epochs": 50,
                "n_flows": 8,
                "batch_size": 16,
            }
        if model_name == "vision_stfpm":
            return {
                "epochs": 100,
                "batch_size": 32,
            }
        if model_name in ("vision_reverse_distillation", "vision_reverse_dist"):
            return {
                "epoch_num": 20,
                "batch_size": 32,
            }
        if model_name == "vision_draem":
            return {
                "image_size": 256,
                "epochs": 100,
                "batch_size": 16,
            }
        return {}

    raise ValueError(
        f"Unknown preset: {preset!r}. Choose from: "
        "industrial-fast, industrial-balanced, industrial-accurate"
    )


def _merge_checkpoint_path(
    model_kwargs: dict[str, Any],
    *,
    checkpoint_path: str | None,
) -> dict[str, Any]:
    if checkpoint_path is None:
        return dict(model_kwargs)

    out = dict(model_kwargs)
    existing = out.get("checkpoint_path", None)
    if existing is not None and str(existing) != str(checkpoint_path):
        raise ValueError(
            "checkpoint_path conflict: "
            f"--checkpoint-path={checkpoint_path!r} but model-kwargs checkpoint_path={existing!r}"
        )

    out["checkpoint_path"] = str(checkpoint_path)
    return out


def _get_model_signature_info(model_name: str) -> tuple[set[str], bool]:
    """Return (accepted_kwarg_names, accepts_var_kwargs) for a registered model."""

    # Ensure the model registry is populated (lazy scan; should stay import-light).
    import pyimgano.models  # noqa: F401

    constructor = materialize_model_constructor(model_name)
    sig = inspect.signature(constructor)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return accepted, accepts_var_kwargs


def _validate_user_model_kwargs(model_name: str, user_kwargs: dict[str, Any]) -> None:
    accepted, accepts_var_kwargs = _get_model_signature_info(model_name)
    if accepts_var_kwargs:
        return

    unknown = sorted(set(user_kwargs) - accepted)
    if unknown:
        allowed = ", ".join(sorted(accepted)) or "<none>"
        raise TypeError(
            f"Model {model_name!r} does not accept model-kwargs: {unknown}. "
            f"Allowed keys: {allowed}"
        )


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

    _validate_user_model_kwargs(model_name, user_kwargs)
    accepted, accepts_var_kwargs = _get_model_signature_info(model_name)

    out: dict[str, Any] = {}
    if preset_kwargs:
        for key, value in preset_kwargs.items():
            if accepts_var_kwargs or key in accepted:
                out[key] = value
    out.update(user_kwargs)
    auto_passthrough = {"contamination", "random_state", "random_seed"}
    for key, value in auto_kwargs.items():
        if key in out:
            continue
        # Auto kwargs are CLI-provided defaults (device/contamination/pretrained/seed-ish).
        # Even if a constructor accepts **kwargs, passing unknown auto keys can be unsafe:
        # many classical wrappers forward **kwargs into sklearn backends that will error.
        if key in accepted or (accepts_var_kwargs and key in auto_passthrough):
            out[key] = value

    if "feature_extractor" in out:
        from pyimgano.features.registry import resolve_feature_extractor

        out["feature_extractor"] = resolve_feature_extractor(out["feature_extractor"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # Import model implementations for side effects (registry population).
        # Keep `pyimgano.cli` importable without heavy deps; discovery/benchmarking
        # happens inside `main()`.
        import pyimgano.models  # noqa: F401
        import pyimgano.features  # noqa: F401

        if bool(getattr(args, "plugins", False)):
            from pyimgano.plugins import load_plugins

            # Best-effort: keep going even if a plugin fails to load.
            load_plugins(groups=("pyimgano.plugins",), on_error="warn")

        tags_raw = getattr(args, "tags", None)
        feature_tags_raw = getattr(args, "feature_tags", None)

        discovery_flags = [
            bool(args.list_models),
            args.model_info is not None,
            bool(args.list_categories),
            bool(args.list_suites),
            args.suite_info is not None,
            bool(args.list_sweeps),
            args.sweep_info is not None,
            bool(args.list_feature_extractors),
            args.feature_info is not None,
        ]
        if sum(1 for f in discovery_flags if f) > 1:
            raise ValueError(
                "--list-models, --model-info, --list-categories, "
                "--list-suites, --suite-info, --list-sweeps, --sweep-info, "
                "--list-feature-extractors, and --feature-info are mutually exclusive."
            )

        if bool(args.list_models):
            tags: list[str] = []
            if tags_raw:
                for item in tags_raw:
                    for tag in str(item).split(","):
                        tag = tag.strip()
                        if tag:
                            tags.append(tag)

            names = MODEL_REGISTRY.available(tags=tags or None)
            if bool(args.json):
                print(json.dumps(names, indent=2))
            else:
                for name in names:
                    print(name)
            return 0

        if bool(args.list_feature_extractors):
            from pyimgano.features import list_feature_extractors

            tags: list[str] = []
            if feature_tags_raw:
                for item in feature_tags_raw:
                    for tag in str(item).split(","):
                        tag = tag.strip()
                        if tag:
                            tags.append(tag)

            names = list_feature_extractors(tags=tags or None)
            if bool(args.json):
                print(json.dumps(names, indent=2))
            else:
                for name in names:
                    print(name)
            return 0

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

            from pyimgano.datasets.catalog import list_dataset_categories

            categories = list_dataset_categories(
                dataset=str(args.dataset),
                root=str(args.root) if args.root is not None else "",
                manifest_path=(str(args.manifest_path) if args.manifest_path is not None else None),
            )
            if bool(args.json):
                print(json.dumps(categories, indent=2))
            else:
                for cat in categories:
                    print(cat)
            return 0

        if bool(args.list_suites):
            from pyimgano.baselines import list_baseline_suites

            suites = list_baseline_suites()
            if bool(args.json):
                print(json.dumps(suites, indent=2))
            else:
                for name in suites:
                    print(name)
            return 0

        if bool(args.list_sweeps):
            from pyimgano.baselines.sweeps import list_sweeps

            sweeps = list_sweeps()
            if bool(args.json):
                print(json.dumps(sweeps, indent=2))
            else:
                for name in sweeps:
                    print(name)
            return 0

        if args.suite_info is not None:
            from pyimgano.baselines import get_baseline_suite, resolve_suite_baselines

            suite = get_baseline_suite(str(args.suite_info))
            baselines = resolve_suite_baselines(str(args.suite_info))

            payload: dict[str, Any] = {
                "name": str(suite.name),
                "description": str(suite.description),
                "entries": list(suite.entries),
                "baselines": [
                    {
                        "name": str(b.name),
                        "model": str(b.model),
                        "optional": bool(b.optional),
                        "requires_extras": list(getattr(b, "requires_extras", ())),
                        "description": str(b.description),
                        "kwargs": dict(b.kwargs),
                    }
                    for b in baselines
                ],
            }

            if bool(args.json):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
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
            from pyimgano.baselines.sweeps import resolve_sweep

            plan = resolve_sweep(str(args.sweep_info))
            variants_by_entry: dict[str, list[dict[str, Any]]] = {}
            for entry_name in sorted(plan.variants_by_entry.keys()):
                variants_by_entry[str(entry_name)] = [
                    {
                        "name": str(v.name),
                        "description": str(v.description),
                        "override": dict(v.override),
                    }
                    for v in plan.variants_by_entry[entry_name]
                ]

            payload: dict[str, Any] = {
                "name": str(plan.name),
                "description": str(plan.description),
                "entries": sorted(variants_by_entry.keys()),
                "variants_by_entry": variants_by_entry,
            }

            if bool(args.json):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
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
            model_name = str(args.model_info)
            try:
                materialize_model_constructor(model_name)
                entry = MODEL_REGISTRY.info(model_name)
            except KeyError as exc:
                raise ValueError(f"Unknown model: {model_name!r}") from exc

            signature = inspect.signature(entry.constructor)
            accepted, accepts_var_kwargs = _get_model_signature_info(model_name)

            payload = {
                "name": entry.name,
                "tags": list(entry.tags),
                "metadata": dict(entry.metadata),
                "signature": str(signature),
                "accepted_kwargs": sorted(accepted),
                "accepts_var_kwargs": bool(accepts_var_kwargs),
                "constructor": {
                    "module": getattr(entry.constructor, "__module__", "<unknown>"),
                    "qualname": getattr(entry.constructor, "__qualname__", "<unknown>"),
                },
            }

            if bool(args.json):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
                print(f"Name: {payload['name']}")
                tags = payload["tags"]
                print(f"Tags: {', '.join(tags) if tags else '<none>'}")
                print("Metadata:")
                metadata = payload["metadata"]
                if metadata:
                    for key in sorted(metadata):
                        print(f"  {key}: {metadata[key]}")
                else:
                    print("  <none>")
                print("Signature:")
                print(f"  {payload['signature']}")
                print(f"Accepts **kwargs: {'yes' if payload['accepts_var_kwargs'] else 'no'}")
                print("Accepted kwargs:")
                for key in payload["accepted_kwargs"]:
                    print(f"  - {key}")
            return 0

        if args.feature_info is not None:
            from pyimgano.features import feature_info as _feature_info

            payload = _feature_info(str(args.feature_info))

            if bool(args.json):
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
            else:
                print(f"Name: {payload['name']}")
                tags = payload["tags"]
                print(f"Tags: {', '.join(tags) if tags else '<none>'}")
                print("Metadata:")
                metadata = payload["metadata"]
                if metadata:
                    for key in sorted(metadata):
                        print(f"  {key}: {metadata[key]}")
                else:
                    print("  <none>")
                print("Signature:")
                print(f"  {payload['signature']}")
                print(f"Accepts **kwargs: {'yes' if payload['accepts_var_kwargs'] else 'no'}")
                print("Accepted kwargs:")
                for key in payload["accepted_kwargs"]:
                    print(f"  - {key}")
            return 0

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
            from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

            postprocess = AnomalyMapPostprocess(
                normalize=True,
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
            if best_metric in {"pixel_auroc", "pixel_average_precision", "aupro", "pixel_segf1"} and not bool(
                args.pixel
            ):
                raise ValueError("--suite-export-best-metric pixel_* requires --pixel.")

            from pyimgano.pipelines.run_suite import run_baseline_suite

            suite_include: list[str] | None = None
            if args.suite_include:
                suite_include = []
                for item in args.suite_include:
                    for name in str(item).split(","):
                        name = name.strip()
                        if name:
                            suite_include.append(name)

            suite_exclude: list[str] | None = None
            if args.suite_exclude:
                suite_exclude = []
                for item in args.suite_exclude:
                    for name in str(item).split(","):
                        name = name.strip()
                        if name:
                            suite_exclude.append(name)

            payload = run_baseline_suite(
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
                    int(args.manifest_split_seed) if args.manifest_split_seed is not None else None
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
                include_baselines=suite_include,
                exclude_baselines=suite_exclude,
                continue_on_error=bool(args.suite_continue_on_error),
                sweep=(str(args.suite_sweep) if args.suite_sweep is not None else None),
                sweep_max_variants=(
                    int(args.suite_sweep_max_variants)
                    if args.suite_sweep_max_variants is not None
                    else None
                ),
            )

            if args.suite_export is not None:
                run_dir = payload.get("run_dir")
                if not isinstance(run_dir, str) or not run_dir:
                    raise RuntimeError("internal error: expected suite run_dir for --suite-export.")

                from pyimgano.reporting.suite_export import export_suite_tables

                fmt = str(args.suite_export)
                formats = ["csv", "md"] if fmt == "both" else [fmt]
                export_suite_tables(payload, Path(run_dir), formats=formats, best_metric=best_metric)

            if args.output:
                from pyimgano.reporting.report import save_run_report

                save_run_report(Path(args.output), payload)
            else:
                print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))

            return 0

        requested_model = str(args.model)
        model_name = requested_model
        preset_model_auto_kwargs: dict[str, Any] = {}

        try:
            entry = MODEL_REGISTRY.info(model_name)
        except KeyError:
            # Allow short preset names (JSON-ready configs) for industrial workflows.
            from pyimgano.cli_presets import resolve_model_preset

            preset = resolve_model_preset(model_name)
            if preset is None:
                raise

            model_name = str(preset.model)
            preset_model_auto_kwargs = dict(preset.kwargs)
            entry = MODEL_REGISTRY.info(model_name)

        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        merged_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

        if (
            bool(entry.metadata.get("requires_checkpoint", False))
            and "checkpoint_path" not in merged_kwargs
        ):
            raise ValueError(
                f"Model {requested_model!r} requires a checkpoint. "
                "Provide --checkpoint-path or set checkpoint_path in --model-kwargs."
            )

        preset_kwargs = _resolve_preset_kwargs(args.preset, model_name)
        auto_kwargs: dict[str, Any] = dict(preset_model_auto_kwargs)
        auto_kwargs.update(
            {
            "device": args.device,
            "contamination": args.contamination,
            "pretrained": args.pretrained,
            }
        )
        if args.seed is not None:
            auto_kwargs["random_seed"] = int(args.seed)
            auto_kwargs["random_state"] = int(args.seed)

        model_kwargs = _build_model_kwargs(
            model_name,
            user_kwargs=merged_kwargs,
            preset_kwargs=preset_kwargs,
            auto_kwargs=auto_kwargs,
        )

        if args.save_detector is not None and bool(args.pixel):
            raise ValueError("--save-detector is only supported without --pixel.")
        if args.load_detector is not None and bool(args.pixel):
            raise ValueError("--load-detector is only supported without --pixel.")
        if args.cache_dir is not None and bool(args.pixel):
            raise ValueError("--cache-dir is only supported without --pixel.")
        if args.cache_dir is not None and str(args.input_mode) != "paths":
            raise ValueError("--cache-dir requires --input-mode paths.")

        if bool(args.pixel):
            if str(args.input_mode) != "paths":
                raise ValueError("--input-mode currently supports only 'paths' when using --pixel.")
            if str(category).lower() == "all":
                raise ValueError("--category all is not yet supported with --pixel.")

            detector = create_model(model_name, **model_kwargs)
            pixel_skip_reason = None
            if dataset.lower() == "manifest":
                import numpy as np

                from pyimgano.datasets.manifest import (
                    ManifestSplitPolicy,
                    load_manifest_benchmark_split,
                )
                from pyimgano.pipelines.mvtec_visa import BenchmarkSplit

                mp = str(args.root) if args.manifest_path is None else str(args.manifest_path)
                root_fallback = None if args.manifest_path is None else str(args.root)
                seed = (
                    int(args.manifest_split_seed)
                    if args.manifest_split_seed is not None
                    else (int(args.seed) if args.seed is not None else 0)
                )
                policy = ManifestSplitPolicy(
                    seed=seed,
                    test_normal_fraction=float(args.manifest_test_normal_fraction),
                )
                ms = load_manifest_benchmark_split(
                    manifest_path=mp,
                    root_fallback=root_fallback,
                    category=str(category),
                    resize=resize,
                    load_masks=True,
                    split_policy=policy,
                )
                pixel_skip_reason = ms.pixel_skip_reason
                split = BenchmarkSplit(
                    train_paths=list(ms.train_paths),
                    test_paths=list(ms.test_paths),
                    test_labels=np.asarray(ms.test_labels),
                    test_masks=ms.test_masks,
                )
            else:
                split = load_benchmark_split(
                    dataset=dataset,  # type: ignore[arg-type]
                    root=str(args.root),
                    category=str(category),
                    resize=resize,
                    load_masks=True,
                )
            results = evaluate_split(
                detector,
                split,
                compute_pixel_scores=True,
                postprocess=postprocess,
                pro_integration_limit=float(args.pixel_aupro_limit),
                pro_num_thresholds=int(args.pixel_aupro_thresholds),
                pixel_segf1=bool(args.pixel_segf1),
                pixel_threshold_strategy=args.pixel_threshold_strategy,
                pixel_normal_quantile=float(args.pixel_normal_quantile),
                calibration_fraction=float(args.pixel_calibration_fraction),
                calibration_seed=int(args.pixel_calibration_seed),
            )
            payload: dict[str, Any] = {
                "dataset": dataset,
                "category": category,
                "model": str(args.model),
                "preset": (str(args.preset) if args.preset is not None else None),
                "input_mode": str(args.input_mode),
                "device": str(args.device),
                "resize": list(resize),
                "results": results,
            }
            if pixel_skip_reason is not None:
                payload["pixel_metrics_status"] = {
                    "enabled": False,
                    "reason": str(pixel_skip_reason),
                }
        else:
            from pyimgano.pipelines.run_benchmark import run_benchmark

            payload = run_benchmark(
                dataset=dataset,
                root=str(args.root),
                manifest_path=(str(args.manifest_path) if args.manifest_path is not None else None),
                category=str(category),
                model=str(model_name),
                input_mode=str(args.input_mode),
                seed=(int(args.seed) if args.seed is not None else None),
                device=str(args.device),
                preset=(str(args.preset) if args.preset is not None else None),
                pretrained=bool(args.pretrained),
                contamination=float(args.contamination),
                resize=resize,
                model_kwargs=model_kwargs,
                calibration_quantile=(
                    float(args.calibration_quantile)
                    if args.calibration_quantile is not None
                    else None
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
                load_detector_path=(
                    str(args.load_detector) if args.load_detector is not None else None
                ),
                save_detector_path=(
                    str(args.save_detector) if args.save_detector is not None else None
                ),
                output_dir=(str(args.output_dir) if args.output_dir is not None else None),
            )

        if args.output:
            from pyimgano.reporting.report import save_run_report

            save_run_report(Path(args.output), payload)
        else:
            print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI surface error
        print(f"error: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            model_name = getattr(args, "model", None)
            if model_name:
                print(f"context: model={model_name!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
