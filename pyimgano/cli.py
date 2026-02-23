from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from pyimgano.models.registry import MODEL_REGISTRY, create_model
from pyimgano.utils.optional_deps import optional_import


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-benchmark")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["mvtec", "mvtec_ad", "mvtec_loco", "mvtec_ad2", "visa", "btad", "custom"],
    )
    parser.add_argument("--root", default=None, help="Dataset root path")
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
        "--output-dir",
        default=None,
        help=(
            "Optional run directory for saved artifacts. "
            "When omitted and --save-run is enabled, writes to runs/<timestamp>_<dataset>_<model>/"
        ),
    )
    parser.add_argument(
        "--save-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write run artifacts (report.json, per_image.jsonl) to disk. Default: true",
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
        "--list-models",
        action="store_true",
        help="List available model names and exit (default output: text, one per line)",
    )
    parser.add_argument(
        "--model-info",
        default=None,
        help="Show tags/metadata/signature/accepted kwargs for a model name and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --list-models/--model-info, output JSON instead of text",
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
    parser.add_argument("--model", default="vision_patchcore", help="Registered model name")
    parser.add_argument(
        "--preset",
        default=None,
        choices=["industrial-fast", "industrial-balanced", "industrial-accurate"],
        help="Optional model preset (applied before --model-kwargs). Default: none",
    )
    parser.add_argument("--device", default="cpu", help="cpu|cuda (model dependent)")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
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
        choices=["normal_pixel_quantile"],
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


_NUMPY, _ = optional_import("numpy")


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


def _to_jsonable(value: Any) -> Any:
    if _NUMPY is not None:
        if isinstance(value, (_NUMPY.floating, _NUMPY.integer)):  # type: ignore[attr-defined]
            return value.item()
        if isinstance(value, _NUMPY.ndarray):  # type: ignore[attr-defined]
            return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


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

    entry = MODEL_REGISTRY.info(model_name)
    constructor = entry.constructor
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
    for key, value in auto_kwargs.items():
        if key in out:
            continue
        if accepts_var_kwargs or key in accepted:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        # Import model implementations for side effects (registry population).
        # Keep `pyimgano.cli` importable without heavy deps; discovery/benchmarking
        # happens inside `main()`.
        import pyimgano.models  # noqa: F401

        tags_raw = getattr(args, "tags", None)

        if bool(args.list_models) and args.model_info is not None:
            raise ValueError("--list-models and --model-info are mutually exclusive.")

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

        if args.model_info is not None:
            model_name = str(args.model_info)
            try:
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
                print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))
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

        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        merged_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

        entry = MODEL_REGISTRY.info(args.model)
        if (
            bool(entry.metadata.get("requires_checkpoint", False))
            and "checkpoint_path" not in merged_kwargs
        ):
            raise ValueError(
                f"Model {args.model!r} requires a checkpoint. "
                "Provide --checkpoint-path or set checkpoint_path in --model-kwargs."
            )

        preset_kwargs = _resolve_preset_kwargs(args.preset, args.model)
        model_kwargs = _build_model_kwargs(
            args.model,
            user_kwargs=merged_kwargs,
            preset_kwargs=preset_kwargs,
            auto_kwargs={
                "device": args.device,
                "contamination": args.contamination,
                "pretrained": args.pretrained,
            },
        )

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

        if bool(args.pixel):
            if str(category).lower() == "all":
                raise ValueError("--category all is not yet supported with --pixel.")

            detector = create_model(args.model, **model_kwargs)
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
                "device": str(args.device),
                "resize": list(resize),
                "results": results,
            }
        else:
            from pyimgano.pipelines.run_benchmark import run_benchmark

            payload = run_benchmark(
                dataset=dataset,
                root=str(args.root),
                category=str(category),
                model=str(args.model),
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
                save_run=bool(args.save_run),
                per_image_jsonl=bool(args.per_image_jsonl),
                output_dir=(str(args.output_dir) if args.output_dir is not None else None),
            )

        if args.output:
            from pyimgano.reporting.report import save_run_report

            save_run_report(Path(args.output), payload)
        else:
            print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI surface error
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
