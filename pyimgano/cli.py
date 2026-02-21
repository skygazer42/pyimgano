from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.models import MODEL_REGISTRY, create_model
from pyimgano.pipelines.mvtec_visa import evaluate_split, load_benchmark_split
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess
from pyimgano.reporting.report import save_run_report
from pyimgano.utils.optional_deps import optional_import


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-benchmark")
    parser.add_argument("--dataset", required=True, choices=["mvtec", "mvtec_ad", "visa"])
    parser.add_argument("--root", required=True, help="Dataset root path")
    parser.add_argument("--category", required=True, help="Dataset category name")
    parser.add_argument("--model", default="vision_patchcore", help="Registered model name")
    parser.add_argument(
        "--preset",
        default=None,
        choices=["industrial-balanced"],
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
    parser.add_argument("--pixel", action="store_true", help="Compute pixel-level metrics if possible")
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


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
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


def _resolve_preset_kwargs(preset: str | None, model_name: str) -> dict[str, Any]:
    if preset is None:
        return {}

    if preset == "industrial-balanced":
        if model_name == "vision_patchcore":
            return {
                "backbone": "resnet50",
                "coreset_sampling_ratio": 0.05,
                "n_neighbors": 5,
                "knn_backend": "faiss" if _faiss_available() else "sklearn",
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
        return {}

    raise ValueError(f"Unknown preset: {preset!r}. Choose from: industrial-balanced")


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
        split = load_benchmark_split(
            dataset=args.dataset,
            root=args.root,
            category=args.category,
            resize=(256, 256),
            load_masks=True,
        )

        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        merged_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

        entry = MODEL_REGISTRY.info(args.model)
        if bool(entry.metadata.get("requires_checkpoint", False)) and "checkpoint_path" not in merged_kwargs:
            raise ValueError(
                f"Model {args.model!r} requires a checkpoint. "
                "Provide --checkpoint-path or set checkpoint_path in --model-kwargs."
            )

        preset_kwargs = _resolve_preset_kwargs(args.preset, args.model)
        detector = create_model(
            args.model,
            **_build_model_kwargs(
                args.model,
                user_kwargs=merged_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": args.device,
                    "contamination": args.contamination,
                    "pretrained": args.pretrained,
                },
            ),
        )

        postprocess = None
        if args.pixel and args.pixel_postprocess:
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

        results = evaluate_split(
            detector,
            split,
            compute_pixel_scores=bool(args.pixel),
            postprocess=postprocess,
            pro_integration_limit=float(args.pixel_aupro_limit),
            pro_num_thresholds=int(args.pixel_aupro_thresholds),
        )

        payload = {
            "dataset": args.dataset,
            "category": args.category,
            "model": args.model,
            "results": results,
        }

        if args.output:
            save_run_report(Path(args.output), payload)
        else:
            print(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True))

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI surface error
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
