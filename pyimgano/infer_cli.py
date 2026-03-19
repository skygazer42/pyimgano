from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import pyimgano.cli_discovery_options as cli_discovery_options
import pyimgano.cli_discovery_rendering as cli_discovery_rendering
import pyimgano.cli_listing as cli_listing
import pyimgano.cli_output as cli_output
import pyimgano.services.discovery_service as discovery_service
import pyimgano.services.infer_artifact_service as infer_artifact_service
import pyimgano.services.infer_context_service as infer_context_service
import pyimgano.services.infer_continue_service as infer_continue_service
import pyimgano.services.infer_load_service as infer_load_service
import pyimgano.services.infer_options_service as infer_options_service
import pyimgano.services.infer_output_service as infer_output_service
import pyimgano.services.infer_runtime_service as infer_runtime_service
import pyimgano.services.infer_wrapper_service as infer_wrapper_service
from pyimgano.inference.api import (
    InferenceTiming,
    calibrate_threshold,
)
from pyimgano.models.registry import create_model

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parse_json_mapping_arg(text: str, *, arg_name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON. Original error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{arg_name} must be a JSON object (e.g. '{{\"k\": 1}}').")

    return dict(parsed)


def _parse_csv_ints_arg(text: str, *, arg_name: str) -> list[int]:
    raw = [t.strip() for t in str(text).split(",")]
    out: list[int] = []
    for item in raw:
        if not item:
            continue
        try:
            out.append(int(item))
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(
                f"{arg_name} must be a comma-separated list of ints, got {text!r}"
            ) from exc
    return out


def _parse_csv_strs_arg(text: str, *, arg_name: str) -> list[str]:
    del arg_name
    raw = [t.strip() for t in str(text).split(",")]
    return [t for t in raw if t]

def _apply_onnx_session_options_shorthand(
    *,
    model_name: str,
    user_kwargs: dict[str, Any],
    session_options: dict[str, Any] | None,
) -> dict[str, Any]:
    import pyimgano.services.model_options as model_options

    return model_options.apply_onnx_session_options_shorthand(
        model_name=model_name,
        user_kwargs=user_kwargs,
        session_options=session_options,
    )


def _default_onnx_sweep_intra_values() -> list[int]:
    import os

    n = int(os.cpu_count() or 8)
    cap = max(1, min(n, 16))
    vals = [1, 2, 4, 8, 16]
    out = [v for v in vals if v <= cap]
    return out or [1]


def _run_onnx_session_options_sweep(
    *,
    checkpoint_path: str,
    device: str,
    image_size: int,
    batch_size: int,
    inputs: list[str],
    base_session_options: dict[str, Any],
    intra_values: list[int],
    opt_levels: list[str],
    repeats: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a small timing+stability sweep for onnx_embed SessionOptions."""

    import statistics
    import time

    if str(device).strip().lower() != "cpu":
        raise ValueError("--onnx-sweep currently supports device='cpu' only.")
    if repeats <= 0:
        raise ValueError("--onnx-sweep-repeats must be > 0")
    if not inputs:
        raise ValueError("--onnx-sweep requires at least one input image")

    from pyimgano.features.onnx_embed import ONNXEmbedExtractor

    sample_inputs = list(inputs)

    candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for intra in intra_values:
        for opt in opt_levels:
            so = dict(base_session_options)
            so["intra_op_num_threads"] = int(intra)
            so["graph_optimization_level"] = str(opt)

            row: dict[str, Any] = {"session_options": dict(so)}
            try:
                extractor = ONNXEmbedExtractor(
                    checkpoint_path=str(checkpoint_path),
                    device="cpu",
                    batch_size=int(batch_size),
                    image_size=int(image_size),
                    session_options=dict(so),
                )

                # Warm up to materialize the ORT session and stabilize timings.
                extractor.extract(sample_inputs[:1])

                timings: list[float] = []
                first_out = None
                second_out = None
                for i in range(int(repeats)):
                    t0 = time.perf_counter()
                    out = extractor.extract(sample_inputs)
                    t1 = time.perf_counter()
                    timings.append(float(t1 - t0))
                    if i == 0:
                        first_out = np.asarray(out)
                    elif i == 1:
                        second_out = np.asarray(out)

                stable = True
                if first_out is None:
                    stable = False
                else:
                    stable = bool(np.all(np.isfinite(first_out)))
                if stable and (first_out is not None) and (second_out is not None):
                    stable = bool(np.allclose(first_out, second_out, rtol=1e-5, atol=1e-6))

                row.update(
                    {
                        "ok": True,
                        "stable": bool(stable),
                        "timing_seconds": list(timings),
                        "median_seconds": float(statistics.median(timings)) if timings else None,
                        "stdev_seconds": (
                            float(statistics.pstdev(timings)) if len(timings) >= 2 else 0.0
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - CLI boundary
                row.update(
                    {
                        "ok": False,
                        "stable": False,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )

            candidates.append(row)

            if bool(row.get("ok")) and bool(row.get("stable")):
                if best is None:
                    best = dict(row)
                else:
                    # Rank by median latency; tie-break by lower stdev.
                    a = float(row.get("median_seconds") or 1e99)
                    b = float(best.get("median_seconds") or 1e99)
                    if a < b:
                        best = dict(row)
                    elif a == b:
                        sa = float(row.get("stdev_seconds") or 1e99)
                        sb = float(best.get("stdev_seconds") or 1e99)
                        if sa < sb:
                            best = dict(row)

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "image_size": int(image_size),
        "batch_size": int(batch_size),
        "samples": int(len(sample_inputs)),
        "repeats": int(repeats),
        "grid": {
            "intra_op_num_threads": [int(v) for v in intra_values],
            "graph_optimization_level": [str(v) for v in opt_levels],
        },
        "base_session_options": dict(base_session_options),
        "candidates": list(candidates),
        "best": None if best is None else dict(best),
    }

    if best is None:
        raise RuntimeError(
            "ONNX sweep failed: no stable candidates. "
            "Try reducing the grid (threads/opt-levels) or inspect --onnx-sweep-json."
        )

    best_opts = dict(best.get("session_options") or {})
    return best_opts, payload


def _extract_onnx_checkpoint_path_for_sweep(user_kwargs: dict[str, Any]) -> str | None:
    ckpt = user_kwargs.get("checkpoint_path", None)
    if ckpt is not None and str(ckpt).strip():
        return str(ckpt)

    ek = user_kwargs.get("embedding_kwargs", None)
    if isinstance(ek, dict):
        ckpt2 = ek.get("checkpoint_path", None)
        if ckpt2 is not None and str(ckpt2).strip():
            return str(ckpt2)

    fx = user_kwargs.get("feature_extractor", None)
    if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
        kw = fx.get("kwargs", None)
        if isinstance(kw, dict):
            ckpt3 = kw.get("checkpoint_path", None)
            if ckpt3 is not None and str(ckpt3).strip():
                return str(ckpt3)

    return None


def _extract_session_options_for_sweep(user_kwargs: dict[str, Any]) -> dict[str, Any]:
    so = user_kwargs.get("session_options", None)
    if isinstance(so, dict):
        return dict(so)
    ek = user_kwargs.get("embedding_kwargs", None)
    if isinstance(ek, dict):
        so2 = ek.get("session_options", None)
        if isinstance(so2, dict):
            return dict(so2)
    fx = user_kwargs.get("feature_extractor", None)
    if isinstance(fx, dict) and str(fx.get("name", "")).strip() == "onnx_embed":
        kw = fx.get("kwargs", None)
        if isinstance(kw, dict):
            so3 = kw.get("session_options", None)
            if isinstance(so3, dict):
                return dict(so3)
    return {}


def _maybe_apply_onnx_session_options_and_sweep(
    *,
    args: argparse.Namespace,
    model_name: str,
    device: str,
    user_kwargs: dict[str, Any],
    inputs: list[str],
    onnx_session_options_cli: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply --onnx-session-options and/or --onnx-sweep to model kwargs (best-effort)."""

    if not bool(getattr(args, "onnx_sweep", False)) and not onnx_session_options_cli:
        return dict(user_kwargs)

    if bool(getattr(args, "onnx_sweep", False)):
        sweep_inputs = inputs[: int(getattr(args, "onnx_sweep_samples", 32) or 32)]
        intra_values = (
            _parse_csv_ints_arg(str(args.onnx_sweep_intra), arg_name="--onnx-sweep-intra")
            if getattr(args, "onnx_sweep_intra", None)
            else _default_onnx_sweep_intra_values()
        )
        opt_levels = (
            _parse_csv_strs_arg(str(args.onnx_sweep_opt_levels), arg_name="--onnx-sweep-opt-levels")
            if getattr(args, "onnx_sweep_opt_levels", None)
            else ["all", "extended"]
        )

        base_so = _extract_session_options_for_sweep(user_kwargs)
        if onnx_session_options_cli:
            base_so.update(dict(onnx_session_options_cli))

        ckpt_for_sweep = _extract_onnx_checkpoint_path_for_sweep(user_kwargs)
        if ckpt_for_sweep is None:
            raise ValueError(
                "--onnx-sweep requires checkpoint_path for ONNX models. "
                "Provide --checkpoint-path (or set checkpoint_path in --model-kwargs)."
            )

        best_so, sweep_payload = _run_onnx_session_options_sweep(
            checkpoint_path=str(ckpt_for_sweep),
            device=str(device),
            image_size=int(user_kwargs.get("image_size", 224)),
            batch_size=int(user_kwargs.get("batch_size", 16)),
            inputs=list(sweep_inputs),
            base_session_options=dict(base_so),
            intra_values=[int(v) for v in intra_values],
            opt_levels=[str(v) for v in opt_levels],
            repeats=int(getattr(args, "onnx_sweep_repeats", 3)),
        )

        if getattr(args, "onnx_sweep_json", None) is not None:
            sweep_path = Path(str(args.onnx_sweep_json))
            sweep_path.parent.mkdir(parents=True, exist_ok=True)
            sweep_path.write_text(
                json.dumps({"tool": "pyimgano-infer", "onnx_sweep": sweep_payload}, indent=2),
                encoding="utf-8",
            )

        return _apply_onnx_session_options_shorthand(
            model_name=model_name,
            user_kwargs=dict(user_kwargs),
            session_options=dict(best_so),
        )

    # No sweep: just apply the shorthand.
    return _apply_onnx_session_options_shorthand(
        model_name=model_name,
        user_kwargs=dict(user_kwargs),
        session_options=dict(onnx_session_options_cli or {}),
    )


def _apply_defects_defaults_from_payload(
    args: argparse.Namespace,
    defects_payload: dict[str, Any],
) -> None:
    infer_options_service.apply_defects_defaults(args, defects_payload)


def _apply_defects_preset_if_requested(args: argparse.Namespace) -> None:
    defects_payload = infer_options_service.resolve_defects_preset_payload(
        getattr(args, "defects_preset", None)
    )
    if defects_payload is None:
        return

    # Presets are intended to be a one-flag industrial on-ramp.
    args.defects = True
    _apply_defects_defaults_from_payload(args, defects_payload)


def _resolve_preprocessing_preset_knobs(args: argparse.Namespace):
    return infer_options_service.resolve_preprocessing_preset_knobs(
        getattr(args, "preprocessing_preset", None)
    )


def _resolve_prediction_cli_options(
    args: argparse.Namespace,
    *,
    prediction_payload: dict[str, Any] | None,
) -> tuple[bool, float | None, int | None]:
    default_reject_confidence_below = None
    default_reject_label = None
    if isinstance(prediction_payload, dict):
        if prediction_payload.get("reject_confidence_below", None) is not None:
            default_reject_confidence_below = float(
                prediction_payload["reject_confidence_below"]
            )
        if prediction_payload.get("reject_label", None) is not None:
            default_reject_label = int(prediction_payload["reject_label"])

    reject_confidence_below = (
        float(args.reject_confidence_below)
        if args.reject_confidence_below is not None
        else default_reject_confidence_below
    )
    reject_label = (
        int(args.reject_label) if args.reject_label is not None else default_reject_label
    )
    include_confidence = bool(getattr(args, "include_confidence", False)) or (
        reject_confidence_below is not None
    )
    return bool(include_confidence), reject_confidence_below, reject_label


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-infer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", default=None, help="Registered model name")
    source.add_argument(
        "--model-preset",
        default=None,
        help=(
            "Model preset name (shortcut to a registered model + kwargs). "
            "Use --list-model-presets to see options."
        ),
    )
    source.add_argument(
        "--from-run",
        default=None,
        help="Load model + threshold + (optional) checkpoint from a workbench run directory",
    )
    source.add_argument(
        "--infer-config",
        default=None,
        help="Load model + threshold + (optional) checkpoint from an exported infer_config.json file",
    )
    source.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit (default output: text, one per line)",
    )
    source.add_argument(
        "--model-info",
        default=None,
        help="Show tags/metadata/signature/accepted kwargs for a model name and exit",
    )
    source.add_argument(
        "--list-model-presets",
        action="store_true",
        help="List available model preset names and exit (default output: text, one per line)",
    )
    source.add_argument(
        "--model-preset-info",
        default=None,
        help="Show model preset details (model/kwargs/description) and exit",
    )
    parser.add_argument(
        "--from-run-category",
        default=None,
        help="When --from-run has multiple categories, select one category name",
    )
    parser.add_argument(
        "--infer-category",
        default=None,
        help="When --infer-config has multiple categories, select one category name",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=["industrial-fast", "industrial-balanced", "industrial-accurate"],
        help="Optional model preset (applied before --model-kwargs). Default: none",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "When used with discovery flags (--list-models/--model-info/"
            "--list-model-presets/--model-preset-info), output JSON instead of text"
        ),
    )
    parser.add_argument(
        "--tags",
        action="append",
        default=None,
        help=(
            "Filter discovery results by required tags (comma-separated or repeatable). "
            "Works with --list-models and --list-model-presets. Example: --tags vision,classical"
        ),
    )
    parser.add_argument(
        "--family",
        default=None,
        help=(
            "Optional algorithm family/tag filter for discovery. "
            "Works with --list-models and --list-model-presets. Example: --family patchcore"
        ),
    )
    parser.add_argument(
        "--type",
        dest="algorithm_type",
        default=None,
        help=(
            "Optional high-level algorithm type/tag filter for discovery. "
            "Works with --list-models. Example: --type deep-vision"
        ),
    )
    parser.add_argument(
        "--year",
        default=None,
        help=(
            "Optional publication year filter for discovery. "
            "Works with --list-models. Example: --year 2021"
        ),
    )
    parser.add_argument(
        "--preprocessing-preset",
        default=None,
        help=(
            "Optional deployable preprocessing preset. "
            "Use `pyim --list preprocessing --deployable-only` to discover names."
        ),
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (model dependent)")
    parser.add_argument("--contamination", type=float, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed for reproducibility (best-effort; passed as "
            "random_seed/random_state when supported)"
        ),
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object of extra model constructor kwargs, e.g. '{\"k\": 1}' (advanced)",
    )
    parser.add_argument(
        "--onnx-session-options",
        default=None,
        help=(
            "Optional JSON object of onnxruntime SessionOptions to pass to ONNX-based routes "
            "(e.g. vision_onnx_* wrappers). This avoids nested --model-kwargs. "
            'Example: \'{"intra_op_num_threads":8,"graph_optimization_level":"all"}\''
        ),
    )
    parser.add_argument(
        "--onnx-sweep",
        action="store_true",
        help=(
            "Run a small onnxruntime CPU SessionOptions sweep (threads + graph optimization level) "
            "and apply the best session_options before inference. Requires an ONNX route and --device cpu."
        ),
    )
    parser.add_argument(
        "--onnx-sweep-intra",
        default=None,
        help="Comma-separated intra_op_num_threads values for --onnx-sweep (default: 1,2,4,8 capped at 16).",
    )
    parser.add_argument(
        "--onnx-sweep-opt-levels",
        default=None,
        help="Comma-separated graph_optimization_level values for --onnx-sweep (default: all,extended).",
    )
    parser.add_argument(
        "--onnx-sweep-repeats",
        type=int,
        default=3,
        help="Timed repeats per candidate for --onnx-sweep (default: 3).",
    )
    parser.add_argument(
        "--onnx-sweep-samples",
        type=int,
        default=32,
        help="Max number of input images used for --onnx-sweep timing (default: 32).",
    )
    parser.add_argument(
        "--onnx-sweep-json",
        default=None,
        help="Optional path to write ONNX sweep results JSON (grid + timings + selected best).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for checkpoint-backed models; sets model kwarg checkpoint_path",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Optional directory of normal images used to `fit()` and calibrate threshold",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help=(
            "Optional directory of golden reference images for reference-based detectors. "
            "Matched by basename (filename)."
        ),
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=None,
        help=(
            "Optional score quantile used to set threshold_ from train scores (e.g. 0.995). "
            "Requires --train-dir. When omitted and the detector does not set threshold_ during fit(), "
            "defaults to 1-contamination when available, else 0.995."
        ),
    )
    parser.add_argument(
        "--input",
        action="append",
        required=False,
        help="Input image path or directory (repeatable). Directories are scanned recursively.",
    )
    parser.add_argument(
        "--include-maps",
        action="store_true",
        help="Request anomaly maps if detector supports them",
    )
    parser.add_argument(
        "--include-confidence",
        action="store_true",
        help="Include label confidence when the detector exposes confidence helpers",
    )
    parser.add_argument(
        "--reject-confidence-below",
        type=float,
        default=None,
        help="Rewrite low-confidence predictions to a reject label",
    )
    parser.add_argument(
        "--reject-label",
        type=int,
        default=None,
        help="Label value used for rejected samples (default: -2)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional tile size for high-resolution inference (requires a numpy-capable detector)",
    )
    parser.add_argument(
        "--tile-stride",
        type=int,
        default=None,
        help="Optional tile stride (defaults to tile-size)",
    )
    parser.add_argument(
        "--tile-score-reduce",
        default="max",
        choices=["max", "mean", "topk_mean"],
        help="How to aggregate tile scores into an image score",
    )
    parser.add_argument(
        "--tile-map-reduce",
        default="max",
        choices=["max", "mean", "hann", "gaussian"],
        help="How to blend overlapping tile maps",
    )
    parser.add_argument(
        "--tile-score-topk",
        type=float,
        default=0.1,
        help="Top-k fraction used for tile-score reduce mode 'topk_mean'",
    )
    parser.add_argument(
        "--u16-max",
        type=int,
        default=None,
        help=(
            "Optional max value for uint16 inputs when the CLI decodes images via OpenCV "
            "IMREAD_UNCHANGED (e.g. 4095 for 12-bit sensors). "
            "Only used for tiling/preprocessing wrapper decode paths."
        ),
    )
    parser.add_argument("--save-jsonl", default=None, help="Optional JSONL output path")
    parser.add_argument(
        "--save-maps",
        default=None,
        help="Optional directory to save anomaly maps as .npy (requires --include-maps)",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Apply standard postprocess to anomaly maps (only if --include-maps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optional chunk size for inference (default: 0, disabled)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print stage timing summary to stderr",
    )
    parser.add_argument(
        "--profile-json",
        default=None,
        help="Optional path to write a JSON profile payload (stable, machine-friendly).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Best-effort AMP/autocast for torch-backed models (requires torch + CUDA)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Best-effort production mode: record per-input errors and keep going. "
            "Exits with code 1 if any errors occurred."
        ),
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help=(
            "Stop early after N errors when --continue-on-error is set. Default: 0 (no limit)."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=0,
        help="Flush JSONL output files every N records (stability vs performance). Default: 0 (no periodic flush).",
    )
    parser.add_argument(
        "--include-anomaly-map-values",
        action="store_true",
        help="Include raw anomaly map values in JSONL (debug only; very large output)",
    )

    # Industrial defects export (mask + regions). Opt-in.
    parser.add_argument(
        "--defects",
        action="store_true",
        help="Enable defects export: binary mask + connected-component regions (requires anomaly maps)",
    )
    from pyimgano.cli_presets import list_defects_presets

    parser.add_argument(
        "--defects-preset",
        default=None,
        choices=list_defects_presets(),
        help=(
            "Optional defects postprocess preset (implies --defects). "
            "Applies default ROI/border/smoothing/hysteresis/shape filters for industrial FP reduction."
        ),
    )
    parser.add_argument(
        "--defects-image-space",
        action="store_true",
        help="Add bbox_xyxy_image to defects regions (best-effort; requires image size)",
    )
    parser.add_argument(
        "--save-masks",
        default=None,
        help="Optional directory to save binary defect masks (requires --defects)",
    )
    parser.add_argument(
        "--defects-regions-jsonl",
        default=None,
        help="Optional JSONL path to write per-image defect regions payloads (requires --defects).",
    )
    parser.add_argument(
        "--save-overlays",
        default=None,
        help="Optional directory to save FP debugging overlays (original + heatmap + mask outline/fill)",
    )
    parser.add_argument(
        "--mask-format",
        default="png",
        choices=["png", "npy", "npz"],
        help="Mask artifact format when saving masks (default: png)",
    )
    parser.add_argument(
        "--defects-mask-space",
        default="roi",
        choices=["roi", "full"],
        help=(
            "Which defect mask to export when ROI is set. "
            "roi=mask is limited to ROI (default); full=mask includes full-map pixels. "
            "Regions are always extracted within ROI when ROI is set."
        ),
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=None,
        help="Optional fixed pixel threshold used to binarize anomaly maps into defect masks",
    )
    parser.add_argument(
        "--pixel-threshold-strategy",
        default="normal_pixel_quantile",
        choices=["normal_pixel_quantile", "fixed", "infer_config"],
        help="How to resolve pixel threshold for defects (default: normal_pixel_quantile)",
    )
    parser.add_argument(
        "--pixel-normal-quantile",
        type=float,
        default=None,
        help="Quantile used for pixel threshold calibration on normal pixels (default: 0.999)",
    )
    parser.add_argument(
        "--defect-min-area",
        type=int,
        default=0,
        help="Remove connected components smaller than this area (default: 0)",
    )
    parser.add_argument(
        "--defect-border-ignore-px",
        type=int,
        default=0,
        help="Ignore N pixels at the anomaly-map border for defects extraction (default: 0)",
    )
    parser.add_argument(
        "--defect-map-smoothing",
        default="none",
        choices=["none", "median", "gaussian", "box"],
        help="Optional anomaly-map smoothing method for defects extraction (default: none)",
    )
    parser.add_argument(
        "--defect-map-smoothing-ksize",
        type=int,
        default=0,
        help="Optional smoothing kernel size (method dependent; default: 0)",
    )
    parser.add_argument(
        "--defect-map-smoothing-sigma",
        type=float,
        default=None,
        help="Optional gaussian sigma for map smoothing (default: 0.0)",
    )
    parser.add_argument(
        "--defect-hysteresis",
        action="store_true",
        help="Enable hysteresis thresholding (keeps low regions connected to high seeds)",
    )
    parser.add_argument(
        "--defect-hysteresis-low",
        type=float,
        default=None,
        help="Low threshold for hysteresis (default: derived from high threshold)",
    )
    parser.add_argument(
        "--defect-hysteresis-high",
        type=float,
        default=None,
        help="High threshold for hysteresis (default: pixel threshold)",
    )
    parser.add_argument(
        "--defect-merge-nearby",
        action="store_true",
        help="Merge nearby defect regions in JSONL output (mask unchanged; default: off)",
    )
    parser.add_argument(
        "--defect-merge-nearby-max-gap-px",
        type=int,
        default=0,
        help="Max bbox gap (px) for merging nearby regions (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-min-fill-ratio",
        type=float,
        default=None,
        help="Optional minimum fill ratio (area / bbox_area) for components (default: none)",
    )
    parser.add_argument(
        "--defect-max-aspect-ratio",
        type=float,
        default=None,
        help="Optional maximum aspect ratio for components (default: none)",
    )
    parser.add_argument(
        "--defect-min-solidity",
        type=float,
        default=None,
        help="Optional minimum solidity for components (default: none)",
    )
    parser.add_argument(
        "--defect-min-score-max",
        type=float,
        default=None,
        help="Remove components whose max anomaly score is below this (default: none)",
    )
    parser.add_argument(
        "--defect-min-score-mean",
        type=float,
        default=None,
        help="Remove components whose mean anomaly score is below this (default: none)",
    )
    parser.add_argument(
        "--defect-open-ksize",
        type=int,
        default=0,
        help="Morphology open kernel size applied to defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-close-ksize",
        type=int,
        default=0,
        help="Morphology close kernel size applied to defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-fill-holes",
        action="store_true",
        help="Fill internal holes in defect masks",
    )
    parser.add_argument(
        "--defects-mask-dilate",
        type=int,
        default=0,
        help="Optional dilation kernel size applied to exported defect masks (default: 0, disabled)",
    )
    parser.add_argument(
        "--defect-max-regions",
        type=int,
        default=None,
        help="Optional maximum number of regions to emit per image (after sorting)",
    )
    parser.add_argument(
        "--defect-max-regions-sort-by",
        default="score_max",
        choices=["score_max", "score_mean", "area"],
        help="Sort key for max-regions selection (default: score_max)",
    )
    parser.add_argument(
        "--roi-xyxy-norm",
        type=float,
        nargs=4,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Optional normalized ROI rectangle (x1 y1 x2 y2 in [0,1]) applied to defects only. "
            "When using --infer-config/--from-run, this defaults from the exported defects.roi_xyxy_norm."
        ),
    )
    return parser


def _collect_image_paths(raw: str | Path) -> list[str]:
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.is_file():
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path}")
        return [str(path)]

    out: list[str] = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
            out.append(str(p))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _apply_defects_preset_if_requested(args)

    try:
        # Import implementations for side effects (registry population).
        import time

        import pyimgano.models  # noqa: F401
        from pyimgano.cli_common import (
            merge_checkpoint_path,
            parse_model_kwargs,
        )
        from pyimgano.services.inference_service import iter_inference_records, run_inference

        preprocessing_preset_knobs = _resolve_preprocessing_preset_knobs(args)

        list_models = bool(getattr(args, "list_models", False))
        list_model_presets = bool(getattr(args, "list_model_presets", False))
        cli_discovery_options.validate_mutually_exclusive_flags(
            [
                ("--list-models", list_models),
                ("--model-info", getattr(args, "model_info", None) is not None),
                ("--list-model-presets", list_model_presets),
                ("--model-preset-info", getattr(args, "model_preset_info", None) is not None),
            ]
        )
        model_list_options = cli_discovery_options.resolve_model_list_discovery_options(
            list_models=list_models,
            tags=getattr(args, "tags", None),
            family=getattr(args, "family", None),
            algorithm_type=getattr(args, "algorithm_type", None),
            year=getattr(args, "year", None),
            allow_family_without_list_models=list_model_presets,
        )

        if list_models:
            names = discovery_service.list_discovery_model_names(
                tags=model_list_options.tags,
                family=model_list_options.family,
                algorithm_type=model_list_options.algorithm_type,
                year=model_list_options.year,
            )
            return cli_listing.emit_listing(
                names,
                json_output=bool(getattr(args, "json", False)),
                sort_keys=False,
            )

        if getattr(args, "model_info", None) is not None:
            model_name = str(getattr(args, "model_info"))
            payload = discovery_service.build_model_info_payload(model_name)
            return cli_discovery_rendering.emit_signature_payload(
                payload,
                json_output=bool(getattr(args, "json", False)),
            )

        if list_model_presets:
            names = discovery_service.list_model_preset_names(
                tags=model_list_options.tags,
                family=model_list_options.family,
            )
            json_output = bool(getattr(args, "json", False))
            json_payload = None
            if json_output:
                json_payload = discovery_service.list_model_preset_infos_payload(
                    tags=model_list_options.tags,
                    family=model_list_options.family,
                )
            return cli_listing.emit_listing(
                names,
                json_output=json_output,
                json_payload=json_payload,
                sort_keys=False,
            )

        if getattr(args, "model_preset_info", None) is not None:
            preset_name = str(getattr(args, "model_preset_info"))
            payload = discovery_service.build_model_preset_info_payload(preset_name)
            return cli_discovery_rendering.emit_model_preset_payload(
                payload,
                json_output=bool(getattr(args, "json", False)),
            )

        t_total_start = time.perf_counter()
        t_load_model = 0.0
        t_fit_calibrate = 0.0
        t_infer = 0.0
        t_artifacts = 0.0

        if not getattr(args, "input", None):
            raise ValueError(
                "--input is required for inference modes. "
                "Use --list-models/--model-info/--list-model-presets for discovery."
            )

        onnx_session_options_cli: dict[str, Any] | None = None
        if getattr(args, "onnx_session_options", None) is not None:
            onnx_session_options_cli = _parse_json_mapping_arg(
                str(args.onnx_session_options), arg_name="--onnx-session-options"
            )

        seed = int(args.seed) if args.seed is not None else None
        if seed is not None:
            from pyimgano.utils.seeding import seed_everything

            seed_everything(int(seed))

        inputs: list[str] = []
        for raw in args.input:
            inputs.extend(_collect_image_paths(raw))
        if not inputs:
            raise ValueError("No input images found.")

        from_run = args.from_run is not None
        infer_config_mode = args.infer_config is not None
        threshold_from_run = None
        infer_config_postprocess = None
        prediction_payload: dict[str, Any] | None = None
        context_postprocess_summary: dict[str, Any] | None = None
        defects_payload: dict[str, Any] | None = None
        defects_payload_source: str | None = None
        include_maps_by_default = False
        illumination_contrast_knobs = None
        tiling_payload: dict[str, Any] | None = None

        t_load_start = time.perf_counter()
        if from_run:
            context = infer_context_service.prepare_from_run_context(
                infer_context_service.FromRunInferContextRequest(
                    run_dir=str(args.from_run),
                    from_run_category=(
                        str(args.from_run_category) if args.from_run_category is not None else None
                    ),
                    preset=(str(args.preset) if args.preset is not None else None),
                    device=(str(args.device) if args.device is not None else None),
                    contamination=(
                        float(args.contamination) if args.contamination is not None else None
                    ),
                    pretrained=(bool(args.pretrained) if args.pretrained is not None else None),
                    model_kwargs=(
                        parse_model_kwargs(args.model_kwargs)
                        if args.model_kwargs is not None
                        else None
                    ),
                    checkpoint_path=(
                        str(args.checkpoint_path) if args.checkpoint_path is not None else None
                    ),
                )
            )
            for warning in context.warnings:
                print(f"warning: {warning}", file=sys.stderr)

            threshold_from_run = context.threshold
            model_name = str(context.model_name)
            defects_payload = context.defects_payload
            prediction_payload = context.prediction_payload
            defects_payload_source = context.defects_payload_source
            illumination_contrast_knobs = context.illumination_contrast_knobs
            tiling_payload = context.tiling_payload
            context_postprocess_summary = (
                dict(context.postprocess_summary)
                if context.postprocess_summary is not None
                else None
            )

            user_kwargs = merge_checkpoint_path(
                dict(context.base_user_kwargs),
                checkpoint_path=context.checkpoint_path,
            )

            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(context.device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            loaded = infer_load_service.load_config_backed_infer_detector(
                infer_load_service.ConfigBackedInferLoadRequest(
                    context=context,
                    seed=seed,
                    user_kwargs=user_kwargs,
                ),
                create_detector=create_model,
            )
            model_name = str(loaded.model_name)
            detector = loaded.detector
        elif infer_config_mode:
            context = infer_context_service.prepare_infer_config_context(
                infer_context_service.InferConfigContextRequest(
                    config_path=str(args.infer_config),
                    infer_category=(
                        str(args.infer_category) if args.infer_category is not None else None
                    ),
                    preset=(str(args.preset) if args.preset is not None else None),
                    device=(str(args.device) if args.device is not None else None),
                    contamination=(
                        float(args.contamination) if args.contamination is not None else None
                    ),
                    pretrained=(bool(args.pretrained) if args.pretrained is not None else None),
                    model_kwargs=(
                        parse_model_kwargs(args.model_kwargs)
                        if args.model_kwargs is not None
                        else None
                    ),
                    checkpoint_path=(
                        str(args.checkpoint_path) if args.checkpoint_path is not None else None
                    ),
                )
            )
            for warning in context.warnings:
                print(f"warning: {warning}", file=sys.stderr)

            model_name = str(context.model_name)
            threshold_from_run = context.threshold
            defects_payload = context.defects_payload
            prediction_payload = context.prediction_payload
            defects_payload_source = context.defects_payload_source
            illumination_contrast_knobs = context.illumination_contrast_knobs
            tiling_payload = context.tiling_payload
            infer_config_postprocess = context.infer_config_postprocess
            include_maps_by_default = bool(context.enable_maps_by_default)
            context_postprocess_summary = (
                dict(context.postprocess_summary)
                if context.postprocess_summary is not None
                else None
            )

            user_kwargs = merge_checkpoint_path(
                dict(context.base_user_kwargs),
                checkpoint_path=context.checkpoint_path,
            )
            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(context.device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            loaded = infer_load_service.load_config_backed_infer_detector(
                infer_load_service.ConfigBackedInferLoadRequest(
                    context=context,
                    seed=seed,
                    user_kwargs=user_kwargs,
                ),
                create_detector=create_model,
            )
            model_name = str(loaded.model_name)
            detector = loaded.detector
        else:
            if getattr(args, "model_preset", None) is not None:
                requested_model = str(args.model_preset)
            else:
                if args.model is None:
                    raise ValueError(
                        "--model or --model-preset is required when --from-run/--infer-config are not provided"
                    )
                requested_model = str(args.model)

            device = str(args.device) if args.device is not None else "cpu"
            contamination = float(args.contamination) if args.contamination is not None else 0.1
            # Industrial default: keep direct CLI usage offline-safe. Users can opt in
            # via `--pretrained` (may download weights).
            pretrained = bool(args.pretrained) if args.pretrained is not None else False

            # ONNX session-options sweep still runs at the CLI layer before detector setup.
            model_name = str(requested_model)
            user_kwargs = parse_model_kwargs(args.model_kwargs)
            user_kwargs = merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)
            user_kwargs = _maybe_apply_onnx_session_options_and_sweep(
                args=args,
                model_name=model_name,
                device=str(device),
                user_kwargs=dict(user_kwargs),
                inputs=list(inputs),
                onnx_session_options_cli=onnx_session_options_cli,
            )

            loaded = infer_load_service.load_direct_infer_detector(
                infer_load_service.DirectInferLoadRequest(
                    requested_model=str(requested_model),
                    preset=(str(args.preset) if args.preset is not None else None),
                    device=device,
                    contamination=contamination,
                    pretrained=pretrained,
                    seed=seed,
                    user_kwargs=user_kwargs,
                ),
                create_detector=create_model,
            )
            model_name = str(loaded.model_name)
            detector = loaded.detector

        if defects_payload is not None:
            _apply_defects_defaults_from_payload(args, defects_payload)

        if preprocessing_preset_knobs is not None:
            illumination_contrast_knobs = preprocessing_preset_knobs

        wrapped = infer_wrapper_service.apply_infer_detector_wrappers(
            infer_wrapper_service.InferDetectorWrapperRequest(
                detector=detector,
                model_name=model_name,
                threshold=threshold_from_run,
                tiling_payload=tiling_payload,
                tile_size=(int(args.tile_size) if args.tile_size is not None else None),
                tile_stride=(int(args.tile_stride) if args.tile_stride is not None else None),
                tile_score_reduce=str(args.tile_score_reduce),
                tile_score_topk=float(args.tile_score_topk),
                tile_map_reduce=str(args.tile_map_reduce),
                illumination_contrast_knobs=illumination_contrast_knobs,
                u16_max=(int(args.u16_max) if args.u16_max is not None else None),
            )
        )
        detector = wrapped.detector

        if args.reference_dir is not None:
            setter = getattr(detector, "set_reference_dir", None)
            if callable(setter):
                setter(str(args.reference_dir))
            else:
                raise ValueError(
                    "--reference-dir is only supported for reference-based detectors "
                    "(detectors implementing set_reference_dir())."
                )

        t_load_model = time.perf_counter() - t_load_start

        t_fit_start = time.perf_counter()
        train_paths: list[str] = []
        if args.train_dir is not None:
            train_paths = _collect_image_paths(args.train_dir)
            if not train_paths:
                raise ValueError(f"No images found in --train-dir={args.train_dir!r}")
            detector.fit(train_paths)

        if args.calibration_quantile is not None and not train_paths:
            raise ValueError("--calibration-quantile requires --train-dir")

        # Industrial-friendly default: when a train set is provided but the detector does
        # not set a threshold during fit(), auto-calibrate by a quantile on train scores.
        # This aligns `pyimgano-infer` with workbench/benchmark behavior while preserving
        # detectors that already provide `threshold_`.
        if train_paths:
            threshold_before = getattr(detector, "threshold_", None)
            should_calibrate = args.calibration_quantile is not None or threshold_before is None
            if should_calibrate:
                from pyimgano.calibration.score_threshold import resolve_calibration_quantile

                q, _src = resolve_calibration_quantile(
                    detector,
                    calibration_quantile=(
                        float(args.calibration_quantile)
                        if args.calibration_quantile is not None
                        else None
                    ),
                )
                batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None
                calibrate_threshold(
                    detector,
                    train_paths,
                    quantile=float(q),
                    batch_size=batch_size,
                    amp=bool(args.amp),
                )

        batch_size = int(args.batch_size) if int(args.batch_size) > 0 else None
        include_confidence, reject_confidence_below, reject_label = _resolve_prediction_cli_options(
            args,
            prediction_payload=prediction_payload,
        )

        runtime_plan = infer_runtime_service.prepare_infer_runtime_plan(
            infer_runtime_service.InferRuntimePlanRequest(
                detector=detector,
                include_maps_requested=bool(args.include_maps),
                include_maps_by_default=bool(include_maps_by_default),
                postprocess_requested=bool(args.postprocess),
                infer_config_postprocess=infer_config_postprocess,
                postprocess_summary=context_postprocess_summary,
                defects_enabled=bool(args.defects),
                defects_payload=defects_payload,
                defects_payload_source=defects_payload_source,
                pixel_threshold=(
                    float(args.pixel_threshold) if args.pixel_threshold is not None else None
                ),
                pixel_threshold_strategy=str(args.pixel_threshold_strategy),
                pixel_normal_quantile=(
                    float(args.pixel_normal_quantile)
                    if args.pixel_normal_quantile is not None
                    else 0.999
                ),
                roi_xyxy_norm=(
                    list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None
                ),
                train_paths=list(train_paths),
                batch_size=batch_size,
                amp=bool(args.amp),
            ),
            run_inference_impl=run_inference,
        )
        include_maps = bool(runtime_plan.include_maps)
        postprocess = runtime_plan.postprocess
        postprocess_summary = runtime_plan.postprocess_summary
        pixel_threshold_value = runtime_plan.pixel_threshold_value
        pixel_threshold_provenance = runtime_plan.pixel_threshold_provenance

        t_fit_calibrate = time.perf_counter() - t_fit_start

        continue_on_error = bool(getattr(args, "continue_on_error", False))
        max_errors = int(getattr(args, "max_errors", 0))
        flush_every = int(getattr(args, "flush_every", 0))

        infer_timing = InferenceTiming()
        results_iter = None
        if not bool(continue_on_error):
            results_iter = iter_inference_records(
                detector=detector,
                inputs=inputs,
                include_maps=bool(include_maps),
                include_confidence=bool(include_confidence),
                reject_confidence_below=reject_confidence_below,
                reject_label=reject_label,
                postprocess=postprocess,
                postprocess_summary=postprocess_summary,
                batch_size=batch_size,
                amp=bool(args.amp),
                timing=infer_timing,
            )

        maps_dir: Path | None = None
        if args.save_maps is not None:
            if not bool(include_maps):
                raise ValueError("--save-maps requires --include-maps")
            maps_dir = Path(args.save_maps)
            maps_dir.mkdir(parents=True, exist_ok=True)

        masks_dir: Path | None = None
        if args.save_masks is not None:
            masks_dir = Path(args.save_masks)
            masks_dir.mkdir(parents=True, exist_ok=True)

        overlays_dir: Path | None = None
        if args.save_overlays is not None:
            overlays_dir = Path(args.save_overlays)
            overlays_dir.mkdir(parents=True, exist_ok=True)

        output_targets = infer_output_service.open_infer_output_targets(
            infer_output_service.InferOutputTargetsRequest(
                save_jsonl=(str(args.save_jsonl) if args.save_jsonl is not None else None),
                defects_enabled=bool(args.defects),
                defects_regions_jsonl=(
                    str(args.defects_regions_jsonl)
                    if args.defects_regions_jsonl is not None
                    else None
                ),
            )
        )

        try:
            t_loop_start = time.perf_counter()
            out_written = 0
            regions_written = 0
            errors = 0
            processed = 0

            def _process_ok_result(
                *, index: int, input_path: str, result: Any, include_status: bool
            ) -> None:
                nonlocal out_written, regions_written
                artifact_request = infer_artifact_service.build_infer_result_artifact_request_from_cli(
                    infer_artifact_service.InferResultArtifactCliRequest(
                        index=int(index),
                        input_path=str(input_path),
                        result=result,
                        cli_args=args,
                        include_status=bool(include_status),
                        maps_dir=(str(maps_dir) if maps_dir is not None else None),
                        overlays_dir=(str(overlays_dir) if overlays_dir is not None else None),
                        masks_dir=(str(masks_dir) if masks_dir is not None else None),
                        pixel_threshold_value=(
                            float(pixel_threshold_value)
                            if pixel_threshold_value is not None
                            else None
                        ),
                        pixel_threshold_provenance=(
                            dict(pixel_threshold_provenance)
                            if pixel_threshold_provenance is not None
                            else None
                        ),
                    )
                )

                artifact = infer_artifact_service.materialize_infer_result_artifacts(
                    artifact_request
                )
                write_result = infer_output_service.write_infer_output_payloads(
                    infer_output_service.InferOutputWriteRequest(
                        record=dict(artifact.record),
                        regions_payload=(
                            dict(artifact.regions_payload)
                            if artifact.regions_payload is not None
                            else None
                        ),
                        output_file=output_targets.output_file,
                        regions_file=output_targets.regions_file,
                        flush_every=int(flush_every),
                        output_written=int(out_written),
                        regions_written=int(regions_written),
                    )
                )
                out_written = int(write_result.output_written)
                regions_written = int(write_result.regions_written)

            def _handle_continue_error(
                *, index: int, input_path: str, exc: Exception, stage: str
            ) -> None:
                nonlocal out_written, regions_written
                error_record = infer_output_service.build_infer_error_record(
                    infer_output_service.InferErrorRecordRequest(
                        index=int(index),
                        input_path=str(input_path),
                        exc=exc,
                        stage=str(stage),
                    )
                )
                write_result = infer_output_service.write_infer_output_payloads(
                    infer_output_service.InferOutputWriteRequest(
                        record=error_record,
                        output_file=output_targets.output_file,
                        regions_file=output_targets.regions_file,
                        flush_every=int(flush_every),
                        output_written=int(out_written),
                        regions_written=int(regions_written),
                    )
                )
                out_written = int(write_result.output_written)
                regions_written = int(write_result.regions_written)

            if bool(continue_on_error):
                continue_result = infer_continue_service.run_continue_on_error_inference(
                    infer_continue_service.ContinueOnErrorInferRequest(
                        detector=detector,
                        inputs=list(inputs),
                        include_maps=bool(include_maps),
                        include_confidence=bool(include_confidence),
                        reject_confidence_below=reject_confidence_below,
                        reject_label=reject_label,
                        postprocess=postprocess,
                        postprocess_summary=postprocess_summary,
                        batch_size=batch_size,
                        amp=bool(args.amp),
                        max_errors=int(max_errors),
                    ),
                    process_ok_result=lambda *, index, input_path, result: _process_ok_result(
                        index=int(index),
                        input_path=str(input_path),
                        result=result,
                        include_status=True,
                    ),
                    handle_error=_handle_continue_error,
                    run_inference_impl=run_inference,
                )
                processed = int(continue_result.processed)
                errors = int(continue_result.errors)
                infer_timing.seconds += float(continue_result.timing_seconds)
            else:
                if results_iter is None:  # pragma: no cover - defensive
                    raise RuntimeError("Internal error: results_iter was not initialized")

                count = 0
                for i, (input_path, result) in enumerate(zip(inputs, results_iter)):
                    _process_ok_result(
                        index=int(i),
                        input_path=str(input_path),
                        result=result,
                        include_status=False,
                    )
                    count += 1

                if count != len(inputs):
                    raise RuntimeError(
                        "Internal error: inference iterator produced fewer results than inputs "
                        f"({count} vs {len(inputs)})."
                    )

            t_loop = time.perf_counter() - t_loop_start
            t_infer = float(infer_timing.seconds)
            t_artifacts = max(0.0, float(t_loop) - float(t_infer))
        finally:
            if output_targets.output_file is not None:
                output_targets.output_file.close()
            if output_targets.regions_file is not None:
                output_targets.regions_file.close()

        if bool(args.profile):
            total = time.perf_counter() - t_total_start
            print(
                "profile: "
                + " ".join(
                    [
                        f"load_model={t_load_model:.3f}s",
                        f"fit_calibrate={t_fit_calibrate:.3f}s",
                        f"infer={t_infer:.3f}s",
                        f"artifacts={t_artifacts:.3f}s",
                        f"total={total:.3f}s",
                    ]
                ),
                file=sys.stderr,
            )

        if args.profile_json is not None:
            profile_path = Path(str(args.profile_json))
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            total = time.perf_counter() - t_total_start
            profile_payload = {
                "tool": "pyimgano-infer",
                "counts": {
                    "inputs": int(len(inputs)),
                    "processed": int(len(inputs) if not bool(continue_on_error) else processed),
                    "errors": int(errors),
                },
                "timing_seconds": {
                    "load_model": float(t_load_model),
                    "fit_calibrate": float(t_fit_calibrate),
                    "infer": float(t_infer),
                    "artifacts": float(t_artifacts),
                    "total": float(total),
                },
            }
            profile_path.write_text(
                json.dumps(profile_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        if bool(continue_on_error) and int(errors) > 0:
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        context_lines: list[str] = []
        from_run = getattr(args, "from_run", None)
        if from_run:
            context_lines.append(f"context: from_run={from_run!r}")
            cat = getattr(args, "from_run_category", None)
            if cat:
                context_lines.append(f"context: from_run_category={cat!r}")
        if isinstance(exc, ImportError):
            model_name = getattr(args, "model", None)
            if model_name:
                context_lines.append(f"context: model={model_name!r}")
            model_preset = getattr(args, "model_preset", None)
            if model_preset:
                context_lines.append(f"context: model_preset={model_preset!r}")
        cli_output.print_cli_error(exc, context_lines=context_lines or None)
        return 2

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
