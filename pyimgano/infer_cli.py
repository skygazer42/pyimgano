from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.inference.api import InferenceTiming, calibrate_threshold, infer_iter, result_to_jsonable
from pyimgano.models.registry import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _apply_defects_defaults_from_payload(
    args: argparse.Namespace,
    defects_payload: dict[str, Any],
) -> None:
    """Apply defaults from an infer-config/run 'defects' payload.

    This keeps `infer_config.json` deploy-friendly: defects settings exported by
    workbench can travel with the model and be picked up by `pyimgano-infer`,
    while still allowing explicit CLI flags to override.
    """

    if not defects_payload:
        return

    def _coerce_roi(value: Any) -> list[float] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError("infer-config defects.roi_xyxy_norm must be a list of length 4 or null")
        try:
            return [float(v) for v in value]
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(
                f"infer-config defects.roi_xyxy_norm must contain floats, got {value!r}"
            ) from exc

    def _coerce_bool(value: Any, *, name: str) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        raise ValueError(f"infer-config defects.{name} must be a boolean, got {value!r}")

    def _coerce_int(value: Any, *, name: str) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(f"infer-config defects.{name} must be an int, got {value!r}") from exc

    def _coerce_float(value: Any, *, name: str) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            raise ValueError(f"infer-config defects.{name} must be a float, got {value!r}") from exc

    def _coerce_mask_format(value: Any) -> str | None:
        if value is None:
            return None
        fmt = str(value)
        if fmt not in ("png", "npy"):
            raise ValueError("infer-config defects.mask_format must be 'png' or 'npy'")
        return fmt

    def _coerce_smoothing_method(value: Any) -> str | None:
        if value is None:
            return None
        method = str(value).lower().strip()
        if method in ("none", "median", "gaussian", "box"):
            return method
        raise ValueError(
            "infer-config defects.map_smoothing.method must be one of: none|median|gaussian|box"
        )

    # Apply defaults only when the CLI did not explicitly set a value.
    if args.roi_xyxy_norm is None:
        roi = _coerce_roi(defects_payload.get("roi_xyxy_norm", None))
        if roi is not None:
            args.roi_xyxy_norm = roi

    # Numeric knobs (only apply if still default).
    if int(getattr(args, "defect_min_area", 0)) == 0:
        v = _coerce_int(defects_payload.get("min_area", None), name="min_area")
        if v is not None:
            args.defect_min_area = int(v)

    if int(getattr(args, "defect_border_ignore_px", 0)) == 0:
        v = _coerce_int(defects_payload.get("border_ignore_px", None), name="border_ignore_px")
        if v is not None:
            args.defect_border_ignore_px = int(v)

    # Optional map smoothing block.
    ms_raw = defects_payload.get("map_smoothing", None)
    if ms_raw is not None:
        if not isinstance(ms_raw, dict):
            raise ValueError("infer-config defects.map_smoothing must be a JSON object/dict.")
        if str(getattr(args, "defect_map_smoothing", "none")) == "none":
            v = _coerce_smoothing_method(ms_raw.get("method", None))
            if v is not None:
                args.defect_map_smoothing = str(v)
        if int(getattr(args, "defect_map_smoothing_ksize", 0)) == 0:
            v = _coerce_int(ms_raw.get("ksize", None), name="map_smoothing.ksize")
            if v is not None:
                args.defect_map_smoothing_ksize = int(v)
        if float(getattr(args, "defect_map_smoothing_sigma", 0.0)) == 0.0:
            v = _coerce_float(ms_raw.get("sigma", None), name="map_smoothing.sigma")
            if v is not None:
                args.defect_map_smoothing_sigma = float(v)

    # Optional hysteresis thresholding block.
    hyst_raw = defects_payload.get("hysteresis", None)
    if hyst_raw is not None:
        if not isinstance(hyst_raw, dict):
            raise ValueError("infer-config defects.hysteresis must be a JSON object/dict.")

        enabled = hyst_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_hysteresis", False)):
            args.defect_hysteresis = True

        if getattr(args, "defect_hysteresis_low", None) is None:
            v = _coerce_float(hyst_raw.get("low", None), name="hysteresis.low")
            if v is not None:
                args.defect_hysteresis_low = float(v)

        if getattr(args, "defect_hysteresis_high", None) is None:
            v = _coerce_float(hyst_raw.get("high", None), name="hysteresis.high")
            if v is not None:
                args.defect_hysteresis_high = float(v)

    # Optional shape filters block.
    shape_raw = defects_payload.get("shape_filters", None)
    if shape_raw is not None:
        if not isinstance(shape_raw, dict):
            raise ValueError("infer-config defects.shape_filters must be a JSON object/dict.")

        if getattr(args, "defect_min_fill_ratio", None) is None:
            v = _coerce_float(shape_raw.get("min_fill_ratio", None), name="shape_filters.min_fill_ratio")
            if v is not None:
                args.defect_min_fill_ratio = float(v)

        if getattr(args, "defect_max_aspect_ratio", None) is None:
            v = _coerce_float(shape_raw.get("max_aspect_ratio", None), name="shape_filters.max_aspect_ratio")
            if v is not None:
                args.defect_max_aspect_ratio = float(v)

        if getattr(args, "defect_min_solidity", None) is None:
            v = _coerce_float(shape_raw.get("min_solidity", None), name="shape_filters.min_solidity")
            if v is not None:
                args.defect_min_solidity = float(v)

    # Optional merge-nearby block (regions output only; mask unchanged).
    merge_raw = defects_payload.get("merge_nearby", None)
    if merge_raw is not None:
        if not isinstance(merge_raw, dict):
            raise ValueError("infer-config defects.merge_nearby must be a JSON object/dict.")

        enabled = merge_raw.get("enabled", None)
        if enabled is True and not bool(getattr(args, "defect_merge_nearby", False)):
            args.defect_merge_nearby = True

        if int(getattr(args, "defect_merge_nearby_max_gap_px", 0)) == 0:
            v = _coerce_int(merge_raw.get("max_gap_px", None), name="merge_nearby.max_gap_px")
            if v is not None:
                args.defect_merge_nearby_max_gap_px = int(v)

    if getattr(args, "defect_min_score_max", None) is None:
        v = _coerce_float(defects_payload.get("min_score_max", None), name="min_score_max")
        if v is not None:
            args.defect_min_score_max = float(v)

    if getattr(args, "defect_min_score_mean", None) is None:
        v = _coerce_float(defects_payload.get("min_score_mean", None), name="min_score_mean")
        if v is not None:
            args.defect_min_score_mean = float(v)

    if int(getattr(args, "defect_open_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("open_ksize", None), name="open_ksize")
        if v is not None:
            args.defect_open_ksize = int(v)

    if int(getattr(args, "defect_close_ksize", 0)) == 0:
        v = _coerce_int(defects_payload.get("close_ksize", None), name="close_ksize")
        if v is not None:
            args.defect_close_ksize = int(v)

    if getattr(args, "defect_max_regions", None) is None:
        v = defects_payload.get("max_regions", None)
        if v is not None:
            args.defect_max_regions = int(v)

    if str(getattr(args, "defect_max_regions_sort_by", "score_max")) == "score_max":
        v = defects_payload.get("max_regions_sort_by", None)
        if v is not None:
            vv = str(v).lower().strip()
            if vv not in ("score_max", "score_mean", "area"):
                raise ValueError(
                    "infer-config defects.max_regions_sort_by must be one of: score_max|score_mean|area"
                )
            args.defect_max_regions_sort_by = vv

    if float(getattr(args, "pixel_normal_quantile", 0.999)) == 0.999:
        v = _coerce_float(defects_payload.get("pixel_normal_quantile", None), name="pixel_normal_quantile")
        if v is not None:
            args.pixel_normal_quantile = float(v)

    if str(getattr(args, "pixel_threshold_strategy", "normal_pixel_quantile")) == "normal_pixel_quantile":
        v = defects_payload.get("pixel_threshold_strategy", None)
        if v is not None:
            args.pixel_threshold_strategy = str(v)

    if str(getattr(args, "mask_format", "png")) == "png":
        v = _coerce_mask_format(defects_payload.get("mask_format", None))
        if v is not None:
            args.mask_format = str(v)

    if not bool(getattr(args, "defect_fill_holes", False)):
        v = _coerce_bool(defects_payload.get("fill_holes", None), name="fill_holes")
        if v is True:
            args.defect_fill_holes = True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-infer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", default=None, help="Registered model name")
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
        required=True,
        help="Input image path or directory (repeatable). Directories are scanned recursively.",
    )
    parser.add_argument(
        "--include-maps",
        action="store_true",
        help="Request anomaly maps if detector supports them",
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
        "--amp",
        action="store_true",
        help="Best-effort AMP/autocast for torch-backed models (requires torch + CUDA)",
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
        default=0.999,
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
        default=0.0,
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

    try:
        # Import implementations for side effects (registry population).
        import time

        import pyimgano.models  # noqa: F401
        from pyimgano.cli import _resolve_preset_kwargs
        from pyimgano.cli_common import build_model_kwargs, merge_checkpoint_path, parse_model_kwargs

        t_total_start = time.perf_counter()
        t_load_model = 0.0
        t_fit_calibrate = 0.0
        t_infer = 0.0
        t_artifacts = 0.0

        seed = (int(args.seed) if args.seed is not None else None)
        if seed is not None:
            from pyimgano.utils.seeding import seed_everything

            seed_everything(int(seed))

        from_run = args.from_run is not None
        infer_config_mode = args.infer_config is not None
        trained_checkpoint_path = None
        threshold_from_run = None
        infer_config_postprocess = None
        defects_payload: dict[str, Any] | None = None
        defects_payload_source: str | None = None
        illumination_contrast_knobs = None

        t_load_start = time.perf_counter()
        if from_run:
            from pyimgano.workbench.load_run import (
                extract_threshold,
                load_checkpoint_into_detector,
                load_report_from_run,
                load_workbench_config_from_run,
                resolve_checkpoint_path,
                select_category_report,
            )

            cfg = load_workbench_config_from_run(args.from_run)
            report = load_report_from_run(args.from_run)
            _cat_name, cat_report = select_category_report(
                report,
                category=(str(args.from_run_category) if args.from_run_category is not None else None),
            )

            threshold_from_run = extract_threshold(cat_report)
            trained_checkpoint_path = resolve_checkpoint_path(args.from_run, cat_report)

            model_name = str(cfg.model.name)
            preset = cfg.model.preset
            device = str(cfg.model.device)
            contamination = float(cfg.model.contamination)
            pretrained = bool(cfg.model.pretrained)

            if args.preset is not None:
                preset = str(args.preset)
            if args.device is not None:
                device = str(args.device)
            if args.contamination is not None:
                contamination = float(args.contamination)
            if args.pretrained is not None:
                pretrained = bool(args.pretrained)

            base_user_kwargs = dict(cfg.model.model_kwargs)
            if args.model_kwargs is not None:
                base_user_kwargs.update(parse_model_kwargs(args.model_kwargs))

            checkpoint_path = (
                str(args.checkpoint_path)
                if args.checkpoint_path is not None
                else (str(cfg.model.checkpoint_path) if cfg.model.checkpoint_path is not None else None)
            )
            user_kwargs = merge_checkpoint_path(base_user_kwargs, checkpoint_path=checkpoint_path)

            preset_kwargs = _resolve_preset_kwargs(preset, model_name)
            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": device,
                    "contamination": contamination,
                    "pretrained": pretrained,
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                },
            )
            detector = create_model(model_name, **model_kwargs)

            if trained_checkpoint_path is not None:
                load_checkpoint_into_detector(detector, trained_checkpoint_path)
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

            # Use workbench defects config as deploy defaults for `pyimgano-infer`.
            defects_payload = asdict(cfg.defects)
            defects_payload_source = "from_run"

            # Optional preprocessing defaults from workbench runs.
            try:
                ic = cfg.preprocessing.illumination_contrast
            except Exception:
                ic = None
            if ic is not None:
                illumination_contrast_knobs = ic
        elif infer_config_mode:
            from pyimgano.inference.config import (
                load_infer_config,
                resolve_infer_checkpoint_path,
                select_infer_category,
            )
            from pyimgano.workbench.load_run import extract_threshold, load_checkpoint_into_detector

            cfg_path = Path(str(args.infer_config))
            payload = load_infer_config(cfg_path)
            payload = select_infer_category(
                payload,
                category=(str(args.infer_category) if args.infer_category is not None else None),
            )
            defects_map = payload.get("defects", None)
            if defects_map is not None:
                if not isinstance(defects_map, dict):
                    raise ValueError("infer-config key 'defects' must be a JSON object/dict.")
                defects_payload = dict(defects_map)
                defects_payload_source = "infer_config"

            preprocessing_map = payload.get("preprocessing", None)
            if preprocessing_map is not None:
                if not isinstance(preprocessing_map, dict):
                    raise ValueError("infer-config key 'preprocessing' must be a JSON object/dict.")
                ic_map = preprocessing_map.get("illumination_contrast", None)
                if ic_map is not None:
                    if not isinstance(ic_map, dict):
                        raise ValueError(
                            "infer-config key 'preprocessing.illumination_contrast' must be a JSON object/dict."
                        )
                    from pyimgano.inference.preprocessing import parse_illumination_contrast_knobs

                    illumination_contrast_knobs = parse_illumination_contrast_knobs(ic_map)

            model_payload = payload.get("model", None)
            if not isinstance(model_payload, dict):
                raise ValueError("infer-config must contain a JSON object at key 'model'.")
            model_name = model_payload.get("name", None)
            if model_name is None:
                raise ValueError("infer-config model.name is required.")
            model_name = str(model_name)

            adaptation_payload = payload.get("adaptation", None)
            if adaptation_payload is None:
                adaptation_payload = {}
            if not isinstance(adaptation_payload, dict):
                raise ValueError("infer-config key 'adaptation' must be a JSON object/dict.")

            threshold_from_run = extract_threshold(payload)
            trained_checkpoint_path = resolve_infer_checkpoint_path(payload, config_path=cfg_path)

            preset = model_payload.get("preset", None)
            device = model_payload.get("device", "cpu")
            contamination = model_payload.get("contamination", 0.1)
            pretrained = model_payload.get("pretrained", True)

            if args.preset is not None:
                preset = str(args.preset)
            if args.device is not None:
                device = str(args.device)
            if args.contamination is not None:
                contamination = float(args.contamination)
            if args.pretrained is not None:
                pretrained = bool(args.pretrained)

            base_user_kwargs = dict(model_payload.get("model_kwargs", {}) or {})
            if args.model_kwargs is not None:
                base_user_kwargs.update(parse_model_kwargs(args.model_kwargs))

            checkpoint_path = (
                str(args.checkpoint_path)
                if args.checkpoint_path is not None
                else (
                    str(model_payload.get("checkpoint_path"))
                    if model_payload.get("checkpoint_path", None) is not None
                    else None
                )
            )
            user_kwargs = merge_checkpoint_path(base_user_kwargs, checkpoint_path=checkpoint_path)

            preset_kwargs = _resolve_preset_kwargs(preset, model_name)
            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": str(device),
                    "contamination": float(contamination),
                    "pretrained": bool(pretrained),
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                },
            )
            detector = create_model(model_name, **model_kwargs)

            if trained_checkpoint_path is not None:
                load_checkpoint_into_detector(detector, trained_checkpoint_path)
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

            # Apply default tiling settings from infer-config when the CLI did not
            # explicitly request tiling.
            tiling = adaptation_payload.get("tiling", None)
            if (
                args.tile_size is None
                and isinstance(tiling, dict)
                and tiling.get("tile_size", None) is not None
            ):
                args.tile_size = int(tiling.get("tile_size"))
                if tiling.get("stride", None) is not None:
                    args.tile_stride = int(tiling.get("stride"))
                if tiling.get("score_reduce", None) is not None:
                    args.tile_score_reduce = str(tiling.get("score_reduce"))
                if tiling.get("map_reduce", None) is not None:
                    args.tile_map_reduce = str(tiling.get("map_reduce"))
                if tiling.get("score_topk", None) is not None:
                    args.tile_score_topk = float(tiling.get("score_topk"))

            # Infer-config may request maps/postprocess by default.
            post_cfg = adaptation_payload.get("postprocess", None)
            if isinstance(post_cfg, dict):
                infer_config_postprocess = dict(post_cfg)

            if not bool(args.include_maps):
                if bool(adaptation_payload.get("save_maps", False)) or infer_config_postprocess is not None:
                    args.include_maps = True
        else:
            if args.model is None:
                raise ValueError("--model is required when --from-run is not provided")

            model_name = str(args.model)
            preset_kwargs = _resolve_preset_kwargs(args.preset, model_name)

            device = str(args.device) if args.device is not None else "cpu"
            contamination = float(args.contamination) if args.contamination is not None else 0.1
            pretrained = bool(args.pretrained) if args.pretrained is not None else True

            user_kwargs = parse_model_kwargs(args.model_kwargs)
            user_kwargs = merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

            model_kwargs = build_model_kwargs(
                model_name,
                user_kwargs=user_kwargs,
                preset_kwargs=preset_kwargs,
                auto_kwargs={
                    "device": device,
                    "contamination": contamination,
                    "pretrained": pretrained,
                    **(
                        {"random_seed": int(seed), "random_state": int(seed)}
                        if seed is not None
                        else {}
                    ),
                },
            )

            detector = create_model(model_name, **model_kwargs)

        if defects_payload is not None:
            _apply_defects_defaults_from_payload(args, defects_payload)

        if args.tile_size is not None:
            from pyimgano.inference.tiling import TiledDetector

            detector = TiledDetector(
                detector=detector,
                tile_size=int(args.tile_size),
                stride=(int(args.tile_stride) if args.tile_stride is not None else None),
                score_reduce=str(args.tile_score_reduce),
                score_topk=float(args.tile_score_topk),
                map_reduce=str(args.tile_map_reduce),
            )
            # Ensure a threshold loaded from --from-run/--infer-config remains visible after wrapping.
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

        # Apply preprocessing outside tiling so illumination/contrast normalization happens
        # on the full image before tiling.
        if illumination_contrast_knobs is not None:
            from pyimgano.inference.preprocessing import PreprocessingDetector

            detector = PreprocessingDetector(
                detector=detector,
                illumination_contrast=illumination_contrast_knobs,
            )
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))

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
                batch_size = (int(args.batch_size) if int(args.batch_size) > 0 else None)
                calibrate_threshold(
                    detector,
                    train_paths,
                    quantile=float(q),
                    batch_size=batch_size,
                    amp=bool(args.amp),
                )

        inputs: list[str] = []
        for raw in args.input:
            inputs.extend(_collect_image_paths(raw))
        if not inputs:
            raise ValueError("No input images found.")

        if bool(args.defects) and not bool(args.include_maps):
            args.include_maps = True

        postprocess: AnomalyMapPostprocess | None = None
        if bool(args.include_maps):
            if bool(args.postprocess):
                postprocess = AnomalyMapPostprocess()
            elif infer_config_postprocess is not None:
                postprocess = _build_postprocess_from_payload(infer_config_postprocess)

        pixel_threshold_value: float | None = None
        pixel_threshold_provenance: dict[str, Any] | None = None
        if bool(args.defects):
            from pyimgano.defects.pixel_threshold import resolve_pixel_threshold

            infer_cfg_source = defects_payload_source or "infer_config"
            strategy = str(args.pixel_threshold_strategy)

            infer_cfg_thr = None
            if defects_payload is not None:
                raw_thr = defects_payload.get("pixel_threshold", None)
                if raw_thr is not None:
                    infer_cfg_thr = float(raw_thr)

            calibration_maps = None
            infer_cfg_thr_for_resolve = infer_cfg_thr

            if args.pixel_threshold is None and strategy == "normal_pixel_quantile":
                if not train_paths:
                    if infer_cfg_thr is None:
                        raise ValueError(
                            "--defects requires a pixel threshold.\n"
                            "Provide --pixel-threshold, set defects.pixel_threshold in infer_config.json, "
                            "or provide --train-dir for normal-pixel quantile calibration."
                        )
                else:
                    # Prefer re-calibration from normal pixels when train data is available,
                    # even if the infer-config/run contains a pre-set pixel threshold.
                    infer_cfg_thr_for_resolve = None

                    batch_size = (int(args.batch_size) if int(args.batch_size) > 0 else None)
                    calibration_maps = []
                    for r in infer_iter(
                        detector,
                        train_paths,
                        include_maps=True,
                        postprocess=postprocess,
                        batch_size=batch_size,
                        amp=bool(args.amp),
                    ):
                        if r.anomaly_map is not None:
                            calibration_maps.append(r.anomaly_map)

            pixel_threshold_value, pixel_threshold_provenance = resolve_pixel_threshold(
                pixel_threshold=(float(args.pixel_threshold) if args.pixel_threshold is not None else None),
                pixel_threshold_strategy=str(args.pixel_threshold_strategy),
                infer_config_pixel_threshold=infer_cfg_thr_for_resolve,
                calibration_maps=calibration_maps,
                pixel_normal_quantile=float(args.pixel_normal_quantile),
                infer_config_source=str(infer_cfg_source),
                roi_xyxy_norm=(list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None),
            )

        t_fit_calibrate = time.perf_counter() - t_fit_start

        infer_timing = InferenceTiming()
        batch_size = (int(args.batch_size) if int(args.batch_size) > 0 else None)
        results_iter = infer_iter(
            detector,
            inputs,
            include_maps=bool(args.include_maps),
            postprocess=postprocess,
            batch_size=batch_size,
            amp=bool(args.amp),
            timing=infer_timing,
        )

        maps_dir: Path | None = None
        if args.save_maps is not None:
            if not bool(args.include_maps):
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

        out_f = None
        if args.save_jsonl is not None:
            out_path = Path(args.save_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_f = out_path.open("w", encoding="utf-8")

        try:
            t_loop_start = time.perf_counter()
            count = 0
            for i, (input_path, result) in enumerate(zip(inputs, results_iter)):
                count += 1
                anomaly_map_path: str | None = None
                if maps_dir is not None:
                    if result.anomaly_map is None:
                        anomaly_map_path = None
                    else:
                        stem = Path(input_path).stem
                        out_path = maps_dir / f"{i:06d}_{stem}.npy"
                        np.save(out_path, np.asarray(result.anomaly_map, dtype=np.float32))
                        anomaly_map_path = str(out_path)

                record = result_to_jsonable(
                    result,
                    anomaly_map_path=anomaly_map_path,
                    include_anomaly_map_values=bool(args.include_anomaly_map_values),
                )
                record["index"] = int(i)
                record["input"] = str(input_path)

                saved_defects_mask: np.ndarray | None = None
                if bool(args.defects):
                    if result.anomaly_map is None:
                        raise ValueError(
                            "Defects export requires anomaly maps, but no anomaly_map was returned.\n"
                            "Try a detector that supports get_anomaly_map/predict_anomaly_map, and "
                            "ensure --include-maps (or --defects) is enabled."
                        )
                    if pixel_threshold_value is None or pixel_threshold_provenance is None:
                        raise RuntimeError("Internal error: pixel threshold was not resolved for --defects.")

                    from pyimgano.defects.extract import extract_defects_from_anomaly_map
                    from pyimgano.defects.io import save_binary_mask

                    defects = extract_defects_from_anomaly_map(
                        np.asarray(result.anomaly_map, dtype=np.float32),
                        pixel_threshold=float(pixel_threshold_value),
                        roi_xyxy_norm=(list(args.roi_xyxy_norm) if args.roi_xyxy_norm is not None else None),
                        border_ignore_px=int(args.defect_border_ignore_px),
                        map_smoothing_method=str(args.defect_map_smoothing),
                        map_smoothing_ksize=int(args.defect_map_smoothing_ksize),
                        map_smoothing_sigma=float(args.defect_map_smoothing_sigma),
                        hysteresis_enabled=bool(args.defect_hysteresis),
                        hysteresis_low=(
                            float(args.defect_hysteresis_low)
                            if args.defect_hysteresis_low is not None
                            else None
                        ),
                        hysteresis_high=(
                            float(args.defect_hysteresis_high)
                            if args.defect_hysteresis_high is not None
                            else None
                        ),
                        open_ksize=int(args.defect_open_ksize),
                        close_ksize=int(args.defect_close_ksize),
                        fill_holes=bool(args.defect_fill_holes),
                        min_area=int(args.defect_min_area),
                        min_fill_ratio=(
                            float(args.defect_min_fill_ratio) if args.defect_min_fill_ratio is not None else None
                        ),
                        max_aspect_ratio=(
                            float(args.defect_max_aspect_ratio) if args.defect_max_aspect_ratio is not None else None
                        ),
                        min_solidity=(
                            float(args.defect_min_solidity) if args.defect_min_solidity is not None else None
                        ),
                        min_score_max=(
                            float(args.defect_min_score_max) if args.defect_min_score_max is not None else None
                        ),
                        min_score_mean=(
                            float(args.defect_min_score_mean) if args.defect_min_score_mean is not None else None
                        ),
                        merge_nearby_enabled=bool(args.defect_merge_nearby),
                        merge_nearby_max_gap_px=int(args.defect_merge_nearby_max_gap_px),
                        max_regions_sort_by=str(args.defect_max_regions_sort_by),
                        max_regions=(int(args.defect_max_regions) if args.defect_max_regions is not None else None),
                    )
                    saved_defects_mask = defects["mask"]

                    mask_meta: dict[str, Any] = {
                        "shape": [int(d) for d in defects["mask"].shape],
                        "dtype": str(defects["mask"].dtype),
                    }
                    if masks_dir is not None:
                        stem = Path(input_path).stem
                        fmt = str(args.mask_format)
                        if fmt == "png":
                            ext = ".png"
                        elif fmt == "npy":
                            ext = ".npy"
                        elif fmt == "npz":
                            ext = ".npz"
                        else:  # pragma: no cover - guarded by argparse choices
                            raise ValueError(f"Unknown mask_format: {fmt!r}")
                        out_path = masks_dir / f"{i:06d}_{stem}{ext}"
                        written = save_binary_mask(defects["mask"], out_path, format=str(args.mask_format))
                        mask_meta.update(
                            {
                                "path": str(written),
                                "encoding": str(args.mask_format),
                            }
                        )
                    else:
                        mask_meta["encoding"] = str(args.mask_format)

                    record["defects"] = {
                        "space": defects["space"],
                        "pixel_threshold": float(pixel_threshold_value),
                        "pixel_threshold_provenance": dict(pixel_threshold_provenance),
                        "mask": mask_meta,
                        "regions": defects["regions"],
                        "map_stats_roi": defects.get("map_stats_roi", None),
                    }

                    if bool(args.defects_image_space):
                        try:
                            from PIL import Image

                            with Image.open(input_path) as im:
                                w_img, h_img = im.size

                            from pyimgano.defects.space import scale_bbox_xyxy_inclusive

                            src_hw = (int(defects["mask"].shape[0]), int(defects["mask"].shape[1]))
                            dst_hw = (int(h_img), int(w_img))
                            for region in record["defects"]["regions"]:
                                bbox = region.get("bbox_xyxy", None)
                                if bbox is None:
                                    continue
                                region["bbox_xyxy_image"] = scale_bbox_xyxy_inclusive(
                                    bbox,
                                    src_hw=src_hw,
                                    dst_hw=dst_hw,
                                )
                        except Exception:
                            # Best-effort: avoid failing the whole run for a debugging-only knob.
                            pass

                if overlays_dir is not None:
                    from pyimgano.defects.overlays import save_overlay_image

                    stem = Path(input_path).stem
                    out_path = overlays_dir / f"{i:06d}_{stem}.png"
                    save_overlay_image(
                        input_path,
                        anomaly_map=(result.anomaly_map if result.anomaly_map is not None else None),
                        defect_mask=saved_defects_mask,
                        out_path=out_path,
                    )

                line = json.dumps(record, sort_keys=True)
                if out_f is not None:
                    out_f.write(line)
                    out_f.write("\n")
                else:
                    print(line)

            t_loop = time.perf_counter() - t_loop_start
            t_infer = float(infer_timing.seconds)
            t_artifacts = max(0.0, float(t_loop) - float(t_infer))

            if count != len(inputs):
                raise RuntimeError(
                    "Internal error: inference iterator produced fewer results than inputs "
                    f"({count} vs {len(inputs)})."
                )
        finally:
            if out_f is not None:
                out_f.close()

        if bool(args.profile):
            import sys

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

        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        import sys

        print(f"error: {exc}", file=sys.stderr)
        from_run = getattr(args, "from_run", None)
        if from_run:
            print(f"context: from_run={from_run!r}", file=sys.stderr)
            cat = getattr(args, "from_run_category", None)
            if cat:
                print(f"context: from_run_category={cat!r}", file=sys.stderr)
        if isinstance(exc, ImportError):
            model_name = getattr(args, "model", None)
            if model_name:
                print(f"context: model={model_name!r}", file=sys.stderr)
        return 2
def _build_postprocess_from_payload(payload: dict[str, Any]) -> AnomalyMapPostprocess:
    pr_raw = payload.get("percentile_range", (1.0, 99.0))
    if isinstance(pr_raw, (list, tuple)) and len(pr_raw) == 2:
        pr = (float(pr_raw[0]), float(pr_raw[1]))
    else:
        pr = (1.0, 99.0)

    ct = payload.get("component_threshold", None)
    component_threshold = float(ct) if ct is not None else None

    return AnomalyMapPostprocess(
        normalize=bool(payload.get("normalize", True)),
        normalize_method=str(payload.get("normalize_method", "minmax")),
        percentile_range=pr,
        gaussian_sigma=float(payload.get("gaussian_sigma", 0.0)),
        morph_open_ksize=int(payload.get("morph_open_ksize", 0)),
        morph_close_ksize=int(payload.get("morph_close_ksize", 0)),
        component_threshold=component_threshold,
        min_component_area=int(payload.get("min_component_area", 0)),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
