from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.inference.api import calibrate_threshold, infer, result_to_jsonable
from pyimgano.models.registry import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


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
        import pyimgano.models  # noqa: F401
        from pyimgano.cli import _resolve_preset_kwargs
        from pyimgano.cli_common import build_model_kwargs, merge_checkpoint_path, parse_model_kwargs

        from_run = args.from_run is not None
        infer_config_mode = args.infer_config is not None
        trained_checkpoint_path = None
        threshold_from_run = None
        infer_config_postprocess = None

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
                },
            )
            detector = create_model(model_name, **model_kwargs)

            if trained_checkpoint_path is not None:
                load_checkpoint_into_detector(detector, trained_checkpoint_path)
            if threshold_from_run is not None:
                setattr(detector, "threshold_", float(threshold_from_run))
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
                },
            )

            detector = create_model(model_name, **model_kwargs)

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
                calibrate_threshold(detector, train_paths, quantile=float(q))

        inputs: list[str] = []
        for raw in args.input:
            inputs.extend(_collect_image_paths(raw))
        if not inputs:
            raise ValueError("No input images found.")

        postprocess: AnomalyMapPostprocess | None = None
        if bool(args.include_maps):
            if bool(args.postprocess):
                postprocess = AnomalyMapPostprocess()
            elif infer_config_postprocess is not None:
                postprocess = _build_postprocess_from_payload(infer_config_postprocess)

        results = infer(
            detector,
            inputs,
            include_maps=bool(args.include_maps),
            postprocess=postprocess,
        )

        map_paths: list[str | None] | None = None
        if args.save_maps is not None:
            if not bool(args.include_maps):
                raise ValueError("--save-maps requires --include-maps")
            out_dir = Path(args.save_maps)
            out_dir.mkdir(parents=True, exist_ok=True)

            map_paths = []
            for i, (input_path, result) in enumerate(zip(inputs, results)):
                if result.anomaly_map is None:
                    map_paths.append(None)
                    continue
                stem = Path(input_path).stem
                out_path = out_dir / f"{i:06d}_{stem}.npy"
                np.save(out_path, np.asarray(result.anomaly_map, dtype=np.float32))
                map_paths.append(str(out_path))

        def emit_records() -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for i, (input_path, result) in enumerate(zip(inputs, results)):
                record = result_to_jsonable(
                    result,
                    anomaly_map_path=(map_paths[i] if map_paths is not None else None),
                )
                record["index"] = int(i)
                record["input"] = str(input_path)
                records.append(record)
            return records

        records = emit_records()

        if args.save_jsonl is not None:
            out_path = Path(args.save_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, sort_keys=True))
                    f.write("\n")
        else:
            for record in records:
                print(json.dumps(record, sort_keys=True))

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
