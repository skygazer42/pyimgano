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
    parser.add_argument("--model", required=True, help="Registered model name")
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
        "--train-dir",
        default=None,
        help="Optional directory of normal images used to `fit()` and calibrate threshold",
    )
    parser.add_argument(
        "--calibration-quantile",
        type=float,
        default=None,
        help="Optional score quantile used to set threshold_ (e.g. 0.995). If omitted, keep detector default.",
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
        from pyimgano.cli import (
            _build_model_kwargs,
            _merge_checkpoint_path,
            _parse_model_kwargs,
            _resolve_preset_kwargs,
        )

        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        user_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)
        preset_kwargs = _resolve_preset_kwargs(args.preset, args.model)

        model_kwargs = _build_model_kwargs(
            args.model,
            user_kwargs=user_kwargs,
            preset_kwargs=preset_kwargs,
            auto_kwargs={
                "device": args.device,
                "contamination": args.contamination,
                "pretrained": args.pretrained,
            },
        )

        detector = create_model(args.model, **model_kwargs)

        train_paths: list[str] = []
        if args.train_dir is not None:
            train_paths = _collect_image_paths(args.train_dir)
            if not train_paths:
                raise ValueError(f"No images found in --train-dir={args.train_dir!r}")
            detector.fit(train_paths)

        if args.calibration_quantile is not None:
            if not train_paths:
                raise ValueError("--calibration-quantile requires --train-dir")
            calibrate_threshold(detector, train_paths, quantile=float(args.calibration_quantile))

        inputs: list[str] = []
        for raw in args.input:
            inputs.extend(_collect_image_paths(raw))
        if not inputs:
            raise ValueError("No input images found.")

        postprocess: AnomalyMapPostprocess | None = None
        if bool(args.postprocess) and bool(args.include_maps):
            postprocess = AnomalyMapPostprocess()

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
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
