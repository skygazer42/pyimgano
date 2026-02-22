from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pyimgano.models.registry import MODEL_REGISTRY, create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess
from pyimgano.reporting.report import save_run_report
from pyimgano.robustness.benchmark import run_robustness_benchmark
from pyimgano.robustness.corruptions import (
    apply_blur,
    apply_geo_jitter,
    apply_glare,
    apply_jpeg,
    apply_lighting,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyimgano-robust-benchmark")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["mvtec", "mvtec_ad", "mvtec_loco", "mvtec_ad2", "visa", "btad"],
    )
    parser.add_argument("--root", default=None, help="Dataset root path")
    parser.add_argument("--category", default=None, help="Dataset category name")
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="Resize images/masks to (H, W) before benchmarking. Default: 256 256",
    )
    parser.add_argument(
        "--input-mode",
        default="numpy",
        choices=["numpy", "paths"],
        help=(
            "How to feed inputs into the detector. "
            "'numpy' loads images as RGB uint8 arrays (required for corruptions). "
            "'paths' feeds file paths to the detector (corruptions are skipped). "
            "Default: numpy"
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model names and exit (default output: text, one per line)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="When used with --list-models, output JSON instead of text",
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
        help="JSON object of extra model constructor kwargs, e.g. '{\"k\": 1 }' (advanced)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Optional checkpoint path for checkpoint-backed models; sets model kwarg checkpoint_path",
    )
    parser.add_argument(
        "--pixel-segf1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compute pixel SegF1/bg-FPR under a single calibrated threshold. "
            "Requires dataset masks. Default: true"
        ),
    )
    parser.add_argument(
        "--pixel-normal-quantile",
        type=float,
        default=0.999,
        help="Quantile used to calibrate pixel threshold on normal pixels. Default: 0.999",
    )
    parser.add_argument(
        "--pixel-calibration-fraction",
        type=float,
        default=0.2,
        help="Fraction of train/good held out for pixel threshold calibration. Default: 0.2",
    )
    parser.add_argument(
        "--pixel-calibration-seed",
        type=int,
        default=0,
        help="RNG seed for calibration hold-out split. Default: 0",
    )
    parser.add_argument(
        "--pixel-postprocess",
        action="store_true",
        help="Apply standard postprocess to anomaly maps before computing pixel metrics",
    )
    parser.add_argument(
        "--corruptions",
        default="lighting,jpeg,blur,glare,geo_jitter",
        help=(
            "Comma-separated corruption names to run. "
            "Available: lighting,jpeg,blur,glare,geo_jitter. "
            "Default: lighting,jpeg,blur,glare,geo_jitter"
        ),
    )
    parser.add_argument(
        "--severities",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 5],
        help="Severity levels to evaluate (1..5). Default: 1 2 3 4 5",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed used to make corruptions deterministic. Default: 0",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional max number of training images used to fit/calibrate (for quick runs)",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional max number of test images evaluated (for quick runs)",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser


def load_benchmark_split(*args, **kwargs):
    # Lazy wrapper to keep CLI import light; also makes it easy to monkeypatch
    # in unit tests without importing cv2-heavy pipeline modules.
    from pyimgano.pipelines.mvtec_visa import load_benchmark_split as _load_benchmark_split

    return _load_benchmark_split(*args, **kwargs)


def _load_u8_rgb(path: str, *, resize: tuple[int, int]) -> NDArray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to load images.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = int(resize[0]), int(resize[1])
    if h > 0 and w > 0 and img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(img_rgb, dtype=np.uint8)


def _apply_resize_to_masks(masks: Optional[NDArray], *, resize: tuple[int, int]) -> Optional[NDArray]:
    if masks is None:
        return None

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

    target_h, target_w = int(resize[0]), int(resize[1])
    if target_h <= 0 or target_w <= 0:
        return masks

    if masks.shape[1:] == (target_h, target_w):
        return masks

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to resize masks.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    out: list[np.ndarray] = []
    for i in range(int(masks.shape[0])):
        m = np.asarray(masks[i])
        resized = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        out.append((np.asarray(resized) > 0).astype(np.uint8, copy=False))
    return np.stack(out, axis=0)


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


def _merge_checkpoint_path(user_kwargs: dict[str, Any], *, checkpoint_path: str | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return dict(user_kwargs)
    merged = dict(user_kwargs)
    merged.setdefault("checkpoint_path", str(checkpoint_path))
    return merged


def _resolve_preset_kwargs(preset: str | None, model: str) -> dict[str, Any]:
    if preset is None:
        return {}
    from pyimgano.cli import _resolve_preset_kwargs as _resolve  # lazy import

    return dict(_resolve(preset, model))


def _build_model_kwargs(
    model: str,
    *,
    user_kwargs: dict[str, Any],
    preset_kwargs: dict[str, Any],
    auto_kwargs: dict[str, Any],
) -> dict[str, Any]:
    from pyimgano.cli import _build_model_kwargs as _build  # lazy import

    return dict(
        _build(model, user_kwargs=user_kwargs, preset_kwargs=preset_kwargs, auto_kwargs=auto_kwargs)
    )


@dataclass(frozen=True)
class _NamedCorruption:
    name: str
    fn: Callable[..., tuple[NDArray, Optional[NDArray]]]

    def __call__(
        self,
        image: NDArray,
        mask: Optional[NDArray],
        *,
        severity: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray, Optional[NDArray]]:
        return self.fn(image, mask=mask, severity=int(severity), rng=rng)


def _resolve_corruptions(names: str) -> list[_NamedCorruption]:
    requested = [n.strip() for n in str(names).split(",") if n.strip()]
    available: dict[str, _NamedCorruption] = {
        "lighting": _NamedCorruption("lighting", apply_lighting),
        "jpeg": _NamedCorruption("jpeg", apply_jpeg),
        "blur": _NamedCorruption("blur", apply_blur),
        "glare": _NamedCorruption("glare", apply_glare),
        "geo_jitter": _NamedCorruption("geo_jitter", apply_geo_jitter),
    }

    out: list[_NamedCorruption] = []
    for name in requested:
        if name not in available:
            raise ValueError(
                f"Unknown corruption: {name!r}. Supported: {', '.join(sorted(available.keys()))}"
            )
        out.append(available[name])
    if not out:
        raise ValueError("--corruptions resolved to an empty list.")
    return out


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        # Ensure registry is populated.
        import pyimgano.models  # noqa: F401

        if bool(args.list_models):
            names = MODEL_REGISTRY.available()
            if bool(args.json):
                print(json.dumps({"models": list(names)}, indent=2, sort_keys=True))
            else:
                for name in names:
                    print(name)
            return 0

        if args.dataset is None or args.root is None or args.category is None:
            raise ValueError("--dataset/--root/--category are required unless --list-models is used.")

        resize_hw = (int(args.resize[0]), int(args.resize[1]))
        split = load_benchmark_split(
            dataset=args.dataset,
            root=str(args.root),
            category=str(args.category),
            resize=resize_hw,
            load_masks=True,
        )

        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        user_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

        entry = MODEL_REGISTRY.info(args.model)
        if bool(entry.metadata.get("requires_checkpoint", False)) and "checkpoint_path" not in user_kwargs:
            raise ValueError(
                f"Model {args.model!r} requires a checkpoint. "
                "Provide --checkpoint-path or set checkpoint_path in --model-kwargs."
            )

        preset_kwargs = _resolve_preset_kwargs(args.preset, args.model)
        model_kwargs = _build_model_kwargs(
            args.model,
            user_kwargs=user_kwargs,
            preset_kwargs=preset_kwargs,
            auto_kwargs={
                "device": args.device,
                "contamination": float(args.contamination),
                "pretrained": bool(args.pretrained),
            },
        )

        def _make_detector():
            return create_model(args.model, **model_kwargs)

        train_paths = list(split.train_paths)
        test_paths = list(split.test_paths)
        if args.limit_train is not None:
            train_paths = train_paths[: int(args.limit_train)]
        if args.limit_test is not None:
            test_paths = test_paths[: int(args.limit_test)]

        test_labels = np.asarray(split.test_labels[: len(test_paths)])
        test_masks = _apply_resize_to_masks(
            None if split.test_masks is None else np.asarray(split.test_masks[: len(test_paths)]),
            resize=resize_hw,
        )
        postprocess = AnomalyMapPostprocess() if bool(args.pixel_postprocess) else None
        requested_corruptions = str(args.corruptions)
        requested_severities = [int(s) for s in list(args.severities)]

        input_mode = str(args.input_mode)
        if input_mode == "numpy":
            train_inputs = [_load_u8_rgb(p, resize=resize_hw) for p in train_paths]
            test_inputs = [_load_u8_rgb(p, resize=resize_hw) for p in test_paths]
            corruptions: Sequence[_NamedCorruption] = _resolve_corruptions(requested_corruptions)
            corruption_mode = "full"
            corruptions_skipped_reason = None
        else:
            train_inputs = train_paths
            test_inputs = test_paths
            corruptions = []
            corruption_mode = "clean_only"
            corruptions_skipped_reason = (
                "input_mode=paths: corruptions are skipped because corruptions require numpy inputs."
            )

        detector = _make_detector()

        notes: list[str] = []
        pixel_segf1_enabled = bool(args.pixel_segf1)
        if pixel_segf1_enabled and test_masks is None:
            pixel_segf1_enabled = False
            notes.append("pixel_segf1 disabled because dataset split has no masks.")

        supports_maps = hasattr(detector, "predict_anomaly_map") or hasattr(detector, "get_anomaly_map")
        if pixel_segf1_enabled and not supports_maps:
            pixel_segf1_enabled = False
            notes.append(
                "pixel_segf1 disabled because detector does not expose "
                "predict_anomaly_map() or get_anomaly_map()."
            )

        if not pixel_segf1_enabled:
            # Skip mask/map computation entirely to keep runs cheap and avoid
            # requiring pixel-capable detectors.
            test_masks = None

        report = run_robustness_benchmark(
            detector,
            train_images=train_inputs,
            test_images=test_inputs,
            test_labels=test_labels,
            test_masks=test_masks,
            corruptions=corruptions,
            severities=requested_severities,
            seed=int(args.seed),
            pixel_segf1=pixel_segf1_enabled,
            pixel_threshold_strategy="normal_pixel_quantile",
            pixel_normal_quantile=float(args.pixel_normal_quantile),
            calibration_fraction=float(args.pixel_calibration_fraction),
            calibration_seed=int(args.pixel_calibration_seed),
            postprocess=postprocess,
        )
        report["input_mode"] = input_mode
        report["corruption_mode"] = corruption_mode
        report["requested_corruptions"] = requested_corruptions
        report["requested_severities"] = requested_severities
        if corruptions_skipped_reason is not None:
            report["corruptions_skipped_reason"] = corruptions_skipped_reason
        if notes:
            report["notes"] = list(notes)

        payload = {
            "dataset": args.dataset,
            "category": args.category,
            "model": args.model,
            "robustness": report,
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
