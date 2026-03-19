from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pyimgano.cli_listing as cli_listing
import pyimgano.cli_output as cli_output
from pyimgano.cli_common import merge_checkpoint_path, parse_model_kwargs
from pyimgano.reporting.report import save_run_report
from pyimgano.services.benchmark_service import PixelPostprocessConfig


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
    # Industrial default: keep CLIs offline-safe and avoid implicit weight downloads.
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
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
def _parse_model_kwargs(text: str | None) -> dict[str, Any]:
    return parse_model_kwargs(text)


def _merge_checkpoint_path(
    user_kwargs: dict[str, Any], *, checkpoint_path: str | None
) -> dict[str, Any]:
    return merge_checkpoint_path(user_kwargs, checkpoint_path=checkpoint_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else None)

    try:
        # Ensure registry is populated.
        import pyimgano.models  # noqa: F401
        import pyimgano.services.discovery_service as discovery_service

        if bool(args.list_models):
            names = discovery_service.list_discovery_model_names()
            return cli_listing.emit_listing(
                names,
                json_output=bool(args.json),
                json_payload={"models": names},
            )

        if args.dataset is None or args.root is None or args.category is None:
            raise ValueError(
                "--dataset/--root/--category are required unless --list-models is used."
            )
        from pyimgano.services.robustness_service import (
            RobustnessRunRequest,
            run_robustness_request,
        )

        postprocess = PixelPostprocessConfig() if bool(args.pixel_postprocess) else None
        user_kwargs = _parse_model_kwargs(args.model_kwargs)
        user_kwargs = _merge_checkpoint_path(user_kwargs, checkpoint_path=args.checkpoint_path)

        payload = run_robustness_request(
            RobustnessRunRequest(
                dataset=str(args.dataset),
                root=str(args.root),
                category=str(args.category),
                model=str(args.model),
                resize=(int(args.resize[0]), int(args.resize[1])),
                input_mode=str(args.input_mode),
                preset=(str(args.preset) if args.preset is not None else None),
                device=str(args.device),
                contamination=float(args.contamination),
                pretrained=bool(args.pretrained),
                model_kwargs=user_kwargs,
                checkpoint_path=None,
                pixel_segf1=bool(args.pixel_segf1),
                pixel_normal_quantile=float(args.pixel_normal_quantile),
                pixel_calibration_fraction=float(args.pixel_calibration_fraction),
                pixel_calibration_seed=int(args.pixel_calibration_seed),
                pixel_postprocess=postprocess,
                corruptions=str(args.corruptions),
                severities=[int(s) for s in args.severities],
                seed=int(args.seed),
                limit_train=(int(args.limit_train) if args.limit_train is not None else None),
                limit_test=(int(args.limit_test) if args.limit_test is not None else None),
            )
        )

        if args.output:
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
