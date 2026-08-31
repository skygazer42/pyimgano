from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pyimgano.services.export_service as export_service

_FORMATS = ("native", "onnx", "torchscript", "openvino")
_VERIFICATION_LEVELS = ("reference-parity", "end-to-end")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-export",
        description="Export executable fitted-detector artifacts from a persisted run.",
    )
    parser.add_argument(
        "--from-run",
        "--train-dir",
        dest="from_run",
        required=True,
        help="Completed workbench run directory.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=_FORMATS,
        default=None,
        help="Artifact format. Repeat to export multiple formats. Default: native.",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="out_dir",
        default=None,
        help="Output root. Default: <run>/artifacts/exported.",
    )
    parser.add_argument("--category", default=None, help="Category for multi-category runs.")
    parser.add_argument(
        "--verification-level",
        choices=_VERIFICATION_LEVELS,
        default="reference-parity",
        help=(
            "Minimum reference parity is always mandatory. end-to-end adds the "
            "available image corpus and full policy verification."
        ),
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Allow verified formats to be published when another requested format is unsupported.",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Allow an integrity-verified checkpoint explicitly marked executable/trust-required.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing target.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _normalize_formats(values: list[str] | None) -> tuple[str, ...]:
    formats = tuple(str(value).strip().lower() for value in (values or ["native"]))
    if len(formats) != len(set(formats)):
        raise ValueError("Duplicate --format values are not allowed.")
    return formats


def _render_summary(payload: dict[str, Any]) -> None:
    print(f"status={payload.get('status', 'unknown')}")
    for item in payload.get("artifacts", []) or []:
        if not isinstance(item, dict):
            continue
        print(f"{item.get('format', 'artifact')}={item.get('path')}")
    for item in payload.get("failures", []) or []:
        if not isinstance(item, dict):
            continue
        print(
            f"failed.{item.get('format', 'unknown')}={item.get('reason', 'export_failed')}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = export_service.export_from_run(
            run_dir=str(args.from_run),
            formats=_normalize_formats(args.formats),
            out_dir=(str(args.out_dir) if args.out_dir is not None else None),
            category=(str(args.category) if args.category is not None else None),
            verification_level=str(args.verification_level).replace("-", "_"),
            strict=not bool(args.non_strict),
            trust_checkpoint=bool(args.trust_checkpoint),
            overwrite=bool(args.overwrite),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(str(exc), file=sys.stderr)
        return 2

    if bool(args.json):
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _render_summary(dict(payload))
    return 0 if str(payload.get("status", "ok")) != "failed" else 2


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
