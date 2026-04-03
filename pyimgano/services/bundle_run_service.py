from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BundleInferenceBatchRequest:
    bundle_dir: str | Path
    input_records: Sequence[Mapping[str, Any]]
    results_jsonl: str | Path
    defects_enabled: bool = False
    masks_dir: str | None = None
    overlays_dir: str | None = None
    defects_regions_jsonl: str | None = None


def build_bundle_infer_argv(request: BundleInferenceBatchRequest) -> list[str]:
    bundle_root = Path(request.bundle_dir)
    results_path = Path(request.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    argv = [
        "--infer-config",
        str(bundle_root / "infer_config.json"),
        "--save-jsonl",
        str(results_path),
    ]
    for input_record in request.input_records:
        argv.extend(["--input", str(input_record["resolved_input_path"])])

    if bool(request.defects_enabled):
        argv.append("--defects")
    if request.masks_dir is not None:
        argv.extend(["--save-masks", str(request.masks_dir)])
    if request.overlays_dir is not None:
        argv.extend(["--save-overlays", str(request.overlays_dir)])
    if request.defects_regions_jsonl is not None:
        argv.extend(["--defects-regions-jsonl", str(request.defects_regions_jsonl)])
    return argv


def run_bundle_inference_batch(
    request: BundleInferenceBatchRequest,
    *,
    infer_main_impl: Callable[[list[str]], int] | None = None,
) -> int:
    if infer_main_impl is None:
        import pyimgano.infer_cli as infer_cli

        infer_main_impl = infer_cli.main
    return int(infer_main_impl(build_bundle_infer_argv(request)))


__all__ = [
    "BundleInferenceBatchRequest",
    "build_bundle_infer_argv",
    "run_bundle_inference_batch",
]
