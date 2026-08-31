from __future__ import annotations

import json
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
    artifact_category: str | None = None
    artifact_format: str | None = None
    artifact_backend: str | None = None
    artifact_id: str | None = None
    device: str | None = None
    onnx_providers: str | None = None
    onnx_provider_options: str | None = None
    onnx_session_options: str | None = None
    trust_checkpoint: bool = False


def _bundle_has_runtime_artifacts(bundle_root: Path) -> bool:
    manifest_path = bundle_root / "bundle_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(isinstance(payload, Mapping) and payload.get("artifact_refs"))


def build_bundle_infer_argv(request: BundleInferenceBatchRequest) -> list[str]:
    bundle_root = Path(request.bundle_dir)
    results_path = Path(request.results_jsonl)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if _bundle_has_runtime_artifacts(bundle_root):
        argv = ["--artifact", str(bundle_root)]
        for option, value in (
            ("--artifact-category", request.artifact_category),
            ("--artifact-format", request.artifact_format),
            ("--artifact-backend", request.artifact_backend),
            ("--artifact-id", request.artifact_id),
            ("--device", request.device),
            ("--onnx-providers", request.onnx_providers),
            ("--onnx-provider-options", request.onnx_provider_options),
            ("--onnx-session-options", request.onnx_session_options),
        ):
            if value is not None:
                argv.extend([str(option), str(value)])
        if bool(request.trust_checkpoint):
            argv.append("--trust-checkpoint")
    else:
        argv = ["--infer-config", str(bundle_root / "infer_config.json")]
    argv.extend(["--save-jsonl", str(results_path)])
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
