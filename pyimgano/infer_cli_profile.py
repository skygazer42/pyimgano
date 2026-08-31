from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def format_infer_profile_summary(
    *,
    load_model: float,
    fit_calibrate: float,
    infer: float,
    artifacts: float,
    total: float,
    runtime_info: Mapping[str, Any] | None = None,
) -> str:
    fields = [
        f"load_model={float(load_model):.3f}s",
        f"fit_calibrate={float(fit_calibrate):.3f}s",
        f"infer={float(infer):.3f}s",
        f"artifacts={float(artifacts):.3f}s",
        f"total={float(total):.3f}s",
    ]
    info = dict(runtime_info or {})
    backend = info.get("backend")
    provider = info.get("selected_provider", info.get("device"))
    if backend is not None:
        fields.append(f"backend={backend}")
    if provider is not None:
        fields.append(f"provider={provider}")
    return "profile: " + " ".join(fields)


def build_infer_profile_payload(
    *,
    inputs: int,
    processed: int,
    errors: int,
    load_model: float,
    fit_calibrate: float,
    infer: float,
    artifacts: float,
    total: float,
    runtime_info: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "tool": "pyimgano-infer",
        "counts": {
            "inputs": int(inputs),
            "processed": int(processed),
            "errors": int(errors),
        },
        "timing_seconds": {
            "load_model": float(load_model),
            "fit_calibrate": float(fit_calibrate),
            "infer": float(infer),
            "artifacts": float(artifacts),
            "total": float(total),
        },
    }
    if runtime_info:
        payload["runtime"] = dict(runtime_info)
    return payload


def write_infer_profile_payload(path: str | Path, payload: dict[str, object]) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "build_infer_profile_payload",
    "format_infer_profile_summary",
    "write_infer_profile_payload",
]
