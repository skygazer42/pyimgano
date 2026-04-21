from __future__ import annotations

import json
from pathlib import Path


def format_infer_profile_summary(
    *,
    load_model: float,
    fit_calibrate: float,
    infer: float,
    artifacts: float,
    total: float,
) -> str:
    return "profile: " + " ".join(
        [
            f"load_model={float(load_model):.3f}s",
            f"fit_calibrate={float(fit_calibrate):.3f}s",
            f"infer={float(infer):.3f}s",
            f"artifacts={float(artifacts):.3f}s",
            f"total={float(total):.3f}s",
        ]
    )


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
) -> dict[str, object]:
    return {
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


def write_infer_profile_payload(path: str | Path, payload: dict[str, object]) -> None:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "build_infer_profile_payload",
    "format_infer_profile_summary",
    "write_infer_profile_payload",
]
