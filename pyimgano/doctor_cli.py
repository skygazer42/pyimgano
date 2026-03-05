from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from pyimgano.utils.optional_deps import optional_import


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _check_module(
    *,
    module: str,
    dist: str | None = None,
    extra: str | None = None,
    purpose: str = "",
) -> dict[str, Any]:
    mod, err = optional_import(str(module))
    available = bool(mod is not None)
    module_version = getattr(mod, "__version__", None) if mod is not None else None
    dist_version = _dist_version(str(dist)) if dist else None

    install_hint = None
    if extra:
        install_hint = f"pip install 'pyimgano[{extra}]'"

    return {
        "module": str(module),
        "dist": (str(dist) if dist else None),
        "purpose": str(purpose),
        "extra": (str(extra) if extra else None),
        "install_hint": install_hint,
        "available": bool(available),
        "module_version": module_version,
        "dist_version": dist_version,
        "error": (str(err) if (not available and err is not None) else None),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-doctor",
        description="Print environment and optional dependency availability for pyimgano.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON payload to stdout (stable, machine-friendly).",
    )
    return parser


def _collect_payload() -> dict[str, Any]:
    import pyimgano

    # Baseline discovery is intentionally import-light.
    from pyimgano.baselines.suites import list_baseline_suites
    from pyimgano.baselines.sweeps import list_sweeps
    from pyimgano.cli_presets import list_model_presets

    optional_modules: list[dict[str, Any]] = [
        # Core deps (import sanity)
        _check_module(module="numpy", dist="numpy", purpose="core numerical backend"),
        _check_module(module="cv2", dist="opencv-python", purpose="image IO / preprocessing"),
        _check_module(module="sklearn", dist="scikit-learn", purpose="classical ML baselines"),
        # Optional extras (industrial deployment/backends)
        _check_module(
            module="torch", dist="torch", extra="torch", purpose="deep models / embeddings"
        ),
        _check_module(
            module="torchvision",
            dist="torchvision",
            extra="torch",
            purpose="torchvision backbones / patch embeddings",
        ),
        _check_module(
            module="onnxruntime", dist="onnxruntime", extra="onnx", purpose="ONNX inference"
        ),
        _check_module(module="onnx", dist="onnx", extra="onnx", purpose="ONNX export"),
        _check_module(
            module="openvino", dist="openvino", extra="openvino", purpose="OpenVINO inference"
        ),
        _check_module(
            module="skimage",
            dist="scikit-image",
            extra="skimage",
            purpose="SSIM/LBP/HOG/Gabor baselines",
        ),
        _check_module(module="numba", dist="numba", extra="numba", purpose="numba-accelerated ops"),
        _check_module(
            module="open_clip", dist="open_clip_torch", extra="clip", purpose="OpenCLIP backends"
        ),
        _check_module(module="faiss", dist="faiss-cpu", extra="faiss", purpose="fast kNN backend"),
        _check_module(
            module="anomalib",
            dist="anomalib",
            extra="anomalib",
            purpose="anomalib checkpoint wrappers",
        ),
    ]

    return {
        "tool": "pyimgano-doctor",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyimgano_version": str(getattr(pyimgano, "__version__", "")),
        "python": {
            "version": str(sys.version),
            "executable": str(sys.executable),
        },
        "platform": {
            "system": str(platform.system()),
            "release": str(platform.release()),
            "machine": str(platform.machine()),
        },
        "baselines": {
            "suites": list_baseline_suites(),
            "sweeps": list_sweeps(),
            "model_presets_count": int(len(list_model_presets())),
        },
        "optional_modules": optional_modules,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = _collect_payload()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Text output (human-friendly).
    print("pyimgano-doctor")
    print(f"pyimgano_version: {payload.get('pyimgano_version')}")
    py = payload.get("python", {}) or {}
    print(f"python: {py.get('version')}")
    plat = payload.get("platform", {}) or {}
    print(f"platform: {plat.get('system')} {plat.get('release')} ({plat.get('machine')})")

    baselines = payload.get("baselines", {}) or {}
    suites = baselines.get("suites", []) or []
    sweeps = baselines.get("sweeps", []) or []
    print(f"suites: {', '.join(suites)}")
    print(f"sweeps: {', '.join(sweeps)}")

    print("optional_modules:")
    for m in payload.get("optional_modules", []) or []:
        name = str(m.get("module"))
        ok = bool(m.get("available"))
        ver = m.get("module_version") or m.get("dist_version") or None
        hint = m.get("install_hint")
        if ok:
            suffix = f" ({ver})" if ver else ""
            print(f"- {name}: OK{suffix}")
        else:
            msg = f"- {name}: MISSING"
            if hint:
                msg += f" → {hint}"
            print(msg)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
