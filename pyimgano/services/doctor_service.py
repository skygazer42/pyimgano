from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from pyimgano.presets.catalog import list_model_presets
from pyimgano.services.discovery_service import (
    list_baseline_suites_payload,
    list_sweeps_payload,
)
from pyimgano.utils.extras import extra_importable, extra_installed, extras_install_hint
from pyimgano.utils.optional_deps import optional_import


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def split_csv_args(values: list[str] | None) -> list[str]:
    if not values:
        return []

    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            stripped = str(part).strip()
            if stripped:
                out.append(stripped)
    return out


def check_module(
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


def build_suite_checks(suite_names: list[str]) -> dict[str, Any]:
    from pyimgano.baselines.suites import get_baseline_suite, resolve_suite_baselines

    out: dict[str, Any] = {}
    for suite_name in suite_names:
        suite = get_baseline_suite(str(suite_name))
        baselines = resolve_suite_baselines(str(suite_name))

        baseline_payloads: list[dict[str, Any]] = []
        missing_extras_union: set[str] = set()
        runnable_count = 0

        for baseline in baselines:
            requires = [str(x) for x in tuple(getattr(baseline, "requires_extras", ()))]
            missing = [extra for extra in requires if not extra_installed(extra)]
            runnable = len(missing) == 0
            if runnable:
                runnable_count += 1
            else:
                missing_extras_union.update(str(extra) for extra in missing)

            install_hint = None
            if missing:
                extra_spec = ",".join(sorted(set(missing)))
                install_hint = f"pip install 'pyimgano[{extra_spec}]'"

            baseline_payloads.append(
                {
                    "name": str(baseline.name),
                    "model": str(baseline.model),
                    "optional": bool(baseline.optional),
                    "requires_extras": requires,
                    "missing_extras": missing,
                    "runnable": bool(runnable),
                    "install_hint": install_hint,
                    "description": str(baseline.description),
                }
            )

        out[str(suite.name)] = {
            "suite": str(suite.name),
            "description": str(suite.description),
            "summary": {
                "total": int(len(baselines)),
                "runnable": int(runnable_count),
                "skipped": int(len(baselines) - runnable_count),
                "missing_extras": sorted(missing_extras_union),
            },
            "baselines": baseline_payloads,
        }

    return out


def build_require_extras_check(required_extras: list[str] | None) -> dict[str, Any]:
    required = split_csv_args(required_extras)
    missing = [extra for extra in required if not extra_importable(extra)]
    ok = len(missing) == 0

    install_hint = None
    if missing:
        install_hint = extras_install_hint(missing)

    return {
        "required": required,
        "missing": missing,
        "ok": bool(ok),
        "install_hint": install_hint,
    }


def build_accelerator_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    torch_mod, torch_err = optional_import("torch")
    if torch_mod is None:
        checks["torch"] = {
            "available": False,
            "install_hint": extras_install_hint(["torch"]),
            "error": str(torch_err) if torch_err is not None else "missing",
        }
    else:
        torch = torch_mod
        payload: dict[str, Any] = {
            "available": True,
            "install_hint": None,
            "torch_version": getattr(torch, "__version__", None),
        }

        cuda_compiled = getattr(getattr(torch, "version", None), "cuda", None) is not None
        cuda_available = False
        try:
            cuda_available = bool(getattr(torch, "cuda").is_available())
        except Exception as exc:  # noqa: BLE001 - best-effort diagnostics
            payload["cuda_error"] = str(exc)

        cuda: dict[str, Any] = {
            "compiled": bool(cuda_compiled),
            "available": bool(cuda_available),
            "version": getattr(getattr(torch, "version", None), "cuda", None),
        }

        if cuda_available:
            try:
                cuda["device_count"] = int(torch.cuda.device_count())
            except Exception as exc:  # noqa: BLE001
                cuda["device_count_error"] = str(exc)

            devices: list[dict[str, Any]] = []
            try:
                for index in range(int(cuda.get("device_count") or 0)):
                    try:
                        props = torch.cuda.get_device_properties(int(index))
                        devices.append(
                            {
                                "index": int(index),
                                "name": getattr(props, "name", None),
                                "total_memory": int(getattr(props, "total_memory", 0)),
                                "major": int(getattr(props, "major", 0)),
                                "minor": int(getattr(props, "minor", 0)),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        devices.append({"index": int(index), "error": str(exc)})
                cuda["devices"] = devices
            except Exception as exc:  # noqa: BLE001
                cuda["devices_error"] = str(exc)

            try:
                cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
                version = cudnn.version() if cudnn is not None and callable(cudnn.version) else None
                cuda["cudnn_version"] = int(version) if version is not None else None
            except Exception as exc:  # noqa: BLE001
                cuda["cudnn_error"] = str(exc)

        payload["cuda"] = cuda

        mps_available = False
        try:
            backends = getattr(torch, "backends", None)
            mps = getattr(backends, "mps", None) if backends is not None else None
            if mps is not None and callable(getattr(mps, "is_available", None)):
                mps_available = bool(mps.is_available())
        except Exception:
            mps_available = False
        payload["mps"] = {"available": bool(mps_available)}

        checks["torch"] = payload

    ort_mod, ort_err = optional_import("onnxruntime")
    if ort_mod is None:
        checks["onnxruntime"] = {
            "available": False,
            "install_hint": extras_install_hint(["onnx"]),
            "error": str(ort_err) if ort_err is not None else "missing",
        }
    else:
        providers = None
        providers_error = None
        try:
            providers = list(ort_mod.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            providers_error = str(exc)

        checks["onnxruntime"] = {
            "available": True,
            "install_hint": None,
            "onnxruntime_version": getattr(ort_mod, "__version__", None),
            "available_providers": providers,
            "providers_error": providers_error,
        }

    ov_mod, ov_err = optional_import("openvino")
    if ov_mod is None:
        checks["openvino"] = {
            "available": False,
            "install_hint": extras_install_hint(["openvino"]),
            "error": str(ov_err) if ov_err is not None else "missing",
        }
    else:
        devices = None
        devices_error = None
        try:
            try:
                from openvino.runtime import Core as core_cls  # type: ignore[import-not-found]
            except Exception:
                core_cls = getattr(getattr(ov_mod, "runtime", None), "Core", None)  # type: ignore[assignment]
            if core_cls is None:
                raise RuntimeError("openvino.runtime.Core not found")

            core = core_cls()
            if hasattr(core, "available_devices"):
                devices = list(core.available_devices)
            elif hasattr(core, "get_available_devices"):
                devices = list(core.get_available_devices())
            else:
                raise RuntimeError("OpenVINO Core does not expose available devices")
        except Exception as exc:  # noqa: BLE001
            devices_error = str(exc)

        checks["openvino"] = {
            "available": True,
            "install_hint": None,
            "openvino_version": getattr(ov_mod, "__version__", None),
            "devices": devices,
            "devices_error": devices_error,
        }

    return checks


def collect_doctor_payload(
    *,
    suites_to_check: list[str] | None = None,
    require_extras: list[str] | None = None,
    accelerators: bool = False,
) -> dict[str, Any]:
    import pyimgano

    optional_modules: list[dict[str, Any]] = [
        check_module(module="numpy", dist="numpy", purpose="core numerical backend"),
        check_module(module="cv2", dist="opencv-python", purpose="image IO / preprocessing"),
        check_module(module="sklearn", dist="scikit-learn", purpose="classical ML baselines"),
        check_module(module="torch", dist="torch", extra="torch", purpose="deep models / embeddings"),
        check_module(
            module="torchvision",
            dist="torchvision",
            extra="torch",
            purpose="torchvision backbones / patch embeddings",
        ),
        check_module(
            module="onnxruntime",
            dist="onnxruntime",
            extra="onnx",
            purpose="ONNX inference",
        ),
        check_module(module="onnx", dist="onnx", extra="onnx", purpose="ONNX export"),
        check_module(
            module="onnxscript",
            dist="onnxscript",
            extra="onnx",
            purpose="torch.onnx.export helper (required by newer torch versions)",
        ),
        check_module(
            module="openvino",
            dist="openvino",
            extra="openvino",
            purpose="OpenVINO inference",
        ),
        check_module(
            module="skimage",
            dist="scikit-image",
            extra="skimage",
            purpose="SSIM/LBP/HOG/Gabor baselines",
        ),
        check_module(
            module="numba",
            dist="numba",
            extra="numba",
            purpose="numba-accelerated ops",
        ),
        check_module(
            module="open_clip",
            dist="open_clip_torch",
            extra="clip",
            purpose="OpenCLIP backends",
        ),
        check_module(
            module="faiss",
            dist="faiss-cpu",
            extra="faiss",
            purpose="fast kNN backend",
        ),
        check_module(
            module="anomalib",
            dist="anomalib",
            extra="anomalib",
            purpose="anomalib checkpoint wrappers",
        ),
    ]

    payload: dict[str, Any] = {
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
            "suites": list_baseline_suites_payload(),
            "sweeps": list_sweeps_payload(),
            "model_presets_count": int(len(list_model_presets())),
        },
        "optional_modules": optional_modules,
    }

    suites = split_csv_args(suites_to_check)
    if suites:
        payload["suite_checks"] = build_suite_checks(suites)

    if require_extras:
        payload["require_extras"] = build_require_extras_check(require_extras)

    if bool(accelerators):
        payload["accelerators"] = build_accelerator_checks()

    return payload


__all__ = [
    "build_accelerator_checks",
    "build_require_extras_check",
    "build_suite_checks",
    "check_module",
    "collect_doctor_payload",
    "split_csv_args",
]
