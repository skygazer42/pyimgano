from __future__ import annotations

from typing import Any

from pyimgano.utils.extras import extra_importable, extras_install_hint
from pyimgano.utils.optional_deps import optional_import


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
        checks["torch"] = {
            "available": True,
            "install_hint": None,
            "torch_version": getattr(torch_mod, "__version__", None),
        }

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
        checks["openvino"] = {
            "available": True,
            "install_hint": None,
            "openvino_version": getattr(ov_mod, "__version__", None),
            "devices": None,
            "devices_error": None,
        }

    return checks


__all__ = [
    "build_accelerator_checks",
    "build_require_extras_check",
    "split_csv_args",
]
