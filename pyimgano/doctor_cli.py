from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from pyimgano.utils.extras import extra_importable, extra_installed, extras_install_hint
from pyimgano.utils.optional_deps import optional_import


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _split_csv_args(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for part in str(raw).split(","):
            p = str(part).strip()
            if p:
                out.append(p)
    return out


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


def _build_suite_checks(suite_names: list[str]) -> dict[str, Any]:
    from pyimgano.baselines.suites import get_baseline_suite, resolve_suite_baselines

    out: dict[str, Any] = {}
    for suite_name in suite_names:
        suite = get_baseline_suite(str(suite_name))
        baselines = resolve_suite_baselines(str(suite_name))

        baseline_payloads: list[dict[str, Any]] = []
        missing_extras_union: set[str] = set()
        runnable_count = 0

        for b in baselines:
            requires = [str(x) for x in tuple(getattr(b, "requires_extras", ()))]
            missing = [e for e in requires if not extra_installed(e)]
            runnable = len(missing) == 0
            if runnable:
                runnable_count += 1
            else:
                missing_extras_union.update(str(e) for e in missing)

            hint = None
            if missing:
                extra_spec = ",".join(sorted(set(missing)))
                hint = f"pip install 'pyimgano[{extra_spec}]'"

            baseline_payloads.append(
                {
                    "name": str(b.name),
                    "model": str(b.model),
                    "optional": bool(b.optional),
                    "requires_extras": requires,
                    "missing_extras": missing,
                    "runnable": bool(runnable),
                    "install_hint": hint,
                    "description": str(b.description),
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
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help=(
            "Optional suite name(s) to check for missing extras. Repeatable and comma-separated. "
            "Example: --suite industrial-v4"
        ),
    )
    parser.add_argument(
        "--require-extras",
        action="append",
        default=None,
        help=(
            "Require one or more extras to be available (for CI/deploy sanity checks). "
            "Comma-separated or repeatable. Exits with code 1 if any are missing. "
            "Example: --require-extras torch,skimage"
        ),
    )
    parser.add_argument(
        "--accelerators",
        action="store_true",
        help=(
            "Include best-effort accelerator runtime checks (torch CUDA/MPS, "
            "onnxruntime providers, openvino devices)."
        ),
    )
    return parser


def _build_require_extras_check(required_extras: list[str]) -> dict[str, Any]:
    required = _split_csv_args(required_extras)
    missing = [e for e in required if not extra_importable(e)]
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


def _build_accelerator_checks() -> dict[str, Any]:
    """Best-effort hardware/runtime accelerator checks.

    Notes
    -----
    This function should be safe to call even when optional runtimes are not
    installed. It must not raise; it returns structured error payloads.
    """

    checks: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # torch
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

        # CUDA
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
                for i in range(int(cuda.get("device_count") or 0)):
                    try:
                        prop = torch.cuda.get_device_properties(int(i))
                        devices.append(
                            {
                                "index": int(i),
                                "name": getattr(prop, "name", None),
                                "total_memory": int(getattr(prop, "total_memory", 0)),
                                "major": int(getattr(prop, "major", 0)),
                                "minor": int(getattr(prop, "minor", 0)),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        devices.append({"index": int(i), "error": str(exc)})
                cuda["devices"] = devices
            except Exception as exc:  # noqa: BLE001
                cuda["devices_error"] = str(exc)

            # cuDNN version (best-effort)
            try:
                cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
                ver = cudnn.version() if cudnn is not None and callable(cudnn.version) else None
                cuda["cudnn_version"] = int(ver) if ver is not None else None
            except Exception as exc:  # noqa: BLE001
                cuda["cudnn_error"] = str(exc)

        payload["cuda"] = cuda

        # MPS (Apple Silicon)
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

    # ------------------------------------------------------------------
    # onnxruntime
    ort_mod, ort_err = optional_import("onnxruntime")
    if ort_mod is None:
        checks["onnxruntime"] = {
            "available": False,
            "install_hint": extras_install_hint(["onnx"]),
            "error": str(ort_err) if ort_err is not None else "missing",
        }
    else:
        ort = ort_mod
        providers = None
        providers_error = None
        try:
            providers = list(ort.get_available_providers())
        except Exception as exc:  # noqa: BLE001
            providers_error = str(exc)

        checks["onnxruntime"] = {
            "available": True,
            "install_hint": None,
            "onnxruntime_version": getattr(ort, "__version__", None),
            "available_providers": providers,
            "providers_error": providers_error,
        }

    # ------------------------------------------------------------------
    # openvino
    ov_mod, ov_err = optional_import("openvino")
    if ov_mod is None:
        checks["openvino"] = {
            "available": False,
            "install_hint": extras_install_hint(["openvino"]),
            "error": str(ov_err) if ov_err is not None else "missing",
        }
    else:
        openvino = ov_mod
        devices = None
        devices_error = None
        try:
            try:
                from openvino.runtime import Core  # type: ignore[import-not-found]
            except Exception:
                # Some versions expose `openvino.runtime` as an attribute.
                Core = getattr(getattr(openvino, "runtime", None), "Core", None)  # type: ignore[assignment]
            if Core is None:
                raise RuntimeError("openvino.runtime.Core not found")

            core = Core()
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
            "openvino_version": getattr(openvino, "__version__", None),
            "devices": devices,
            "devices_error": devices_error,
        }

    return checks


def _collect_payload(
    *,
    suites_to_check: list[str] | None = None,
    require_extras: list[str] | None = None,
    accelerators: bool = False,
) -> dict[str, Any]:
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
            module="onnxscript",
            dist="onnxscript",
            extra="onnx",
            purpose="torch.onnx.export helper (required by newer torch versions)",
        ),
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
            "suites": list_baseline_suites(),
            "sweeps": list_sweeps(),
            "model_presets_count": int(len(list_model_presets())),
        },
        "optional_modules": optional_modules,
    }

    suites = _split_csv_args(suites_to_check)
    if suites:
        payload["suite_checks"] = _build_suite_checks(suites)

    if require_extras:
        payload["require_extras"] = _build_require_extras_check(require_extras)

    if bool(accelerators):
        payload["accelerators"] = _build_accelerator_checks()

    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = _collect_payload(
            suites_to_check=args.suite,
            require_extras=args.require_extras,
            accelerators=bool(getattr(args, "accelerators", False)),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(args.json):
        print(json.dumps(payload, sort_keys=True))
        req = payload.get("require_extras")
        if isinstance(req, dict) and req.get("ok") is False:
            return 1
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

    suite_checks = payload.get("suite_checks", None)
    if isinstance(suite_checks, dict) and suite_checks:
        print("suite_checks:")
        for suite_name in sorted(suite_checks):
            info = suite_checks.get(suite_name) or {}
            summary = info.get("summary", {}) or {}
            total = summary.get("total", None)
            runnable = summary.get("runnable", None)
            missing = summary.get("missing_extras", []) or []
            suffix = ""
            if missing:
                suffix = f" (missing extras: {', '.join(missing)})"
            print(f"- {suite_name}: runnable {runnable}/{total}{suffix}")

    req = payload.get("require_extras")
    if isinstance(req, dict) and req.get("required"):
        missing = req.get("missing", []) or []
        if missing:
            hint = req.get("install_hint")
            msg = f"require_extras: MISSING ({', '.join(str(x) for x in missing)})"
            if hint:
                msg += f" → {hint}"
            print(msg)
        else:
            print("require_extras: OK")

    accelerators = payload.get("accelerators")
    if isinstance(accelerators, dict) and accelerators:
        print("accelerators:")
        t = accelerators.get("torch") or {}
        if isinstance(t, dict):
            if bool(t.get("available")):
                cuda = t.get("cuda") or {}
                cuda_ok = bool(isinstance(cuda, dict) and cuda.get("available"))
                count = None
                if isinstance(cuda, dict):
                    count = cuda.get("device_count")
                suffix = ""
                if cuda_ok:
                    suffix = f" (cuda devices: {count})"
                print(f"- torch: OK{suffix}")
            else:
                hint = t.get("install_hint")
                msg = "- torch: MISSING"
                if hint:
                    msg += f" → {hint}"
                print(msg)

        ort = accelerators.get("onnxruntime") or {}
        if isinstance(ort, dict):
            if bool(ort.get("available")):
                prov = ort.get("available_providers") or []
                suffix = ""
                if isinstance(prov, list) and prov:
                    suffix = f" (providers: {', '.join(str(x) for x in prov)})"
                print(f"- onnxruntime: OK{suffix}")
            else:
                hint = ort.get("install_hint")
                msg = "- onnxruntime: MISSING"
                if hint:
                    msg += f" → {hint}"
                print(msg)

        ov = accelerators.get("openvino") or {}
        if isinstance(ov, dict):
            if bool(ov.get("available")):
                devs = ov.get("devices") or []
                suffix = ""
                if isinstance(devs, list) and devs:
                    suffix = f" (devices: {', '.join(str(x) for x in devs)})"
                print(f"- openvino: OK{suffix}")
            else:
                hint = ov.get("install_hint")
                msg = "- openvino: MISSING"
                if hint:
                    msg += f" → {hint}"
                print(msg)

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

    if isinstance(req, dict) and req.get("ok") is False:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
