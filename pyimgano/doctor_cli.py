from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
from typing import Any

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


def _can_find_root(module_root: str) -> bool:
    """Best-effort root-module existence check (no import side effects)."""

    return find_spec(str(module_root)) is not None


_EXTRA_ROOT_MODULES: dict[str, tuple[str, ...]] = {
    "torch": ("torch", "torchvision"),
    "onnx": ("onnxruntime", "onnx", "onnxscript"),
    "openvino": ("openvino",),
    "skimage": ("skimage",),
    "numba": ("numba",),
    "faiss": ("faiss",),
    # Extras that imply torch.
    "clip": ("open_clip", "torch"),
    "anomalib": ("anomalib", "torch"),
    "mamba": ("mamba_ssm", "torch"),
}


def _extra_installed(extra: str) -> bool:
    roots = _EXTRA_ROOT_MODULES.get(str(extra), (str(extra),))
    return all(_can_find_root(root) for root in roots)


def _can_import_root(module_root: str) -> bool:
    """Best-effort import check (catches broken wheels / missing shared libs)."""

    mod, _err = optional_import(str(module_root))
    return bool(mod is not None)


def _extra_importable(extra: str) -> bool:
    roots = _EXTRA_ROOT_MODULES.get(str(extra), (str(extra),))
    return all(_can_import_root(root) for root in roots)


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
            missing = [e for e in requires if not _extra_installed(e)]
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
    return parser


def _build_require_extras_check(required_extras: list[str]) -> dict[str, Any]:
    required = _split_csv_args(required_extras)
    missing = [e for e in required if not _extra_importable(e)]
    ok = len(missing) == 0

    install_hint = None
    if missing:
        extra_spec = ",".join(sorted({str(e) for e in missing}))
        install_hint = f"pip install 'pyimgano[{extra_spec}]'"

    return {
        "required": required,
        "missing": missing,
        "ok": bool(ok),
        "install_hint": install_hint,
    }


def _collect_payload(
    *, suites_to_check: list[str] | None = None, require_extras: list[str] | None = None
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

    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = _collect_payload(suites_to_check=args.suite, require_extras=args.require_extras)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
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
