from __future__ import annotations

import argparse

import pyimgano.cli_output as cli_output
import pyimgano.services.doctor_service as doctor_service


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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = doctor_service.collect_doctor_payload(
            suites_to_check=args.suite,
            require_extras=args.require_extras,
            accelerators=bool(getattr(args, "accelerators", False)),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cli_output.print_cli_error(exc)
        return 1

    if bool(args.json):
        rc = cli_output.emit_json(payload)
        req = payload.get("require_extras")
        if isinstance(req, dict) and req.get("ok") is False:
            return 1
        return rc

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
