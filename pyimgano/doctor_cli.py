from __future__ import annotations

import argparse
import importlib
from typing import Any

import pyimgano.cli_output as cli_output
import pyimgano.doctor_rendering as doctor_rendering


class _LazyModuleProxy:
    def __init__(self, module_path: str) -> None:
        self._module_path = str(module_path)
        self._module = None
        self._overrides: dict[str, Any] = {}

    def _load(self):
        module = self._module
        if module is None:
            module = importlib.import_module(self._module_path)
            self._module = module
        return module

    def __getattr__(self, name: str) -> Any:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._load(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._overrides[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        if name in self._overrides:
            del self._overrides[name]
            return
        delattr(self._load(), name)


doctor_service = _LazyModuleProxy("pyimgano.services.doctor_service")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyimgano-doctor",
        description="Print environment and optional dependency availability for pyimgano.",
    )
    readiness = parser.add_mutually_exclusive_group(required=False)
    readiness.add_argument(
        "--run-dir",
        default=None,
        help="Optional run directory to evaluate for deployment readiness.",
    )
    readiness.add_argument(
        "--deploy-bundle",
        default=None,
        help="Optional deploy bundle directory to validate for deployment readiness.",
    )
    readiness.add_argument(
        "--publication-target",
        default=None,
        help="Optional suite export directory or leaderboard metadata to evaluate for publication readiness.",
    )
    readiness.add_argument(
        "--dataset-target",
        default=None,
        help="Optional dataset root or manifest to evaluate for industrial AD readiness.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["first-run", "deploy-smoke", "benchmark", "deploy", "publish"],
        help="Optional guided workflow profile to validate and summarize.",
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
        "--recommend-extras",
        action="store_true",
        help="Emit task-oriented extras recommendations for a command or model.",
    )
    parser.add_argument(
        "--for-command",
        default=None,
        help="Command name used with --recommend-extras. Example: --for-command export-onnx",
    )
    parser.add_argument(
        "--for-model",
        default=None,
        help="Model name used with --recommend-extras. Example: --for-model vision_patchcore",
    )
    parser.add_argument(
        "--accelerators",
        action="store_true",
        help=(
            "Include best-effort accelerator runtime checks (torch CUDA/MPS, "
            "onnxruntime providers, openvino devices)."
        ),
    )
    parser.add_argument(
        "--check-bundle-hashes",
        action="store_true",
        help="When evaluating run/bundle readiness, verify recorded bundle hashes when available.",
    )
    parser.add_argument(
        "--dataset",
        default="auto",
        help="Dataset converter name, 'manifest', or 'auto' for --dataset-target.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category for --dataset-target when the source layout is category-scoped.",
    )
    parser.add_argument(
        "--root-fallback",
        default=None,
        help="Optional root fallback for manifest-backed --dataset-target checks.",
    )
    parser.add_argument(
        "--objective",
        default=None,
        choices=["balanced", "latency", "localization"],
        help="Recommendation objective for --dataset-target selection.",
    )
    parser.add_argument(
        "--allow-upstream",
        default=None,
        choices=["native-only", "native+wrapped"],
        help="Whether dataset-target recommendations may include upstream checkpoint wrappers.",
    )
    parser.add_argument(
        "--selection-profile",
        default=None,
        choices=["balanced", "benchmark-parity", "cpu-screening", "deploy-readiness"],
        help="Selection profile preset for --dataset-target recommendation and parity surfacing.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Maximum number of dataset-target recommendations to emit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        payload = doctor_service.collect_doctor_payload(
            suites_to_check=args.suite,
            require_extras=args.require_extras,
            accelerators=bool(getattr(args, "accelerators", False)),
            profile=(str(args.profile) if getattr(args, "profile", None) is not None else None),
            run_dir=(str(args.run_dir) if getattr(args, "run_dir", None) is not None else None),
            deploy_bundle=(
                str(args.deploy_bundle)
                if getattr(args, "deploy_bundle", None) is not None
                else None
            ),
            publication_target=(
                str(args.publication_target)
                if getattr(args, "publication_target", None) is not None
                else None
            ),
            dataset_target=(
                str(args.dataset_target)
                if getattr(args, "dataset_target", None) is not None
                else None
            ),
            dataset=str(getattr(args, "dataset", "auto")),
            category=(str(args.category) if getattr(args, "category", None) is not None else None),
            root_fallback=(
                str(args.root_fallback)
                if getattr(args, "root_fallback", None) is not None
                else None
            ),
            objective=(
                str(args.objective) if getattr(args, "objective", None) is not None else None
            ),
            allow_upstream=(
                str(args.allow_upstream)
                if getattr(args, "allow_upstream", None) is not None
                else None
            ),
            selection_profile=(
                str(args.selection_profile)
                if getattr(args, "selection_profile", None) is not None
                else None
            ),
            topk=(int(args.topk) if getattr(args, "topk", None) is not None else None),
            recommend_extras=bool(getattr(args, "recommend_extras", False)),
            for_command=(
                str(args.for_command) if getattr(args, "for_command", None) is not None else None
            ),
            for_model=(
                str(args.for_model) if getattr(args, "for_model", None) is not None else None
            ),
            check_bundle_hashes=bool(getattr(args, "check_bundle_hashes", False)),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        cli_output.print_cli_error(exc)
        return 1

    if bool(args.json):
        rc = cli_output.emit_json(payload, indent=None)
        req = payload.get("require_extras")
        if isinstance(req, dict) and req.get("ok") is False:
            return 1
        readiness = payload.get("readiness")
        if isinstance(readiness, dict) and str(readiness.get("status")) == "error":
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
            info = dict(suite_checks.get(suite_name) or {})
            print(
                doctor_rendering.format_suite_check_line(
                    suite_name=str(suite_name),
                    info=info,
                )
            )

    req = payload.get("require_extras")
    if isinstance(req, dict) and req.get("required"):
        line = doctor_rendering.format_require_extras_line(dict(req))
        if line is not None:
            print(line)

    extras_recommendation = payload.get("extras_recommendation")
    if isinstance(extras_recommendation, dict):
        for line in doctor_rendering.format_extra_recommendation_lines(dict(extras_recommendation)):
            print(line)

    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        for line in doctor_rendering.format_readiness_lines(dict(readiness)):
            print(line)

    workflow_profile = payload.get("workflow_profile")
    formatter = getattr(doctor_rendering, "_format_workflow_profile_lines", None)
    if isinstance(workflow_profile, dict) and callable(formatter):
        for line in formatter(dict(workflow_profile)):
            print(line)

    dataset_profile = payload.get("dataset_profile")
    if isinstance(dataset_profile, dict):
        print("dataset_profile:")
        print(f"- total_records: {dataset_profile.get('total_records')}")
        print(f"- train_count: {dataset_profile.get('train_count')}")
        print(f"- test_count: {dataset_profile.get('test_count')}")
        print(f"- pixel_metrics_available: {dataset_profile.get('pixel_metrics_available')}")
        print(f"- fewshot_risk: {dataset_profile.get('fewshot_risk')}")

    selection_context = payload.get("selection_context")
    if isinstance(selection_context, dict):
        print("selection_context:")
        print(f"- objective: {selection_context.get('objective')}")
        print(f"- allow_upstream: {selection_context.get('allow_upstream')}")
        print(f"- topk: {selection_context.get('topk')}")

    recommendations = payload.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        print("recommendations:")
        for item in recommendations:
            if not isinstance(item, dict):
                continue
            preset = item.get("preset") or item.get("model")
            reasons = item.get("reasons", []) or []
            suffix = ""
            if reasons:
                suffix = f" ({', '.join(str(reason) for reason in reasons)})"
            print(f"- {preset}{suffix}")

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
    if isinstance(readiness, dict) and str(readiness.get("status")) == "error":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
