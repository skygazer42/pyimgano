from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyimgano.presets.catalog import list_model_presets
from pyimgano.services.discovery_service import list_baseline_suites_payload, list_sweeps_payload
from pyimgano.services.doctor_service_helpers import (
    build_accelerator_checks as _build_accelerator_checks_helper,
)
from pyimgano.services.doctor_service_helpers import (
    build_require_extras_check as _build_require_extras_check_helper,
)
from pyimgano.services.doctor_service_helpers import (
    split_csv_args as _split_csv_args_helper,
)
from pyimgano.utils.extras import extra_installed
from pyimgano.utils.extras import extras_install_hint
from pyimgano.utils.optional_deps import optional_import
from pyimgano.workflow_guidance import artifact_hints_for_command
from pyimgano.workflow_guidance import command_workflow_guidance
from pyimgano.workflow_guidance import default_starter_benchmark_name
from pyimgano.workflow_guidance import deploy_smoke_commands
from pyimgano.workflow_guidance import first_ten_minutes_commands
from pyimgano.workflow_guidance import model_workflow_guidance
from pyimgano.workflow_guidance import model_info_command_for_model
from pyimgano.workflow_guidance import next_step_commands_for_command
from pyimgano.workflow_guidance import next_step_commands_for_model
from pyimgano.workflow_guidance import suggested_commands_for_command
from pyimgano.workflow_guidance import suggested_commands_for_model
from pyimgano.workflow_guidance import starter_benchmark_info_command
from pyimgano.workflow_guidance import starter_benchmark_list_command
from pyimgano.workflow_guidance import starter_benchmark_run_command
from pyimgano.workflow_guidance import workflow_stage_for_command
from pyimgano.workflow_guidance import workflow_stage_for_model

_DOCTOR_PROFILE_CHOICES = {"first-run", "deploy-smoke", "benchmark", "deploy", "publish"}


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def split_csv_args(values: list[str] | None) -> list[str]:
    return _split_csv_args_helper(values)


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
    return _build_require_extras_check_helper(required_extras)


def build_accelerator_checks() -> dict[str, Any]:
    return _build_accelerator_checks_helper()


def _load_json_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in: {path}")
    return payload


def _resolve_artifact_path(config_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return path

    candidates = [
        (config_path.parent / path).resolve(),
        (config_path.parent.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_external_checkpoint_audit(infer_config_path: Path) -> dict[str, Any] | None:
    if not infer_config_path.is_file():
        return None

    try:
        payload = _load_json_dict(infer_config_path)
    except Exception:
        return None

    model = payload.get("model")
    if not isinstance(model, Mapping):
        return None

    model_name = str(model.get("name", "")).strip()
    if model_name != "vision_patchcore_inspection_checkpoint":
        return None

    checkpoint_path = model.get("checkpoint_path")
    if checkpoint_path is None and isinstance(model.get("model_kwargs"), Mapping):
        checkpoint_path = model.get("model_kwargs", {}).get("checkpoint_path")

    if checkpoint_path is None or not str(checkpoint_path).strip():
        return {
            "model": model_name,
            "artifact_format": "patchcore-saved-model",
            "artifact_format_status": "missing",
            "checkpoint_version_sensitive": True,
            "errors": ["checkpoint_path_missing_from_infer_config"],
        }

    from pyimgano.models.patchcore_inspection_backend import audit_patchcore_inspection_saved_model

    resolved_path = _resolve_artifact_path(infer_config_path, str(checkpoint_path))
    return {
        "model": model_name,
        **audit_patchcore_inspection_saved_model(str(resolved_path)),
    }


def _build_run_readiness(
    *,
    run_dir: str | Path,
    check_bundle_hashes: bool,
) -> dict[str, Any]:
    from pyimgano.reporting.run_acceptance import evaluate_run_acceptance

    acceptance = evaluate_run_acceptance(
        run_dir,
        required_quality="audited",
        check_bundle_hashes=bool(check_bundle_hashes),
    )
    quality = dict(acceptance.get("quality", {}))
    issues = [str(item) for item in acceptance.get("blocking_reasons", []) if str(item)]
    if bool(acceptance.get("ready")):
        status = "audited-ready"
    elif str(quality.get("status", "")).strip().lower() in {
        "reproducible",
        "audited",
        "deployable",
    }:
        status = "warning"
    else:
        status = "error"

    infer_config = acceptance.get("infer_config", {})
    infer_config_path = None
    if isinstance(infer_config, Mapping):
        raw_path = infer_config.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            infer_config_path = Path(raw_path)
    external_checkpoint_audit = (
        _build_external_checkpoint_audit(infer_config_path)
        if infer_config_path is not None
        else None
    )

    payload = {
        "target_kind": "run",
        "path": str(Path(run_dir)),
        "status": str(status),
        "issues": issues,
        "acceptance": acceptance,
    }
    if external_checkpoint_audit is not None:
        payload["external_checkpoint_audit"] = external_checkpoint_audit
    return payload


def _bundle_manifest_validation_payload(
    *,
    bundle_dir: Path,
    check_hashes: bool,
) -> dict[str, Any]:
    from pyimgano.reporting.deploy_bundle import (
        normalize_deploy_bundle_runtime_policy,
        validate_deploy_bundle_manifest,
    )

    manifest_path = bundle_dir / "bundle_manifest.json"
    payload = {
        "path": str(manifest_path),
        "present": bool(manifest_path.is_file()),
        "valid": None,
        "errors": [],
        "runtime_policy": normalize_deploy_bundle_runtime_policy(None),
    }
    if not manifest_path.is_file():
        payload["errors"] = ["missing_bundle_manifest"]
        return payload

    try:
        manifest = _load_json_dict(manifest_path)
    except Exception as exc:  # noqa: BLE001 - diagnostics boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        return payload

    payload["runtime_policy"] = normalize_deploy_bundle_runtime_policy(
        manifest.get("runtime_policy", None)
    )
    errors = validate_deploy_bundle_manifest(
        manifest,
        bundle_dir=bundle_dir,
        check_hashes=bool(check_hashes),
    )
    payload["valid"] = len(errors) == 0
    payload["errors"] = list(errors)
    return payload


def _bundle_infer_config_validation_payload(bundle_dir: Path) -> dict[str, Any]:
    from pyimgano.inference.validate_infer_config import validate_infer_config_file

    infer_path = bundle_dir / "infer_config.json"
    payload = {
        "path": str(infer_path),
        "present": bool(infer_path.is_file()),
        "valid": None,
        "warnings": [],
        "errors": [],
        "trust_summary": {},
    }
    if not infer_path.is_file():
        payload["errors"] = ["missing_infer_config"]
        return payload

    try:
        validation = validate_infer_config_file(infer_path, check_files=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics boundary
        payload["valid"] = False
        payload["errors"] = [str(exc)]
        return payload

    payload["valid"] = True
    payload["warnings"] = list(validation.warnings)
    payload["trust_summary"] = dict(validation.trust_summary)
    return payload


def _build_bundle_readiness(
    *,
    bundle_dir: str | Path,
    check_bundle_hashes: bool,
) -> dict[str, Any]:
    from pyimgano.weights.bundle_audit import evaluate_bundle_weights_audit

    bundle_root = Path(bundle_dir)
    manifest_payload = _bundle_manifest_validation_payload(
        bundle_dir=bundle_root,
        check_hashes=bool(check_bundle_hashes),
    )
    infer_payload = _bundle_infer_config_validation_payload(bundle_root)
    weights_audit = evaluate_bundle_weights_audit(
        bundle_root,
        check_hashes=bool(check_bundle_hashes),
    )

    issues: list[str] = []
    warnings: list[str] = []

    if manifest_payload.get("present") is not True:
        issues.extend(str(item) for item in manifest_payload.get("errors", []))
    elif manifest_payload.get("valid") is not True:
        issues.extend(str(item) for item in manifest_payload.get("errors", []))

    if infer_payload.get("present") is not True:
        issues.extend(str(item) for item in infer_payload.get("errors", []))
    elif infer_payload.get("valid") is not True:
        issues.extend(str(item) for item in infer_payload.get("errors", []))
    else:
        warnings.extend(str(item) for item in infer_payload.get("warnings", []))

    if bool(weights_audit.get("present")) and weights_audit.get("ready") is not True:
        issues.append("bundle_weights_not_ready")
        issues.extend(str(item) for item in weights_audit.get("errors", []))
    else:
        warnings.extend(str(item) for item in weights_audit.get("warnings", []))

    if issues:
        status = "error"
    elif warnings:
        status = "warning"
    else:
        status = "ok"

    external_checkpoint_audit = _build_external_checkpoint_audit(bundle_root / "infer_config.json")

    payload = {
        "target_kind": "deploy_bundle",
        "path": str(bundle_root),
        "status": str(status),
        "issues": list(dict.fromkeys(issues)),
        "warnings": list(dict.fromkeys(warnings)),
        "bundle_manifest": manifest_payload,
        "infer_config": infer_payload,
        "weights_audit": weights_audit,
    }
    if external_checkpoint_audit is not None:
        payload["external_checkpoint_audit"] = external_checkpoint_audit
    return payload


def _build_publication_readiness(*, publication_target: str | Path) -> dict[str, Any]:
    from pyimgano.reporting.publication_quality import evaluate_publication_quality

    target = Path(publication_target)
    publication = evaluate_publication_quality(target)
    issues: list[str] = []
    if bool(publication.get("missing_required")):
        issues.append("missing_required_exports")
    if bool(publication.get("invalid_declared")):
        issues.append("invalid_declared_assets")
    if publication.get("publication_ready") is not True:
        issues.append("publication_not_ready")

    status = "ok" if str(publication.get("status")) == "ready" and not issues else "error"
    return {
        "target_kind": "publication",
        "path": str(target),
        "status": str(status),
        "issues": list(dict.fromkeys(issues)),
        "publication": publication,
    }


_DEFAULT_OBJECTIVE = "balanced"
_DEFAULT_ALLOW_UPSTREAM = "native+wrapped"
_DEFAULT_TOPK = 5
_DEFAULT_SELECTION_PROFILE = "balanced"
_ALLOWED_OBJECTIVES = {"balanced", "latency", "localization"}
_ALLOWED_UPSTREAM = {"native-only", "native+wrapped"}
_SELECTION_PROFILES = {
    "balanced": {
        "description": "Balanced native-first discovery with wrapped parity still visible.",
        "objective": "balanced",
        "allow_upstream": _DEFAULT_ALLOW_UPSTREAM,
        "topk": 5,
    },
    "benchmark-parity": {
        "description": "Surface native PatchCore and wrapper parity candidates for paper-style comparison.",
        "objective": "localization",
        "allow_upstream": _DEFAULT_ALLOW_UPSTREAM,
        "topk": 5,
    },
    "cpu-screening": {
        "description": "Prefer CPU-friendly native screening baselines and avoid upstream wrappers.",
        "objective": "latency",
        "allow_upstream": "native-only",
        "topk": 3,
    },
    "deploy-readiness": {
        "description": "Favor deployment-friendly native candidates with lower integration risk.",
        "objective": "balanced",
        "allow_upstream": "native-only",
        "topk": 4,
    },
}


def _append_candidate_spec(
    out: list[dict[str, Any]],
    *,
    ref: str,
    reasons: list[str],
) -> None:
    normalized_reasons = [str(item) for item in reasons if str(item).strip()]
    for item in out:
        if str(item.get("ref")) != str(ref):
            continue
        merged = list(item.get("reasons", []))
        for reason in normalized_reasons:
            if reason not in merged:
                merged.append(reason)
        item["reasons"] = merged
        return
    out.append({"ref": str(ref), "reasons": normalized_reasons})


def _normalize_objective(objective: str | None) -> str:
    key = str(objective or _DEFAULT_OBJECTIVE).strip().lower()
    if key not in _ALLOWED_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(sorted(_ALLOWED_OBJECTIVES))}")
    return key


def _normalize_allow_upstream(allow_upstream: str | None) -> str:
    key = str(allow_upstream or _DEFAULT_ALLOW_UPSTREAM).strip().lower()
    if key not in _ALLOWED_UPSTREAM:
        raise ValueError(f"allow_upstream must be one of: {', '.join(sorted(_ALLOWED_UPSTREAM))}")
    return key


def _normalize_topk(topk: int | None) -> int:
    value = int(_DEFAULT_TOPK if topk is None else topk)
    if value < 1:
        raise ValueError("topk must be >= 1")
    return value


def _normalize_selection_profile(selection_profile: str | None) -> str:
    key = str(selection_profile or _DEFAULT_SELECTION_PROFILE).strip().lower()
    if key not in _SELECTION_PROFILES:
        raise ValueError(
            f"selection_profile must be one of: {', '.join(sorted(_SELECTION_PROFILES))}"
        )
    return key


def _resolve_selection_profile(
    *,
    selection_profile: str | None,
    objective: str | None,
    allow_upstream: str | None,
    topk: int | None,
) -> dict[str, Any]:
    profile_key = _normalize_selection_profile(selection_profile)
    profile_spec = _SELECTION_PROFILES[profile_key]
    objective_key = _normalize_objective(
        objective if objective is not None else str(profile_spec["objective"])
    )
    upstream_key = _normalize_allow_upstream(
        allow_upstream if allow_upstream is not None else str(profile_spec["allow_upstream"])
    )
    topk_value = _normalize_topk(topk if topk is not None else int(profile_spec["topk"]))
    return {
        "profile": profile_key,
        "objective": objective_key,
        "allow_upstream": upstream_key,
        "topk": int(topk_value),
        "summary": {
            "requested": profile_key,
            "applied": profile_key,
            "description": str(profile_spec["description"]),
            "defaults": {
                "objective": str(profile_spec["objective"]),
                "allow_upstream": str(profile_spec["allow_upstream"]),
                "topk": int(profile_spec["topk"]),
            },
            "explicit_overrides": {
                "objective": objective is not None,
                "allow_upstream": allow_upstream is not None,
                "topk": topk is not None,
            },
        },
    }


def _build_dataset_candidate_specs(dataset_profile: dict[str, Any]) -> list[dict[str, Any]]:
    fewshot_risk = bool(dataset_profile.get("fewshot_risk"))
    pixel_ready = bool(dataset_profile.get("pixel_metrics_available"))
    multi_category = bool(dataset_profile.get("multi_category"))

    out: list[dict[str, Any]] = []

    _append_candidate_spec(
        out,
        ref="industrial-structural-ecod",
        reasons=["image_level_screening", "cpu_friendly_baseline"],
    )
    _append_candidate_spec(
        out,
        ref="industrial-embedding-core-balanced",
        reasons=["default_balanced_recommendation", "shared_embedding_space"],
    )

    if pixel_ready:
        _append_candidate_spec(
            out,
            ref="industrial-template-ncc-map",
            reasons=["pixel_metrics_available", "reference_inspection_baseline"],
        )
        _append_candidate_spec(
            out,
            ref="industrial-patchcore-lite-map",
            reasons=["pixel_metrics_available", "high_recall_pixel_map"],
        )
        _append_candidate_spec(
            out,
            ref="industrial-ssim-template-map",
            reasons=["pixel_metrics_available", "lightweight_similarity_map"],
        )
        _append_candidate_spec(
            out,
            ref="vision_patchcore",
            reasons=["pixel_metrics_available", "native_patchcore_reference"],
        )
        _append_candidate_spec(
            out,
            ref="vision_patchcore_anomalib",
            reasons=["pixel_metrics_available", "upstream_parity_reference"],
        )
        _append_candidate_spec(
            out,
            ref="vision_patchcore_inspection_checkpoint",
            reasons=["pixel_metrics_available", "upstream_saved_model_reference"],
        )

    if fewshot_risk:
        _append_candidate_spec(
            out,
            ref="industrial-pixel-mad-map",
            reasons=["fewshot_risk", "robust_reference_baseline"],
        )
        _append_candidate_spec(
            out,
            ref="industrial-embedding-core-balanced",
            reasons=["fewshot_risk", "balanced_generalist_baseline"],
        )

    if multi_category:
        _append_candidate_spec(
            out,
            ref="industrial-embedding-core-balanced",
            reasons=["multi_category_dataset", "shared_embedding_space"],
        )
        _append_candidate_spec(
            out,
            ref="industrial-reverse-distillation",
            reasons=["multi_category_dataset", "deep_generalization_baseline"],
        )

    return out


def _direct_model_required_extras(
    *,
    model_name: str,
    deployment_profile: Mapping[str, Any],
) -> list[str]:
    upstream_project = str(deployment_profile.get("upstream_project", "native"))
    tested_runtime = str(deployment_profile.get("tested_runtime", "numpy"))
    out: list[str] = []
    if upstream_project == "anomalib":
        out.append("anomalib")
    elif upstream_project == "patchcore_inspection":
        out.extend(["torch", "faiss"])
    elif tested_runtime == "torch":
        out.append("torch")
    elif tested_runtime == "onnxruntime":
        out.append("onnx")
    elif tested_runtime == "openvino":
        out.append("openvino")

    if model_name.endswith("_anomalib") and "anomalib" not in out:
        out.append("anomalib")
    return list(dict.fromkeys(out))


_BENCHMARK_REFERENCE_ROLE_BY_REASON = {
    "reference_inspection_baseline": "reference_inspection_baseline",
    "cpu_friendly_baseline": "cpu_friendly_baseline",
    "default_balanced_recommendation": "balanced_generalist_recommendation",
    "shared_embedding_space": "shared_embedding_space",
    "high_recall_pixel_map": "high_recall_pixel_map",
    "lightweight_similarity_map": "lightweight_similarity_map",
    "native_patchcore_reference": "native_patchcore_reference",
    "upstream_parity_reference": "upstream_parity_reference",
    "upstream_saved_model_reference": "upstream_saved_model_reference",
    "robust_reference_baseline": "robust_reference_baseline",
    "balanced_generalist_baseline": "balanced_generalist_baseline",
    "deep_generalization_baseline": "deep_generalization_baseline",
}

_BENCHMARK_REFERENCE_ROLE_BY_REF = {
    "industrial-template-ncc-map": "reference_inspection_baseline",
    "industrial-structural-ecod": "cpu_friendly_baseline",
    "industrial-embedding-core-balanced": "balanced_generalist_baseline",
    "industrial-patchcore-lite-map": "high_recall_pixel_map",
    "industrial-ssim-template-map": "lightweight_similarity_map",
    "industrial-pixel-mad-map": "robust_reference_baseline",
    "industrial-reverse-distillation": "deep_generalization_baseline",
    "vision_patchcore": "native_patchcore_reference",
    "vision_patchcore_anomalib": "upstream_parity_reference",
    "vision_patchcore_inspection_checkpoint": "upstream_saved_model_reference",
}


def _resolve_benchmark_reference_role(
    *,
    ref: str,
    reasons: list[str],
    deployment_profile: Mapping[str, Any],
) -> str | None:
    role = deployment_profile.get("benchmark_reference_role")
    if isinstance(role, str) and role.strip():
        return role.strip()

    for reason in reasons:
        mapped = _BENCHMARK_REFERENCE_ROLE_BY_REASON.get(str(reason))
        if mapped is not None:
            return mapped

    return _BENCHMARK_REFERENCE_ROLE_BY_REF.get(str(ref))


def _build_recommendation_candidate(ref: str, reasons: list[str]) -> dict[str, Any] | None:
    from pyimgano.models.registry import model_info
    from pyimgano.presets.catalog import resolve_model_preset

    preset = resolve_model_preset(ref)
    if preset is not None:
        info = model_info(str(preset.model))
        requires_extras = list(preset.requires_extras)
        candidate = {
            "ref": str(ref),
            "preset": str(preset.name),
            "model": str(preset.model),
            "optional": bool(preset.optional),
            "requires_extras": requires_extras,
            "description": str(preset.description),
            "reasons": [str(item) for item in reasons if str(item).strip()],
            "deployment_profile": dict(info.get("deployment_profile", {})),
        }
    else:
        info = model_info(str(ref))
        deployment_profile = dict(info.get("deployment_profile", {}))
        requires_extras = _direct_model_required_extras(
            model_name=str(ref),
            deployment_profile=deployment_profile,
        )
        candidate = {
            "ref": str(ref),
            "preset": None,
            "model": str(ref),
            "optional": bool(requires_extras),
            "requires_extras": requires_extras,
            "description": f"Direct model recommendation: {ref}",
            "reasons": [str(item) for item in reasons if str(item).strip()],
            "deployment_profile": deployment_profile,
        }

    candidate["missing_extras"] = [
        extra for extra in candidate["requires_extras"] if not extra_installed(str(extra))
    ]
    deployment_profile = candidate.get("deployment_profile", {})
    if isinstance(deployment_profile, Mapping):
        candidate["benchmark_reference_role"] = _resolve_benchmark_reference_role(
            ref=str(ref),
            reasons=[str(item) for item in reasons if str(item).strip()],
            deployment_profile=deployment_profile,
        )
    return candidate


def _parity_candidate_refs(selection_profile: str) -> list[str]:
    if selection_profile == "cpu-screening":
        return [
            "industrial-template-ncc-map",
            "industrial-structural-ecod",
            "industrial-embedding-core-balanced",
        ]
    return [
        "industrial-template-ncc-map",
        "industrial-structural-ecod",
        "vision_patchcore",
        "vision_patchcore_anomalib",
        "vision_patchcore_inspection_checkpoint",
    ]


def _build_parity_candidates(selection_profile: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref in _parity_candidate_refs(selection_profile):
        candidate = _build_recommendation_candidate(ref, reasons=["benchmark_parity_candidate"])
        if candidate is None:
            continue
        candidate["eligible"] = len(candidate.get("missing_extras", [])) == 0
        out.append(candidate)
    return out


def _selection_score(
    candidate: Mapping[str, Any],
    *,
    objective: str,
    dataset_profile: Mapping[str, Any],
) -> int:
    profile, industrial_fit = _selection_score_inputs(candidate)
    runtime_score = {"low": 30, "medium": 15, "high": 0}
    memory_score = {"low": 15, "medium": 8, "high": 0}
    runtime_hint = str(profile.get("runtime_cost_hint", "high"))
    memory_hint = str(profile.get("memory_cost_hint", "high"))
    has_checkpoint = bool(profile.get("artifact_requirements"))
    upstream_project = str(profile.get("upstream_project", "native"))

    if objective == "latency":
        return _latency_selection_score(
            runtime_score=runtime_score,
            memory_score=memory_score,
            runtime_hint=runtime_hint,
            memory_hint=memory_hint,
            has_checkpoint=has_checkpoint,
            upstream_project=upstream_project,
            industrial_fit=industrial_fit,
            dataset_profile=dataset_profile,
        )

    if objective == "localization":
        return _localization_selection_score(
            industrial_fit=industrial_fit,
            dataset_profile=dataset_profile,
            upstream_project=upstream_project,
        )

    return _balanced_selection_score(
        candidate=candidate,
        runtime_score=runtime_score,
        memory_score=memory_score,
        runtime_hint=runtime_hint,
        memory_hint=memory_hint,
        has_checkpoint=has_checkpoint,
        upstream_project=upstream_project,
        industrial_fit=industrial_fit,
        dataset_profile=dataset_profile,
    )


def _selection_score_inputs(
    candidate: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    profile = candidate.get("deployment_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    industrial_fit = profile.get("industrial_fit", {})
    if not isinstance(industrial_fit, Mapping):
        industrial_fit = {}
    return profile, industrial_fit


def _latency_selection_score(
    *,
    runtime_score: Mapping[str, int],
    memory_score: Mapping[str, int],
    runtime_hint: str,
    memory_hint: str,
    has_checkpoint: bool,
    upstream_project: str,
    industrial_fit: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
) -> int:
    score = runtime_score.get(runtime_hint, 0) + memory_score.get(memory_hint, 0)
    if not has_checkpoint:
        score += 12
    if bool(industrial_fit.get("reference_inspection")):
        score += 10
    if bool(industrial_fit.get("pixel_localization")) and bool(
        dataset_profile.get("pixel_metrics_available")
    ):
        score += 4
    if upstream_project == "native":
        score += 4
    return int(score)


def _localization_selection_score(
    *,
    industrial_fit: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
    upstream_project: str,
) -> int:
    score = 0
    if bool(dataset_profile.get("pixel_metrics_available")):
        score += 25 if bool(industrial_fit.get("pixel_localization")) else 0
    if bool(industrial_fit.get("reference_inspection")):
        score += 10
    if upstream_project == "native":
        score += 3
    return int(score)


def _balanced_selection_score(
    *,
    candidate: Mapping[str, Any],
    runtime_score: Mapping[str, int],
    memory_score: Mapping[str, int],
    runtime_hint: str,
    memory_hint: str,
    has_checkpoint: bool,
    upstream_project: str,
    industrial_fit: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
) -> int:
    score = 0
    if bool(dataset_profile.get("pixel_metrics_available")) and bool(
        industrial_fit.get("pixel_localization")
    ):
        score += 20
    if bool(industrial_fit.get("reference_inspection")):
        score += 12
    if "fewshot_risk" in set(candidate.get("reasons", [])):
        score += 8
    if "multi_category_dataset" in set(candidate.get("reasons", [])):
        score += 8
    if not has_checkpoint:
        score += 6
    score += runtime_score.get(runtime_hint, 0) // 3
    score += memory_score.get(memory_hint, 0) // 3
    if upstream_project == "native":
        score += 3
    return int(score)


def _build_recommendation_explanations(
    recommendations: list[dict[str, Any]],
    *,
    objective: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rank, candidate in enumerate(recommendations, start=1):
        profile = candidate.get("deployment_profile", {})
        if not isinstance(profile, Mapping):
            profile = {}
        out.append(
            {
                "rank": int(rank),
                "target": candidate.get("preset") or candidate.get("model"),
                "objective": str(objective),
                "upstream_project": profile.get("upstream_project"),
                "runtime_cost_hint": profile.get("runtime_cost_hint"),
                "summary": ", ".join(str(item) for item in candidate.get("reasons", [])[:3]),
            }
        )
    return out


def _build_dataset_recommendations(
    dataset_profile: dict[str, Any],
    *,
    objective: str,
    allow_upstream: str,
    topk: int,
) -> dict[str, Any]:
    objective_key = _normalize_objective(objective)
    upstream_key = _normalize_allow_upstream(allow_upstream)
    topk_value = _normalize_topk(topk)

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for spec in _build_dataset_candidate_specs(dataset_profile):
        candidate = _build_recommendation_candidate(
            str(spec.get("ref")),
            [str(item) for item in spec.get("reasons", [])],
        )
        if candidate is None:
            continue

        rejection_reasons = _recommendation_rejection_reasons(
            candidate,
            upstream_key=upstream_key,
        )
        if rejection_reasons:
            candidate["reasons"] = _merge_recommendation_reasons(
                candidate.get("reasons", []),
                rejection_reasons,
            )
            rejected.append(candidate)
            continue

        candidate["selection_score"] = _selection_score(
            candidate,
            objective=objective_key,
            dataset_profile=dataset_profile,
        )
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            -int(item.get("selection_score", 0)),
            str(item.get("preset") or item.get("model") or item.get("ref")),
        )
    )
    recommendations = candidates[:topk_value]

    upstream_counts = {"native": 0, "anomalib": 0, "patchcore_inspection": 0}
    _count_recommendation_upstreams(upstream_counts, [*recommendations, *rejected])

    return {
        "selection_context": {
            "objective": objective_key,
            "allow_upstream": upstream_key,
            "topk": int(topk_value),
        },
        "candidate_pool_summary": {
            "total_candidates": int(len(candidates) + len(rejected)),
            "eligible_count": int(len(candidates)),
            "selected_count": int(len(recommendations)),
            "rejected_count": int(len(rejected)),
            "upstream_counts": upstream_counts,
        },
        "recommendations": recommendations,
        "rejected_candidates": rejected,
        "recommendation_explanations": _build_recommendation_explanations(
            recommendations,
            objective=objective_key,
        ),
    }


def _recommendation_rejection_reasons(
    candidate: Mapping[str, Any],
    *,
    upstream_key: str,
) -> list[str]:
    profile = candidate.get("deployment_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    upstream_project = str(profile.get("upstream_project", "native"))
    rejection_reasons: list[str] = []
    if upstream_key == "native-only" and upstream_project != "native":
        rejection_reasons.append("upstream_disallowed:native-only")
    for extra in candidate.get("missing_extras", []):
        rejection_reasons.append(f"missing_extra:{extra}")
    return rejection_reasons


def _merge_recommendation_reasons(
    existing: Any,
    new_reasons: Sequence[str],
) -> list[str]:
    merged = [str(item) for item in existing if str(item)]
    for reason in new_reasons:
        if reason not in merged:
            merged.append(reason)
    return merged


def _count_recommendation_upstreams(
    upstream_counts: dict[str, int],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    for candidate in candidates:
        profile = candidate.get("deployment_profile", {})
        if not isinstance(profile, Mapping):
            continue
        upstream_project = str(profile.get("upstream_project", "native"))
        upstream_counts[upstream_project] = upstream_counts.get(upstream_project, 0) + 1


def _build_dataset_target_payload(
    *,
    dataset_target: str | Path,
    dataset: str,
    category: str | None,
    root_fallback: str | Path | None,
    objective: str | None,
    allow_upstream: str | None,
    selection_profile: str | None,
    topk: int | None,
) -> dict[str, Any]:
    from pyimgano.datasets.inspection import profile_dataset_target

    profile_payload = profile_dataset_target(
        target=dataset_target,
        dataset=str(dataset),
        category=category,
        root_fallback=root_fallback,
    )
    dataset_profile = dict(profile_payload.get("dataset_profile", {}))
    evaluation_readiness = dict(profile_payload.get("evaluation_readiness", {}))
    constraints = dict(profile_payload.get("constraints", {}))

    readiness_payload = dict(profile_payload.get("readiness", {}) or {})
    issues = [str(item) for item in readiness_payload.get("issues", []) if str(item).strip()]
    status = str(readiness_payload.get("status", "ok") or "ok")
    issue_codes = [str(item) for item in readiness_payload.get("issue_codes", []) if str(item).strip()]
    issue_details = [
        dict(item)
        for item in readiness_payload.get("issue_details", [])
        if isinstance(item, Mapping)
    ]

    selection_profile_payload = _resolve_selection_profile(
        selection_profile=selection_profile,
        objective=objective,
        allow_upstream=allow_upstream,
        topk=topk,
    )

    recommendation_payload = _build_dataset_recommendations(
        dataset_profile,
        objective=str(selection_profile_payload["objective"]),
        allow_upstream=str(selection_profile_payload["allow_upstream"]),
        topk=int(selection_profile_payload["topk"]),
    )

    return {
        "dataset_profile": dataset_profile,
        "task_profile": dict(profile_payload.get("task_profile", {})),
        "constraints": constraints,
        "evaluation_readiness": evaluation_readiness,
        "selection_profile_summary": selection_profile_payload["summary"],
        "selection_context": recommendation_payload["selection_context"],
        "candidate_pool_summary": recommendation_payload["candidate_pool_summary"],
        "parity_candidates": _build_parity_candidates(str(selection_profile_payload["profile"])),
        "recommendations": recommendation_payload["recommendations"],
        "rejected_candidates": recommendation_payload["rejected_candidates"],
        "recommendation_explanations": recommendation_payload["recommendation_explanations"],
        "readiness": {
            "target_kind": "dataset",
            "path": str(Path(dataset_target)),
            "status": str(status),
            "issues": issues,
            "issue_codes": issue_codes,
            "issue_details": issue_details,
            "dataset": profile_payload.get("dataset"),
            "category": profile_payload.get("category"),
        },
    }


_COMMAND_EXTRA_SPECS: dict[str, dict[str, Any]] = {
    "export-onnx": {
        "required_extras": ["onnx", "torch"],
        "recommended_extras": [],
        "recommended_extra_profiles": ["deploy"],
        "notes": ["Exports require torch plus ONNX tooling."],
    },
    "export-torchscript": {
        "required_extras": ["torch"],
        "recommended_extras": [],
        "recommended_extra_profiles": ["deploy"],
        "notes": ["TorchScript export depends on the torch extra."],
    },
    "train": {
        "required_extras": ["torch"],
        "recommended_extras": ["faiss"],
        "recommended_extra_profiles": ["deploy", "tracking"],
        "notes": ["Most training-first deep workflows depend on torch."],
    },
    "infer": {
        "required_extras": [],
        "recommended_extras": ["torch", "onnx", "openvino"],
        "recommended_extra_profiles": ["deploy"],
        "notes": ["Base install supports classical CPU inference; extras unlock deep and deploy runtimes."],
    },
    "runs": {
        "required_extras": [],
        "recommended_extras": [],
        "notes": ["Run inspection and release gates work on the base install."],
    },
    "demo": {
        "required_extras": [],
        "recommended_extras": [],
        "recommended_extra_profiles": ["cpu-offline"],
        "notes": ["The smoke demo is designed to run on the base install."],
    },
}


def _sorted_unique(items: Sequence[str]) -> list[str]:
    return sorted({str(item) for item in items if str(item).strip()})


def _build_extra_recommendation_payload(
    *,
    target_kind: str,
    target: str,
    required_extras: Sequence[str],
    recommended_extras: Sequence[str],
    recommended_extra_profiles: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    required = _sorted_unique(required_extras)
    recommended = [item for item in _sorted_unique(recommended_extras) if item not in required]
    combined = [*required, *recommended]
    missing = [extra for extra in combined if not extra_installed(extra)]
    available = [extra for extra in combined if extra not in missing]
    profile_names = [
        str(item)
        for item in recommended_extra_profiles or []
        if str(item).strip()
    ]
    install_hint = extras_install_hint(missing or combined) if combined else None
    install_command = extras_install_hint([profile_names[0]]) if profile_names else install_hint
    return {
        "target_kind": str(target_kind),
        "target": str(target),
        "required_extras": required,
        "recommended_extras": recommended,
        "recommended_extra_profiles": profile_names,
        "missing_extras": missing,
        "available_extras": available,
        "install_hint": install_hint,
        "install_command": install_command,
        "notes": [str(item) for item in (notes or []) if str(item).strip()],
    }


def _build_command_extra_recommendation(command_name: str) -> dict[str, Any]:
    key = str(command_name).strip()
    if not key:
        raise ValueError("for_command must not be empty.")
    if key == "benchmark":
        from pyimgano.reporting.benchmark_config import list_starter_benchmark_configs

        starter_configs = list_starter_benchmark_configs()
        recommended_extras: list[str] = []
        starter_names: list[str] = []
        optional_baseline_count = 0
        for item in starter_configs:
            starter_names.append(str(item.get("name")))
            optional_baseline_count = max(
                optional_baseline_count,
                int(item.get("optional_baseline_count", 0) or 0),
            )
            for extra in item.get("optional_extras", []) or []:
                text = str(extra).strip()
                if text and text not in recommended_extras:
                    recommended_extras.append(text)

        payload = _build_extra_recommendation_payload(
            target_kind="command",
            target=key,
            required_extras=[],
            recommended_extras=recommended_extras,
            recommended_extra_profiles=["benchmark", "cpu-offline"],
            notes=[
                "Starter CPU baselines run without extras; broader suites often benefit from clip, torch, and skimage."
            ],
        )
        payload["optional_baseline_count"] = int(optional_baseline_count)
        payload["starter_configs"] = sorted(starter_names)
        payload["starter_list_command"] = starter_benchmark_list_command()
        payload["workflow_stage"] = workflow_stage_for_command(key)
        if starter_names:
            preferred = (
                default_starter_benchmark_name()
                if default_starter_benchmark_name() in starter_names
                else sorted(starter_names)[0]
            )
            payload["starter_info_command"] = starter_benchmark_info_command(preferred)
            payload["starter_run_command"] = starter_benchmark_run_command(preferred)
            payload["suggested_commands"] = suggested_commands_for_command(key)
            payload["next_step_commands"] = next_step_commands_for_command(key)
        else:
            payload["suggested_commands"] = suggested_commands_for_command(key)
            payload["next_step_commands"] = next_step_commands_for_command(key)
        payload["artifact_hints"] = artifact_hints_for_command(key)
        return payload

    spec = _COMMAND_EXTRA_SPECS.get(key)
    if spec is None:
        raise ValueError(f"Unknown command for extras recommendation: {key!r}")
    guidance = command_workflow_guidance(key)
    return _build_extra_recommendation_payload(
        target_kind="command",
        target=key,
        required_extras=spec.get("required_extras", []),
        recommended_extras=spec.get("recommended_extras", []),
        recommended_extra_profiles=spec.get("recommended_extra_profiles", []),
        notes=spec.get("notes", []),
    ) | {
        "workflow_stage": (None if guidance is None else guidance.workflow_stage),
        "recipe_list_command": (None if guidance is None else guidance.recipe_list_command),
        "recipe_info_command": (None if guidance is None else guidance.recipe_info_command),
        "dry_run_command": (None if guidance is None else guidance.dry_run_command),
        "recipe_run_command": (None if guidance is None else guidance.recipe_run_command),
        "preset_infer_command": (None if guidance is None else guidance.preset_infer_command),
        "from_run_infer_command": (None if guidance is None else guidance.from_run_infer_command),
        "quality_command": (None if guidance is None else guidance.quality_command),
        "acceptance_command": (None if guidance is None else guidance.acceptance_command),
        "bundle_audit_command": (None if guidance is None else guidance.bundle_audit_command),
        "suggested_commands": ([] if guidance is None else list(guidance.suggested_commands)),
        "next_step_commands": ([] if guidance is None else list(guidance.next_step_commands)),
        "artifact_hints": ([] if guidance is None else list(guidance.artifact_hints)),
    }


def _build_model_extra_recommendation(model_name: str) -> dict[str, Any]:
    from pyimgano.services.discovery_service import build_model_info_payload

    info = build_model_info_payload(str(model_name))
    metadata = dict(info.get("metadata", {}))
    tags = {str(item).strip().lower() for item in info.get("tags", [])}
    profile = dict(info.get("deployment_profile", {}))
    required: list[str] = []
    recommended: list[str] = []
    notes: list[str] = []

    tested_runtime = str(profile.get("tested_runtime", "")).strip().lower()
    upstream_project = str(profile.get("upstream_project", "")).strip().lower()
    backend = str(metadata.get("backend", "")).strip().lower()
    weights_source = str(metadata.get("weights_source", "")).strip().lower()
    memory_bank = dict(profile.get("memory_bank", {}))

    if (
        tested_runtime == "torch"
        or "torch" in tags
        or "deep" in tags
        or "clip" in tags
        or upstream_project in {"anomalib", "patchcore_inspection"}
        or backend in {"anomalib", "patchcore_inspection", "openclip"}
    ):
        required.append("torch")
    if tested_runtime == "onnxruntime" or "onnx" in tags or weights_source == "local-exported-onnx":
        required.append("onnx")
    if "clip" in tags or "openclip" in tags or backend == "openclip":
        required.append("clip")
    if upstream_project == "anomalib" or "anomalib" in tags or backend == "anomalib":
        required.append("anomalib")
    if (
        upstream_project == "patchcore_inspection"
        or "patchcore_inspection" in tags
        or backend == "patchcore_inspection"
    ):
        required.append("patchcore_inspection")
        notes.append(
            "PatchCore-Inspection is not published on PyPI; install the upstream package separately when needed."
        )
    if bool(memory_bank.get("enabled")):
        recommended.append("faiss")
    if "ssim" in tags:
        recommended.append("skimage")

    payload = _build_extra_recommendation_payload(
        target_kind="model",
        target=str(model_name),
        required_extras=required,
        recommended_extras=recommended,
        notes=notes,
    )
    guidance = model_workflow_guidance(model_name)
    payload["workflow_stage"] = guidance.workflow_stage
    payload["supports_pixel_map"] = bool(info.get("supports_pixel_map", False))
    payload["tested_runtime"] = str(profile.get("tested_runtime", ""))
    payload["model_info_command"] = guidance.model_info_command
    payload["suggested_commands"] = list(guidance.suggested_commands)
    payload["next_step_commands"] = list(guidance.next_step_commands)
    return payload


def _resolve_extra_recommendation(
    *,
    recommend_extras: bool,
    for_command: str | None,
    for_model: str | None,
) -> dict[str, Any] | None:
    if not recommend_extras and for_command is None and for_model is None:
        return None

    if for_model is not None:
        return _build_model_extra_recommendation(str(for_model))
    if for_command is not None:
        return _build_command_extra_recommendation(str(for_command))
    return _build_extra_recommendation_payload(
        target_kind="command",
        target="base-install",
        required_extras=[],
        recommended_extras=[],
        notes=["Base install is sufficient for the lightweight CPU smoke path."],
    )


def collect_doctor_payload(
    *,
    suites_to_check: list[str] | None = None,
    require_extras: list[str] | None = None,
    accelerators: bool = False,
    profile: str | None = None,
    run_dir: str | None = None,
    deploy_bundle: str | None = None,
    publication_target: str | None = None,
    dataset_target: str | None = None,
    dataset: str = "auto",
    category: str | None = None,
    root_fallback: str | None = None,
    objective: str | None = None,
    allow_upstream: str | None = None,
    selection_profile: str | None = None,
    topk: int | None = None,
    recommend_extras: bool = False,
    for_command: str | None = None,
    for_model: str | None = None,
    check_bundle_hashes: bool = False,
) -> dict[str, Any]:
    import pyimgano

    optional_modules = _doctor_optional_modules()
    payload = _doctor_base_payload(
        pyimgano_version=str(getattr(pyimgano, "__version__", "")),
        optional_modules=optional_modules,
    )
    _apply_doctor_runtime_checks(
        payload,
        suites_to_check=suites_to_check,
        require_extras=require_extras,
        accelerators=accelerators,
    )
    extras_recommendation = _resolve_extra_recommendation(
        recommend_extras=bool(recommend_extras),
        for_command=for_command,
        for_model=for_model,
    )
    if extras_recommendation is not None:
        payload["extras_recommendation"] = extras_recommendation
    _apply_doctor_readiness_targets(
        payload,
        run_dir=run_dir,
        deploy_bundle=deploy_bundle,
        publication_target=publication_target,
        dataset_target=dataset_target,
        dataset=dataset,
        category=category,
        root_fallback=root_fallback,
        objective=objective,
        allow_upstream=allow_upstream,
        selection_profile=selection_profile,
        topk=topk,
        check_bundle_hashes=check_bundle_hashes,
    )
    if dataset_target is not None:
        payload["dataset_target"] = str(dataset_target)
    if publication_target is not None:
        payload["publication_target"] = str(publication_target)
    profile_payload = _resolve_doctor_profile_payload(
        profile=(str(profile) if profile is not None else None),
        payload=payload,
    )
    if profile_payload is not None:
        payload.update(profile_payload)
    return payload


def _doctor_optional_modules() -> list[dict[str, Any]]:
    return [
        check_module(module="numpy", dist="numpy", purpose="core numerical backend"),
        check_module(module="cv2", dist="opencv-python", purpose="image IO / preprocessing"),
        check_module(module="sklearn", dist="scikit-learn", purpose="classical ML baselines"),
        check_module(
            module="torch", dist="torch", extra="torch", purpose="deep models / embeddings"
        ),
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


def _doctor_base_payload(
    *,
    pyimgano_version: str,
    optional_modules: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "tool": "pyimgano-doctor",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pyimgano_version": pyimgano_version,
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


def _core_module_availability(optional_modules: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    tracked = {"cv2": False, "numpy": False, "sklearn": False}
    for item in optional_modules:
        module_name = str(item.get("module", "")).strip()
        if module_name not in tracked:
            continue
        tracked[module_name] = bool(item.get("available"))
    return tracked


def _profile_readiness_payload(
    *,
    profile: str,
    status: str,
    issues: Sequence[str],
) -> dict[str, Any]:
    return {
        "target_kind": "profile",
        "path": str(profile),
        "status": str(status),
        "issues": [str(item) for item in issues if str(item).strip()],
    }


def _build_first_run_profile_payload(
    *,
    optional_modules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module_state = _core_module_availability(optional_modules)
    required_modules = sorted(module_state)
    missing_modules = [name for name in required_modules if not bool(module_state.get(name))]
    issues = [f"missing_core_module:{name}" for name in missing_modules]
    status = "ok" if not issues else "error"
    workflow_profile = {
        "target_kind": "profile",
        "profile": "first-run",
        "status": str(status),
        "summary": "doctor -> demo -> benchmark -> infer -> runs quality",
        "offline_safe": True,
        "required_modules": required_modules,
        "missing_modules": missing_modules,
        "required_extras": [],
        "recommended_extras": [],
        "missing_extras": [],
        "starter_commands": first_ten_minutes_commands(),
        "artifact_hints": [
            "./_demo_suite_run/report.json",
            "./_demo_benchmark_run/report.json",
            "./_demo_benchmark_run/leaderboard.csv",
            "./_demo_results.jsonl",
        ],
        "issues": issues,
    }
    return {
        "workflow_profile": workflow_profile,
        "readiness": _profile_readiness_payload(
            profile="first-run",
            status=status,
            issues=issues,
        ),
    }


def _build_deploy_smoke_profile_payload(
    *,
    optional_modules: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    module_state = _core_module_availability(optional_modules)
    required_modules = sorted(module_state)
    missing_modules = [name for name in required_modules if not bool(module_state.get(name))]
    issues = [f"missing_core_module:{name}" for name in missing_modules]
    status = "ok" if not issues else "error"
    workflow_profile = {
        "target_kind": "profile",
        "profile": "deploy-smoke",
        "status": str(status),
        "summary": "doctor -> demo -> train/export bundle -> validate infer-config -> bundle validate -> runs quality",
        "offline_safe": True,
        "required_modules": required_modules,
        "missing_modules": missing_modules,
        "required_extras": [],
        "recommended_extras": [],
        "missing_extras": [],
        "starter_commands": deploy_smoke_commands(),
        "artifact_hints": [
            "report.json",
            "config.json",
            "environment.json",
            "artifacts/infer_config.json",
            "deploy_bundle/infer_config.json",
            "deploy_bundle/bundle_manifest.json",
        ],
        "issues": issues,
    }
    return {
        "workflow_profile": workflow_profile,
        "readiness": _profile_readiness_payload(
            profile="deploy-smoke",
            status=status,
            issues=issues,
        ),
    }


def _build_benchmark_profile_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    recommendation = _build_command_extra_recommendation("benchmark")
    dataset_target = payload.get("dataset_target")
    dataset_readiness = payload.get("readiness")
    if dataset_target is None:
        status = "warning"
        issues = ["dataset_target_missing_for_benchmark_profile"]
    elif isinstance(dataset_readiness, Mapping):
        status = str(dataset_readiness.get("status", "warning") or "warning")
        issues = [str(item) for item in dataset_readiness.get("issues", []) if str(item).strip()]
    else:
        status = "warning"
        issues = ["dataset_target_readiness_unavailable"]

    workflow_profile = {
        "target_kind": "profile",
        "profile": "benchmark",
        "status": str(status),
        "summary": "dataset readiness + starter benchmark config + expected benchmark artifacts",
        "starter_config": default_starter_benchmark_name(),
        "required_extras": list(recommendation.get("required_extras", [])),
        "recommended_extras": list(recommendation.get("recommended_extras", [])),
        "missing_extras": list(recommendation.get("missing_extras", [])),
        "starter_commands": list(recommendation.get("suggested_commands", [])),
        "next_step_commands": [
            *next_step_commands_for_command("benchmark"),
            "pyimgano-infer --model-preset industrial-template-ncc-map --train-dir /path/to/train/normal --input /path/to/images --save-jsonl /tmp/pyimgano_results.jsonl",
            "pyimgano runs quality runs/<run_dir> --require-status audited --json",
        ],
        "artifact_hints": list(recommendation.get("artifact_hints", [])),
        "dataset_target": None if dataset_target is None else str(dataset_target),
        "issues": issues,
    }
    return {
        "workflow_profile": workflow_profile,
        "readiness": _profile_readiness_payload(
            profile="benchmark",
            status=status,
            issues=issues,
        ),
    }


def _build_deploy_profile_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    readiness = payload.get("readiness")
    if isinstance(readiness, Mapping) and str(readiness.get("target_kind")) in {
        "run",
        "deploy_bundle",
    }:
        status = str(readiness.get("status", "warning") or "warning")
        issues = [str(item) for item in readiness.get("issues", []) if str(item).strip()]
        target_path = str(readiness.get("path", "") or "")
        target_source = "run_dir" if str(readiness.get("target_kind")) == "run" else "deploy_bundle"
    else:
        status = "warning"
        issues = ["deploy_target_missing_for_deploy_profile"]
        target_path = ""
        target_source = "missing"

    workflow_profile = {
        "target_kind": "profile",
        "profile": "deploy",
        "status": str(status),
        "summary": "run/deploy bundle readiness plus acceptance-adjacent deployment checks",
        "target_path": target_path,
        "target_source": target_source,
        "starter_commands": [
            "pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json",
            "pyimgano-doctor --profile deploy --deploy-bundle runs/<run_dir>/deploy_bundle --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        ],
        "next_step_commands": [
            "pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json",
            "pyimgano bundle validate runs/<run_dir>/deploy_bundle --json",
            "pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json",
        ],
        "artifact_hints": [
            "report.json",
            "config.json",
            "environment.json",
            "deploy_bundle/infer_config.json",
            "deploy_bundle/bundle_manifest.json",
        ],
        "issues": issues,
    }
    return {
        "workflow_profile": workflow_profile,
        "readiness": _profile_readiness_payload(
            profile="deploy",
            status=status,
            issues=issues,
        ),
    }


def _build_publish_profile_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    publication_target = payload.get("publication_target")
    publication_payload = payload.get("publication")
    if publication_target is None or not isinstance(publication_payload, Mapping):
        status = "warning"
        issues = ["publication_target_missing_for_publish_profile"]
        target_path = ""
        publication = None
    else:
        publication = dict(publication_payload)
        issues: list[str] = []
        if bool(publication.get("missing_required")):
            issues.append("missing_required_exports")
        if bool(publication.get("invalid_declared")):
            issues.append("invalid_declared_assets")
        if publication.get("publication_ready") is not True:
            issues.append("publication_not_ready")
        status = "ok" if str(publication.get("status")) == "ready" and not issues else "error"
        target_path = str(publication_target)

    workflow_profile = {
        "target_kind": "profile",
        "profile": "publish",
        "status": str(status),
        "summary": "suite export publication readiness plus trust-signal gate checks",
        "target_path": target_path,
        "target_source": "publication_target" if target_path else "missing",
        "starter_commands": [
            "pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json",
            "pyimgano runs acceptance /path/to/suite_export --json",
            "pyimgano runs publication /path/to/suite_export --json",
        ],
        "next_step_commands": [
            "pyimgano benchmark --list-starter-configs",
            "pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json",
            "pyimgano runs publication /path/to/suite_export --json",
        ],
        "artifact_hints": [
            "leaderboard.csv",
            "leaderboard_metadata.json",
            "report.json",
            "config.json",
            "environment.json",
        ],
        "issues": issues,
    }
    out = {
        "workflow_profile": workflow_profile,
        "readiness": _profile_readiness_payload(
            profile="publish",
            status=status,
            issues=issues,
        ),
    }
    if publication is not None:
        out["publication"] = publication
    return out


def _resolve_doctor_profile_payload(
    *,
    profile: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    if profile is None:
        return None

    key = str(profile).strip().lower()
    if key not in _DOCTOR_PROFILE_CHOICES:
        raise ValueError(
            f"profile must be one of: {', '.join(sorted(_DOCTOR_PROFILE_CHOICES))}"
        )

    optional_modules = payload.get("optional_modules", [])
    if not isinstance(optional_modules, Sequence):
        optional_modules = []

    if key == "first-run":
        return _build_first_run_profile_payload(optional_modules=optional_modules)
    if key == "deploy-smoke":
        return _build_deploy_smoke_profile_payload(optional_modules=optional_modules)
    if key == "benchmark":
        return _build_benchmark_profile_payload(payload)
    if key == "deploy":
        return _build_deploy_profile_payload(payload)
    if key == "publish":
        return _build_publish_profile_payload(payload)
    raise ValueError(f"Unsupported doctor profile: {key}")


def _apply_doctor_runtime_checks(
    payload: dict[str, Any],
    *,
    suites_to_check: list[str] | None,
    require_extras: list[str] | None,
    accelerators: bool,
) -> None:
    suites = split_csv_args(suites_to_check)
    if suites:
        payload["suite_checks"] = build_suite_checks(suites)

    if require_extras:
        payload["require_extras"] = build_require_extras_check(require_extras)

    if bool(accelerators):
        payload["accelerators"] = build_accelerator_checks()


def _apply_doctor_readiness_targets(
    payload: dict[str, Any],
    *,
    run_dir: str | None,
    deploy_bundle: str | None,
    publication_target: str | None,
    dataset_target: str | None,
    dataset: str,
    category: str | None,
    root_fallback: str | None,
    objective: str | None,
    allow_upstream: str | None,
    selection_profile: str | None,
    topk: int | None,
    check_bundle_hashes: bool,
) -> None:
    if run_dir is not None:
        payload["readiness"] = _doctor_readiness_payload(
            _build_run_readiness(
                run_dir=str(run_dir),
                check_bundle_hashes=bool(check_bundle_hashes),
            )
        )
    if deploy_bundle is not None:
        payload["readiness"] = _doctor_readiness_payload(
            _build_bundle_readiness(
                bundle_dir=str(deploy_bundle),
                check_bundle_hashes=bool(check_bundle_hashes),
            )
        )
    if publication_target is not None:
        publication_readiness = _build_publication_readiness(
            publication_target=str(publication_target)
        )
        payload["publication"] = dict(publication_readiness.get("publication", {}))
        payload["readiness"] = _doctor_readiness_payload(publication_readiness)
    if dataset_target is not None:
        payload.update(
            _build_dataset_target_payload(
                dataset_target=str(dataset_target),
                dataset=str(dataset),
                category=(str(category) if category is not None else None),
                root_fallback=(str(root_fallback) if root_fallback is not None else None),
                objective=(str(objective) if objective is not None else None),
                allow_upstream=(str(allow_upstream) if allow_upstream is not None else None),
                selection_profile=(
                    str(selection_profile) if selection_profile is not None else None
                ),
                topk=(int(topk) if topk is not None else None),
            )
        )


def _doctor_readiness_payload(readiness: Mapping[str, Any]) -> dict[str, Any]:
    readiness_payload = dict(readiness)
    external_checkpoint_audit = readiness_payload.get("external_checkpoint_audit")
    if isinstance(external_checkpoint_audit, Mapping):
        readiness_payload["external_artifact_audit"] = {
            "provider": "patchcore_inspection_saved_model",
            "artifact_kind": "saved_model_directory",
            "model": external_checkpoint_audit.get("model"),
            "audit": dict(external_checkpoint_audit),
        }
    return readiness_payload


__all__ = [
    "build_accelerator_checks",
    "build_require_extras_check",
    "build_suite_checks",
    "check_module",
    "collect_doctor_payload",
    "split_csv_args",
]
