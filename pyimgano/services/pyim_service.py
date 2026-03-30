from __future__ import annotations

from typing import Any

import pyimgano.services.pyim_payload_collectors as pyim_payload_collectors
from pyimgano.pyim_contracts import PyimListPayload, PyimListRequest
from pyimgano.utils.extras import extra_installed
from pyimgano.utils.extras import extras_install_hint


_DEFAULT_OBJECTIVE = "balanced"
_DEFAULT_SELECTION_PROFILE = "balanced"
_DEFAULT_TOPK = 5

_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "balanced": {
        "objective": "balanced",
        "topk": 5,
        "description": "General-purpose starter picks across CPU, localization, and parity routes.",
    },
    "benchmark-parity": {
        "objective": "balanced",
        "topk": 5,
        "description": "Prefer benchmark-oriented and upstream-parity reference models.",
    },
    "cpu-screening": {
        "objective": "latency",
        "topk": 5,
        "description": "Prefer lightweight CPU-friendly screening baselines.",
    },
    "deploy-readiness": {
        "objective": "balanced",
        "topk": 5,
        "description": "Prefer lower-friction deploy and export-adjacent routes.",
    },
}

_CURATED_MODEL_PICKS: tuple[dict[str, Any], ...] = (
    {
        "name": "vision_ecod",
        "priority": 10,
        "profiles": ("balanced", "cpu-screening", "deploy-readiness"),
        "objectives": ("balanced", "latency"),
        "required_extras": (),
        "recommended_extras": (),
        "summary": "Fast CPU image-level baseline with minimal setup.",
    },
    {
        "name": "ssim_template_map",
        "priority": 20,
        "profiles": ("balanced", "cpu-screening", "deploy-readiness"),
        "objectives": ("balanced", "latency", "localization"),
        "required_extras": (),
        "recommended_extras": ("skimage",),
        "summary": "Reference-style pixel localization baseline that stays CPU friendly.",
    },
    {
        "name": "vision_patchcore",
        "priority": 30,
        "profiles": ("balanced", "benchmark-parity"),
        "objectives": ("balanced", "localization"),
        "required_extras": ("torch",),
        "recommended_extras": ("faiss",),
        "summary": "Strong memory-bank localization baseline for industrial inspection.",
    },
    {
        "name": "vision_patchcore_anomalib",
        "priority": 40,
        "profiles": ("benchmark-parity",),
        "objectives": ("balanced", "localization"),
        "required_extras": ("anomalib", "torch"),
        "recommended_extras": (),
        "summary": "anomalib-compatible PatchCore checkpoint route.",
    },
    {
        "name": "vision_patchcore_inspection_checkpoint",
        "priority": 50,
        "profiles": ("benchmark-parity",),
        "objectives": ("balanced", "localization"),
        "required_extras": ("patchcore_inspection", "torch"),
        "recommended_extras": (),
        "summary": "PatchCore-Inspection checkpoint interoperability path.",
    },
    {
        "name": "vision_onnx_ecod",
        "priority": 60,
        "profiles": ("cpu-screening", "deploy-readiness"),
        "objectives": ("balanced", "latency"),
        "required_extras": ("onnx",),
        "recommended_extras": (),
        "summary": "Deploy-oriented ONNX wrapper for lightweight screening.",
    },
    {
        "name": "vision_openclip_patch_map",
        "priority": 70,
        "profiles": ("balanced",),
        "objectives": ("localization",),
        "required_extras": ("clip", "torch"),
        "recommended_extras": (),
        "summary": "OpenCLIP localization route for foundation-model style inspection.",
    },
)


def _normalize_objective(value: str | None) -> str:
    key = str(value or _DEFAULT_OBJECTIVE).strip().lower()
    if key not in {"balanced", "latency", "localization"}:
        raise ValueError("--objective must be one of: balanced, latency, localization.")
    return key


def _normalize_selection_profile(value: str | None) -> str:
    key = str(value or _DEFAULT_SELECTION_PROFILE).strip().lower()
    if key not in set(_PROFILE_DEFAULTS):
        raise ValueError(
            "--selection-profile must be one of: balanced, benchmark-parity, cpu-screening, deploy-readiness."
        )
    return key


def _normalize_topk(value: int | None, *, profile: str) -> int:
    raw = _PROFILE_DEFAULTS[profile]["topk"] if value is None else value
    topk = int(raw)
    if topk < 1:
        raise ValueError("--topk must be >= 1.")
    return topk


def _score_curated_pick(spec: dict[str, Any], *, objective: str, selection_profile: str) -> int:
    score = 0
    if selection_profile in set(spec.get("profiles", ())):
        score += 100
    if objective in set(spec.get("objectives", ())):
        score += 20
    return score


def _build_pick_payload(spec: dict[str, Any]) -> dict[str, Any]:
    import pyimgano.services.discovery_service as discovery_service

    required = sorted({str(item) for item in spec.get("required_extras", ()) if str(item).strip()})
    recommended = [
        item
        for item in sorted({str(item) for item in spec.get("recommended_extras", ()) if str(item).strip()})
        if item not in required
    ]
    combined = [*required, *recommended]
    missing = [extra for extra in combined if not extra_installed(extra)]
    install_hint = extras_install_hint(missing or combined) if combined else None
    info = discovery_service.build_model_info_payload(str(spec["name"]))
    deployment_profile = dict(info.get("deployment_profile", {}))
    return {
        "name": str(spec["name"]),
        "summary": str(spec["summary"]),
        "required_extras": required,
        "recommended_extras": recommended,
        "missing_extras": missing,
        "install_hint": install_hint,
        "supports_pixel_map": bool(info.get("supports_pixel_map", False)),
        "tested_runtime": str(deployment_profile.get("tested_runtime", "")),
        "deployment_family": [str(item) for item in deployment_profile.get("family", [])],
        "doctor_command": (
            f"pyimgano-doctor --recommend-extras --for-model {spec['name']} --json"
        ),
        "model_info_command": f"pyimgano-benchmark --model-info {spec['name']} --json",
    }


def collect_pyim_listing_payload(request: PyimListRequest) -> PyimListPayload:
    payload_kwargs = pyim_payload_collectors.empty_pyim_payload_kwargs()

    for field_name in request.requested_payload_fields():
        payload_kwargs[field_name] = pyim_payload_collectors.collect_pyim_payload_field(
            field_name,
            request,
        )

    return PyimListPayload(**payload_kwargs)


def collect_pyim_model_selection_payload(request: PyimListRequest) -> dict[str, Any]:
    import pyimgano.services.discovery_service as discovery_service

    selection_profile = _normalize_selection_profile(request.selection_profile)
    profile_defaults = _PROFILE_DEFAULTS[selection_profile]
    objective = _normalize_objective(
        request.objective if request.objective is not None else str(profile_defaults["objective"])
    )
    topk = _normalize_topk(request.topk, profile=selection_profile)

    allowed_names = set(
        discovery_service.list_discovery_model_names(
            tags=request.tags,
            family=request.family,
            algorithm_type=request.algorithm_type,
            year=request.year,
        )
    )

    ranked = sorted(
        (
            spec
            for spec in _CURATED_MODEL_PICKS
            if str(spec["name"]) in allowed_names
            and _score_curated_pick(
                spec,
                objective=objective,
                selection_profile=selection_profile,
            )
            > 0
        ),
        key=lambda spec: (
            -_score_curated_pick(
                spec,
                objective=objective,
                selection_profile=selection_profile,
            ),
            int(spec.get("priority", 999)),
            str(spec["name"]),
        ),
    )

    starter_picks = [_build_pick_payload(spec) for spec in ranked[:topk]]
    if not starter_picks:
        fallback_names = sorted(allowed_names)[:topk]
        starter_picks = [
            {
                "name": name,
                "summary": "Matches the current discovery filters.",
                "required_extras": [],
                "recommended_extras": [],
                "missing_extras": [],
                "install_hint": None,
            }
            for name in fallback_names
        ]

    suggested_commands: list[str] = []
    if starter_picks:
        top = starter_picks[0]
        for key in ("doctor_command", "model_info_command"):
            cmd = str(top.get(key, "")).strip()
            if cmd:
                suggested_commands.append(cmd)

    return {
        "selection_context": {
            "objective": objective,
            "selection_profile": selection_profile,
            "topk": int(topk),
        },
        "selection_profile_summary": {
            "requested": selection_profile,
            "description": str(profile_defaults["description"]),
        },
        "starter_picks": starter_picks,
        "suggested_commands": suggested_commands,
    }


__all__ = [
    "PyimListPayload",
    "PyimListRequest",
    "collect_pyim_model_selection_payload",
    "collect_pyim_listing_payload",
]
