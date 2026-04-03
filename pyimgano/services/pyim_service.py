from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pyimgano.services.pyim_payload_collectors as pyim_payload_collectors
from pyimgano.config import load_config
from pyimgano.pyim_contracts import PyimListPayload, PyimListRequest
from pyimgano.utils.extras import extra_installed
from pyimgano.utils.extras import extras_install_hint
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workflow_guidance import starter_path_by_name


_DEFAULT_OBJECTIVE = "balanced"
_DEFAULT_SELECTION_PROFILE = "balanced"
_DEFAULT_TOPK = 5
_REPO_ROOT = Path(__file__).resolve().parents[2]

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

_GOAL_SPECS: dict[str, dict[str, Any]] = {
    "first-run": {
        "title": "First Run",
        "summary": "Fastest offline-safe route from discovery to a bounded benchmark and inference artifact.",
        "objective": "latency",
        "selection_profile": "cpu-screening",
        "topk": 3,
        "starter_path": "first-run",
        "recipe_picks": [
            {
                "name": "industrial-adapt",
                "summary": "Audited train/export loop once you move beyond the demo path.",
            }
        ],
        "dataset_picks": [
            {
                "name": "custom",
                "summary": "Folder-layout starter dataset path for the demo and first real trials.",
            },
            {
                "name": "manifest",
                "summary": "Use when your production data does not match the built-in folder layout.",
            },
        ],
    },
    "cpu-screening": {
        "title": "CPU Screening",
        "summary": "Prefer lightweight CPU baselines and low-friction install paths.",
        "objective": "latency",
        "selection_profile": "cpu-screening",
        "topk": 3,
        "starter_path": "benchmark",
        "recipe_picks": [
            {
                "name": "classical-colorhist-mahalanobis",
                "summary": "CPU-only screening recipe using HSV color histograms plus Mahalanobis distance.",
            },
            {
                "name": "classical-edge-ecod",
                "summary": "CPU-only screening recipe using edge-statistics features plus ECOD.",
            },
            {
                "name": "classical-fft-lowfreq-ecod",
                "summary": "CPU-only screening recipe using FFT low-frequency energy ratios plus ECOD.",
            },
            {
                "name": "classical-hog-ecod",
                "summary": "CPU-only screening recipe using HOG features plus ECOD.",
            },
            {
                "name": "classical-lbp-loop",
                "summary": "CPU-only screening recipe using LBP features plus LoOP.",
            },
            {
                "name": "classical-patch-stats-ecod",
                "summary": "CPU-only screening recipe using patch-grid statistics plus ECOD.",
            },
            {
                "name": "classical-structural-ecod",
                "summary": "Lowest-friction workbench recipe for CPU-only screening with structural features.",
            },
            {
                "name": "industrial-adapt",
                "summary": "Starter workbench recipe once CPU screening narrows the candidate set.",
            }
        ],
        "dataset_picks": [
            {
                "name": "custom",
                "summary": "Fastest path for local CPU screening on your own folder-layout dataset.",
            },
            {
                "name": "mvtec",
                "summary": "Public benchmark reference for CPU-friendly baseline comparisons.",
            },
        ],
    },
    "pixel-localization": {
        "title": "Pixel Localization",
        "summary": "Prioritize anomaly-map quality and defect export readiness.",
        "objective": "localization",
        "selection_profile": "balanced",
        "topk": 3,
        "starter_path": "benchmark",
        "recipe_picks": [
            {
                "name": "industrial-adapt-fp40",
                "expand_starter_configs": True,
                "starter_summaries": {
                    "examples/configs/industrial_adapt_defects_fp40.json": (
                        "Deploy-style inference defaults for pixel-map and defect export loops."
                    ),
                    "examples/configs/industrial_adapt_defects_roi.json": (
                        "ROI-first defect export starter when you want a simpler false-positive reduction path."
                    ),
                },
            },
            {
                "name": "industrial-adapt",
                "summary": "General workbench recipe when you need adaptation-first training plus maps.",
                "config_path": "examples/configs/industrial_adapt_maps_tiling.json",
            },
        ],
        "dataset_picks": [
            {
                "name": "custom",
                "summary": "Own defect masks or aligned inspection data for direct pixel-map validation.",
            },
            {
                "name": "mvtec",
                "summary": "Public pixel-localization benchmark reference.",
            },
        ],
    },
    "deployable": {
        "title": "Deployable",
        "summary": "Favor lower-friction runtime/export paths and audited artifact loops.",
        "objective": "balanced",
        "selection_profile": "deploy-readiness",
        "topk": 4,
        "starter_path": "deploy-smoke",
        "recipe_picks": [
            {
                "name": "industrial-adapt",
                "expand_starter_configs": True,
                "starter_summaries": {
                    "examples/configs/deploy_smoke_custom_cpu.json": (
                        "Smallest offline-safe deploy-bundle smoke path."
                    ),
                    "examples/configs/industrial_adapt_audited.json": (
                        "Audited GPU-backed train/export route for stronger deployable handoff coverage."
                    ),
                    "examples/configs/manifest_industrial_workflow_balanced.json": (
                        "Manifest-first deploy route when ingestion must preserve explicit metadata and paths."
                    ),
                },
            }
        ],
        "dataset_picks": [
            {
                "name": "custom",
                "summary": "Direct path when you already have production-like folder-layout data.",
            },
            {
                "name": "manifest",
                "summary": "Preferred when deployment data ingestion needs explicit metadata and paths.",
            },
        ],
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


@lru_cache(maxsize=None)
def _load_recipe_pick_config_meta(config_path: str) -> dict[str, Any]:
    cfg = WorkbenchConfig.from_dict(load_config(_REPO_ROOT / str(config_path)))
    return {
        "name": str(cfg.recipe),
        "purpose": cfg.meta.purpose,
        "runtime_profile": cfg.meta.runtime_profile,
        "required_extras": [str(item) for item in cfg.meta.required_extras],
        "expected_artifacts": [str(item) for item in cfg.meta.expected_artifacts],
    }


@lru_cache(maxsize=None)
def _load_recipe_metadata(recipe_name: str) -> dict[str, Any]:
    import pyimgano.recipes  # noqa: F401
    from pyimgano.recipes.registry import recipe_info as _recipe_info

    info = _recipe_info(str(recipe_name))
    return dict(info.get("metadata", {}) or {})


def _build_goal_recipe_pick(spec: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(spec)
    config_path = str(enriched.get("config_path", "")).strip()
    if not config_path:
        recipe_name = str(enriched.get("name", "")).strip()
        if recipe_name:
            try:
                recipe_meta = _load_recipe_metadata(recipe_name)
            except Exception:
                recipe_meta = {}
            config_path = str(recipe_meta.get("default_config", "")).strip()
            if config_path:
                enriched["config_path"] = config_path
    if not config_path:
        return enriched

    try:
        config_meta = _load_recipe_pick_config_meta(config_path)
    except Exception:
        return enriched

    recipe_name = str(config_meta.get("name", "")).strip()
    if recipe_name and not str(enriched.get("name", "")).strip():
        enriched["name"] = recipe_name

    for key in ("runtime_profile", "required_extras", "expected_artifacts"):
        if not enriched.get(key):
            enriched[key] = config_meta.get(key, enriched.get(key))

    recipe_name = str(enriched.get("name", "")).strip()
    required_extras = sorted(
        {str(item) for item in enriched.get("required_extras", ()) or [] if str(item).strip()}
    )
    if required_extras:
        enriched["required_extras"] = required_extras
        enriched["install_hint"] = extras_install_hint(required_extras)
    enriched["recipe_list_command"] = "pyimgano train --list-recipes"
    if recipe_name:
        enriched["recipe_info_command"] = f"pyimgano train --recipe-info {recipe_name} --json"
    if config_path:
        enriched["dry_run_command"] = f"pyimgano train --dry-run --config {config_path}"
    if config_path:
        enriched["preflight_command"] = (
            f"pyimgano train --preflight --config {config_path} --json"
        )
    if config_path:
        enriched["recipe_run_command"] = f"pyimgano train --config {config_path}"

    return enriched


def _expand_goal_recipe_picks(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for raw_spec in specs:
        spec = dict(raw_spec)
        if not bool(spec.get("expand_starter_configs")):
            expanded.append(_build_goal_recipe_pick(spec))
            continue

        recipe_name = str(spec.get("name", "")).strip()
        starter_summaries = {
            str(path): str(summary)
            for path, summary in dict(spec.get("starter_summaries", {}) or {}).items()
            if str(path).strip()
        }

        starter_configs: list[str] = []
        try:
            recipe_meta = _load_recipe_metadata(recipe_name)
        except Exception:
            recipe_meta = {}

        starter_configs.extend(
            str(item)
            for item in recipe_meta.get("starter_configs", []) or []
            if str(item).strip()
        )
        if not starter_configs:
            default_config = str(recipe_meta.get("default_config", "")).strip()
            if default_config:
                starter_configs.append(default_config)

        spec_template = {
            key: value
            for key, value in spec.items()
            if key not in {"expand_starter_configs", "starter_summaries"}
        }
        if not starter_configs:
            expanded.append(_build_goal_recipe_pick(spec_template))
            continue

        for config_path in starter_configs:
            pick = dict(spec_template)
            pick["config_path"] = str(config_path)
            summary = starter_summaries.get(str(config_path), "").strip()
            if summary:
                pick["summary"] = summary
            expanded.append(_build_goal_recipe_pick(pick))

    return expanded


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
    required = sorted({str(item) for item in spec.get("required_extras", ()) if str(item).strip()})
    recommended = [
        item
        for item in sorted({str(item) for item in spec.get("recommended_extras", ()) if str(item).strip()})
        if item not in required
    ]
    combined = [*required, *recommended]
    missing = [extra for extra in combined if not extra_installed(extra)]
    install_hint = extras_install_hint(missing or combined) if combined else None
    info = pyim_payload_collectors.build_model_info_payload(str(spec["name"]))
    deployment_profile = dict(info.get("deployment_profile", {}))
    return {
        "name": str(spec["name"]),
        "summary": str(spec["summary"]),
        "why_this_pick": (
            f"{spec['summary']} Fits the current starter objective without requiring a separate registry lookup."
        ),
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


def _goal_spec(goal: str) -> dict[str, Any]:
    key = str(goal).strip().lower()
    try:
        return dict(_GOAL_SPECS[key])
    except KeyError as exc:
        raise ValueError(
            "--goal must be one of: cpu-screening, deployable, first-run, pixel-localization."
        ) from exc


def _request_for_goal(request: PyimListRequest, spec: dict[str, Any]) -> PyimListRequest:
    return PyimListRequest(
        list_kind="models",
        tags=request.tags,
        family=request.family,
        algorithm_type=request.algorithm_type,
        year=request.year,
        deployable_only=bool(request.deployable_only),
        goal=request.goal,
        objective=str(spec["objective"]),
        selection_profile=str(spec["selection_profile"]),
        topk=int(spec["topk"]),
    )


def collect_pyim_listing_payload(request: PyimListRequest) -> PyimListPayload:
    payload_kwargs = pyim_payload_collectors.empty_pyim_payload_kwargs()

    for field_name in request.requested_payload_fields():
        payload_kwargs[field_name] = pyim_payload_collectors.collect_pyim_payload_field(
            field_name,
            request,
        )

    return PyimListPayload(**payload_kwargs)


def collect_pyim_model_selection_payload(request: PyimListRequest) -> dict[str, Any]:
    selection_profile = _normalize_selection_profile(request.selection_profile)
    profile_defaults = _PROFILE_DEFAULTS[selection_profile]
    objective = _normalize_objective(
        request.objective if request.objective is not None else str(profile_defaults["objective"])
    )
    topk = _normalize_topk(request.topk, profile=selection_profile)

    allowed_names = set(pyim_payload_collectors.list_filtered_model_names(request))

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


def _extend_goal_suggested_commands_with_recipe_followups(
    commands: list[str],
    recipe_picks: list[dict[str, Any]],
) -> list[str]:
    deduped = [str(item) for item in commands if str(item).strip()]
    seen = set(deduped)
    has_train_config_command = any(item.startswith("pyimgano train --config ") for item in deduped)

    if not recipe_picks:
        return deduped

    top_recipe = next(
        (
            item
            for item in recipe_picks
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ),
        None,
    )
    if top_recipe is None:
        return deduped

    list_cmd = "pyimgano train --list-recipes"
    if list_cmd not in seen:
        deduped.append(list_cmd)
        seen.add(list_cmd)

    for key in ("recipe_info_command",):
        cmd = str(top_recipe.get(key, "")).strip()
        if cmd and cmd not in seen:
            deduped.append(cmd)
            seen.add(cmd)

    dry_run_cmd = str(top_recipe.get("dry_run_command", "")).strip()
    if dry_run_cmd and dry_run_cmd not in seen:
        deduped.append(dry_run_cmd)
        seen.add(dry_run_cmd)

    preflight_cmd = str(top_recipe.get("preflight_command", "")).strip()
    if preflight_cmd and preflight_cmd not in seen:
        deduped.append(preflight_cmd)
        seen.add(preflight_cmd)

    run_cmd = str(top_recipe.get("recipe_run_command", "")).strip()
    if run_cmd and run_cmd not in seen and not has_train_config_command:
        deduped.append(run_cmd)

    return deduped


def collect_pyim_goal_payload(request: PyimListRequest) -> dict[str, Any]:
    if request.goal is None:
        raise ValueError("collect_pyim_goal_payload requires request.goal.")

    spec = _goal_spec(str(request.goal))
    selection_request = _request_for_goal(request, spec)
    selection_payload = collect_pyim_model_selection_payload(selection_request)
    starter_path = starter_path_by_name(str(spec["starter_path"]))
    suggested_commands = (
        [] if starter_path is None else [str(item) for item in starter_path.commands]
    )
    recipe_picks = _expand_goal_recipe_picks([dict(item) for item in spec.get("recipe_picks", [])])
    suggested_commands = _extend_goal_suggested_commands_with_recipe_followups(
        suggested_commands,
        recipe_picks,
    )

    return {
        "goal_context": {
            "goal": str(request.goal),
            "title": str(spec["title"]),
            "summary": str(spec["summary"]),
            "objective": str(spec["objective"]),
            "selection_profile": str(spec["selection_profile"]),
            "topk": int(spec["topk"]),
        },
        "goal_picks": {
            "models": list(selection_payload.get("starter_picks", [])),
            "recipes": recipe_picks,
            "datasets": [dict(item) for item in spec.get("dataset_picks", [])],
        },
        "suggested_commands": suggested_commands,
    }


__all__ = [
    "PyimListPayload",
    "PyimListRequest",
    "collect_pyim_goal_payload",
    "collect_pyim_model_selection_payload",
    "collect_pyim_listing_payload",
]
