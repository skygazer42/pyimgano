"""Curated parameter sweeps (import-light).

Sweeps are small grid searches over a baseline's JSON-friendly kwargs. They are
meant for industrial selection workflows where you want a bit more than a
single fixed preset, but still need:

- offline-safe defaults (no implicit weight downloads)
- bounded runtime (small search spaces)
- strict optional-deps behavior (missing extras => skipped)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SweepVariant:
    """One kwargs override variant for a baseline.

    `override` is deep-merged into the baseline's preset kwargs.
    """

    name: str
    override: Mapping[str, Any]
    description: str = ""


@dataclass(frozen=True)
class SweepPlan:
    name: str
    description: str
    variants_by_entry: Mapping[str, tuple[SweepVariant, ...]]


def _sweep_plan_from_mapping(obj: Mapping[str, Any]) -> SweepPlan:
    name = str(obj.get("name", "custom-sweep")).strip() or "custom-sweep"
    description = str(obj.get("description", "")).strip()

    raw_variants = obj.get("variants_by_entry", None)
    if not isinstance(raw_variants, Mapping):
        raise ValueError("Custom sweep JSON must contain a 'variants_by_entry' mapping.")

    variants_by_entry: dict[str, tuple[SweepVariant, ...]] = {}
    for entry_name, variants in raw_variants.items():
        if variants is None:
            variants_by_entry[str(entry_name)] = ()
            continue
        if not isinstance(variants, (list, tuple)):
            raise ValueError(
                "Custom sweep JSON 'variants_by_entry' values must be lists of variants."
            )

        out: list[SweepVariant] = []
        for v in variants:
            if not isinstance(v, Mapping):
                raise ValueError("Custom sweep variants must be JSON objects.")

            v_name = str(v.get("name", "")).strip()
            if not v_name:
                raise ValueError("Custom sweep variants must have a non-empty 'name'.")

            override = v.get("override", {})
            if override is None:
                override = {}
            if not isinstance(override, Mapping):
                raise ValueError("Custom sweep variant 'override' must be a JSON object (mapping).")

            v_desc = str(v.get("description", "")).strip()
            out.append(SweepVariant(name=v_name, override=dict(override), description=v_desc))

        variants_by_entry[str(entry_name)] = tuple(out)

    return SweepPlan(
        name=name,
        description=description,
        variants_by_entry=variants_by_entry,
    )


def load_sweep_json(path: str | Path) -> SweepPlan:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Custom sweep JSON must be an object (mapping).")
    return _sweep_plan_from_mapping(data)


def load_sweep_json_text(text: str) -> SweepPlan:
    data = json.loads(str(text))
    if not isinstance(data, Mapping):
        raise ValueError("Custom sweep JSON must be an object (mapping).")
    return _sweep_plan_from_mapping(data)


def resolve_sweep(spec: str) -> SweepPlan:
    """Resolve a sweep spec into a plan.

    Supported forms:
    - built-in sweep name (see `list_sweeps()`)
    - JSON file path (or prefixed with '@', e.g. '@/path/to/sweep.json')
    - inline JSON text (must start with '{')
    """

    s = str(spec).strip()
    if not s:
        raise ValueError("Sweep spec cannot be empty.")

    if s in _SWEEPS:
        return get_sweep(s)

    # File path: allow '@path.json' to avoid ambiguity with sweep names.
    candidate = s[1:] if s.startswith("@") else s
    p = Path(candidate)
    if p.exists() and p.is_file():
        return load_sweep_json(p)

    if s.startswith("{"):
        return load_sweep_json_text(s)

    available = ", ".join(list_sweeps()) or "<none>"
    raise KeyError(
        f"Unknown sweep {spec!r}. Available: {available}. "
        "You can also pass a JSON file path (e.g. --suite-sweep ./my_sweep.json)."
    )


def _industrial_small() -> SweepPlan:
    # Keep this intentionally small and CPU-friendly.
    return SweepPlan(
        name="industrial-small",
        description="Small, CPU-friendly sweep over a few stable industrial baseline knobs (topk/window/max_size).",
        variants_by_entry={
            # Structural: sweep feature max_size.
            "industrial-structural-ecod": (
                SweepVariant(
                    name="max_size_256",
                    override={"feature_extractor": {"kwargs": {"max_size": 256}}},
                    description="Structural features with smaller max_size (faster, may lose detail).",
                ),
                SweepVariant(
                    name="max_size_768",
                    override={"feature_extractor": {"kwargs": {"max_size": 768}}},
                    description="Structural features with larger max_size (more detail, slower).",
                ),
            ),
            "industrial-structural-iforest": (
                SweepVariant(
                    name="max_size_256",
                    override={"feature_extractor": {"kwargs": {"max_size": 256}}},
                ),
                SweepVariant(
                    name="max_size_768",
                    override={"feature_extractor": {"kwargs": {"max_size": 768}}},
                ),
            ),
            "industrial-structural-mst": (
                SweepVariant(
                    name="max_size_256",
                    override={"feature_extractor": {"kwargs": {"max_size": 256}}},
                ),
                SweepVariant(
                    name="max_size_768",
                    override={"feature_extractor": {"kwargs": {"max_size": 768}}},
                ),
            ),
            # Pixel stats: sweep topk reducer.
            "industrial-pixel-mean-absdiff-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
            ),
            "industrial-pixel-gaussian-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="channel_reduce_mean", override={"channel_reduce": "mean"}),
            ),
            "industrial-pixel-mad-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="max_train_64", override={"max_train_images": 64}),
            ),
            # Template NCC: sweep window and topk.
            "industrial-template-ncc-map": (
                SweepVariant(name="win_7", override={"window_hw": [7, 7]}),
                SweepVariant(name="win_21", override={"window_hw": [21, 21]}),
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
            ),
            # Optional skimage baselines: sweep topk only (when installed).
            "industrial-ssim-template-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
            ),
            "industrial-ssim-struct-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
            ),
            "industrial-phase-correlation-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
            ),
        },
    )


def _industrial_template_small() -> SweepPlan:
    # Focus on template-style pixel baselines: tune window/templates/topk/upsample.
    return SweepPlan(
        name="industrial-template-small",
        description="Small sweep focused on template inspection baselines (NCC/SSIM/phase-corr knobs).",
        variants_by_entry={
            "industrial-template-ncc-map": (
                SweepVariant(name="win_7", override={"window_hw": [7, 7]}),
                SweepVariant(name="win_21", override={"window_hw": [21, 21]}),
                SweepVariant(name="n_templates_3", override={"n_templates": 3}),
                SweepVariant(
                    name="reduction_mean",
                    override={"reduction": "mean"},
                    description="Use mean map reduction instead of top-k mean (sometimes better for diffuse defects).",
                ),
            ),
            # Optional skimage: SSIM template/struct map.
            "industrial-ssim-template-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="n_templates_3", override={"n_templates": 3}),
            ),
            "industrial-ssim-struct-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="n_templates_3", override={"n_templates": 3}),
            ),
            "industrial-phase-correlation-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="n_templates_3", override={"n_templates": 3}),
                SweepVariant(
                    name="upsample_10",
                    override={"upsample_factor": 10},
                    description="Higher upsample_factor can improve subpixel alignment (slower).",
                ),
            ),
        },
    )


def _industrial_pixel_small() -> SweepPlan:
    # Focus on per-pixel stats baselines. CPU-friendly knobs only.
    return SweepPlan(
        name="industrial-pixel-small",
        description="Small sweep focused on pixel-stats baselines (topk/reduction/channel_reduce/floors).",
        variants_by_entry={
            "industrial-pixel-mean-absdiff-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(
                    name="reduction_mean",
                    override={"reduction": "mean"},
                    description="Mean reduction can be more stable for large-area anomalies.",
                ),
                SweepVariant(name="reduction_max", override={"reduction": "max"}),
            ),
            "industrial-pixel-gaussian-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="channel_reduce_l2", override={"channel_reduce": "l2"}),
                SweepVariant(name="std_floor_0_5", override={"std_floor": 0.5}),
            ),
            "industrial-pixel-mad-map": (
                SweepVariant(name="topk_0005", override={"topk": 0.005}),
                SweepVariant(name="topk_002", override={"topk": 0.02}),
                SweepVariant(name="max_train_64", override={"max_train_images": 64}),
                SweepVariant(name="mad_floor_0_5", override={"mad_floor": 0.5}),
            ),
        },
    )


def _industrial_embedding_small() -> SweepPlan:
    # Optional torch baselines: keep it tiny to avoid heavy runtimes.
    return SweepPlan(
        name="industrial-embedding-small",
        description="Small sweep for embedding-based baselines (kNN k / score standardization method). Requires torch extras.",
        variants_by_entry={
            "industrial-embed-knn-cosine": (
                SweepVariant(name="k_3", override={"core_kwargs": {"n_neighbors": 3}}),
                SweepVariant(name="k_10", override={"core_kwargs": {"n_neighbors": 10}}),
            ),
            "industrial-embed-mahalanobis-shrinkage-rank": (
                SweepVariant(
                    name="std_zscore",
                    override={"core_kwargs": {"method": "zscore"}},
                    description="Switch score standardizer method from rank to zscore.",
                ),
                SweepVariant(
                    name="std_robust",
                    override={"core_kwargs": {"method": "robust"}},
                    description="Switch score standardizer method from rank to robust zscore.",
                ),
            ),
            "industrial-openclip-knn": (
                SweepVariant(name="k_3", override={"core_kwargs": {"n_neighbors": 3}}),
                SweepVariant(name="k_10", override={"core_kwargs": {"n_neighbors": 10}}),
            ),
        },
    )


def _industrial_deep_map_small() -> SweepPlan:
    # Optional torch baselines that output anomaly maps. Keep this tiny to avoid
    # exploding runtime for suite sweeps.
    return SweepPlan(
        name="industrial-deep-map-small",
        description=(
            "Small sweep for deep pixel-map baselines (PatchCore-lite-map / patch-embedding-core-map). "
            "Requires torch extras."
        ),
        variants_by_entry={
            "industrial-patchcore-lite-map": (
                SweepVariant(
                    name="coreset_005",
                    override={"coreset_sampling_ratio": 0.05},
                    description="Smaller coreset (faster, may reduce accuracy).",
                ),
                SweepVariant(
                    name="k_3",
                    override={"n_neighbors": 3},
                    description="Use k=3 neighbors (can stabilize noisy patch scores).",
                ),
                SweepVariant(
                    name="agg_topk_0005",
                    override={"aggregation_topk": 0.005},
                    description="More selective top-k aggregation (focus on smallest defect regions).",
                ),
            ),
            "industrial-patch-embedding-core-map": (
                SweepVariant(
                    name="agg_topk_0005",
                    override={"aggregation_topk": 0.005},
                    description="More selective top-k aggregation (focus on smallest defect regions).",
                ),
                SweepVariant(
                    name="core_ecod",
                    override={"core_detector": "core_ecod", "core_kwargs": {}},
                    description="Swap core detector to ECOD (fast, stable).",
                ),
                SweepVariant(
                    name="core_iforest",
                    override={
                        "core_detector": "core_iforest",
                        "core_kwargs": {"n_estimators": 200, "n_jobs": 1},
                    },
                    description="Swap core detector to IsolationForest (robust, slower).",
                ),
            ),
        },
    )


def _industrial_feature_small() -> SweepPlan:
    # CPU-friendly feature pipelines: focus on stable extractor knobs.
    return SweepPlan(
        name="industrial-feature-small",
        description=(
            "Small sweep for CPU-friendly feature pipelines (edge/patch-stats/color-hist/FFT). "
            "Optional texture entries are included when skimage extras are installed."
        ),
        variants_by_entry={
            "industrial-edge-ecod": (
                SweepVariant(
                    name="canny_soft",
                    override={
                        "feature_extractor": {
                            "kwargs": {"canny_threshold1": 30, "canny_threshold2": 100}
                        }
                    },
                    description="More sensitive Canny thresholds (can catch faint scratches; may increase noise).",
                ),
                SweepVariant(
                    name="canny_hard",
                    override={
                        "feature_extractor": {
                            "kwargs": {"canny_threshold1": 80, "canny_threshold2": 200}
                        }
                    },
                    description="Stricter Canny thresholds (more robust to texture noise).",
                ),
            ),
            "industrial-patch-stats-ecod": (
                SweepVariant(
                    name="grid_3x3",
                    override={"feature_extractor": {"kwargs": {"grid": [3, 3]}}},
                    description="Lower-dim patch grid (faster, coarser).",
                ),
                SweepVariant(
                    name="grid_6x6",
                    override={"feature_extractor": {"kwargs": {"grid": [6, 6]}}},
                    description="Higher-dim patch grid (slower, more local detail).",
                ),
                SweepVariant(
                    name="resize_192",
                    override={"feature_extractor": {"kwargs": {"resize_hw": [192, 192]}}},
                    description="Larger resize for finer patch statistics (slower).",
                ),
            ),
            "industrial-color-hist-ecod": (
                SweepVariant(
                    name="colorspace_lab",
                    override={"feature_extractor": {"kwargs": {"colorspace": "lab"}}},
                    description="LAB can be more stable under illumination changes than HSV in some factories.",
                ),
                SweepVariant(
                    name="bins_8",
                    override={"feature_extractor": {"kwargs": {"bins": [8, 8, 8]}}},
                    description="Coarser color histogram bins (faster, less sensitive).",
                ),
                SweepVariant(
                    name="bins_32",
                    override={"feature_extractor": {"kwargs": {"bins": [32, 32, 32]}}},
                    description="Finer histogram bins (more sensitive, higher variance).",
                ),
            ),
            "industrial-fft-lowfreq-ecod": (
                SweepVariant(
                    name="size_48",
                    override={"feature_extractor": {"kwargs": {"size_hw": [48, 48]}}},
                    description="Smaller FFT grid (faster).",
                ),
                SweepVariant(
                    name="size_96",
                    override={"feature_extractor": {"kwargs": {"size_hw": [96, 96]}}},
                    description="Larger FFT grid (slower, more frequency detail).",
                ),
                SweepVariant(
                    name="radii_alt",
                    override={"feature_extractor": {"kwargs": {"radii": [3, 6, 12]}}},
                    description="Alternative low-frequency radii; sometimes better for periodic textures.",
                ),
            ),
            # Optional skimage texture baselines (when installed).
            "industrial-lbp-ecod": (
                SweepVariant(
                    name="p16_r2",
                    override={"feature_extractor": {"kwargs": {"n_points": 16, "radius": 2.0}}},
                ),
            ),
            "industrial-hog-ecod": (
                SweepVariant(
                    name="ori_6", override={"feature_extractor": {"kwargs": {"orientations": 6}}}
                ),
                SweepVariant(
                    name="ori_12", override={"feature_extractor": {"kwargs": {"orientations": 12}}}
                ),
            ),
            "industrial-gabor-ecod": (
                SweepVariant(
                    name="freq_alt",
                    override={"feature_extractor": {"kwargs": {"frequencies": [0.15, 0.25, 0.35]}}},
                ),
            ),
        },
    )


_SWEEPS: dict[str, SweepPlan] = {
    "industrial-small": _industrial_small(),
    "industrial-template-small": _industrial_template_small(),
    "industrial-pixel-small": _industrial_pixel_small(),
    "industrial-embedding-small": _industrial_embedding_small(),
    "industrial-deep-map-small": _industrial_deep_map_small(),
    "industrial-feature-small": _industrial_feature_small(),
}


def list_sweeps() -> list[str]:
    return sorted(_SWEEPS.keys())


def get_sweep(name: str) -> SweepPlan:
    key = str(name).strip()
    if key not in _SWEEPS:
        available = ", ".join(list_sweeps()) or "<none>"
        raise KeyError(f"Unknown sweep {name!r}. Available: {available}")
    return _SWEEPS[key]


def variants_for_entry(sweep_name: str, entry_name: str) -> tuple[SweepVariant, ...]:
    plan = get_sweep(sweep_name)
    return tuple(plan.variants_by_entry.get(str(entry_name), ()))


__all__ = [
    "SweepVariant",
    "SweepPlan",
    "list_sweeps",
    "get_sweep",
    "resolve_sweep",
    "load_sweep_json",
    "load_sweep_json_text",
    "variants_for_entry",
]
