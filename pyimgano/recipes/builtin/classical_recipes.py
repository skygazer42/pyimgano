from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pyimgano.services.workbench_service as workbench_service
from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import WorkbenchConfig

BUILTIN_TAG = "builtin"
CLASSICAL_TAG = "classical"
CPU_TAG = "cpu"
SYNTHESIS_TAG = "synthesis"
DESCRIPTION_KEY = "description"
FEATURE_EXTRACTOR_KEY = "feature_extractor"
KWARGS_KEY = "kwargs"
MANIFEST_FILENAME = "manifest.jsonl"
SCRATCH_PRESET = "scratch"
SYNTHETIC_CATEGORY = "synthetic"
VISION_ECOD_MODEL = "vision_ecod"


def _feature_extractor_spec(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, KWARGS_KEY: kwargs}


def _run_structural_iforest_synthetic_manifest(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
    synth_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("structural", {"max_size": 512}),
    )
    cfg: WorkbenchConfig = replace(
        config,
        dataset=replace(
            config.dataset,
            name="manifest",
            root=str(synth_root),
            manifest_path=str(manifest_path),
            category=SYNTHETIC_CATEGORY,
            input_mode="paths",
        ),
        model=replace(config.model, name="vision_iforest", model_kwargs=model_kwargs),
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-hog-ecod",
    tags=(BUILTIN_TAG, CLASSICAL_TAG),
    metadata={DESCRIPTION_KEY: "Classical baseline: HOG features + ECOD"},
)
def classical_hog_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-hog-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("hog", {"resize_hw": [128, 128]}),
    )
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name=VISION_ECOD_MODEL, model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-structural-ecod",
    tags=(BUILTIN_TAG, CLASSICAL_TAG, CPU_TAG),
    metadata={DESCRIPTION_KEY: "CPU-friendly baseline: structural features + ECOD"},
)
def classical_structural_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-structural-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("structural", {"max_size": 512}),
    )
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name=VISION_ECOD_MODEL, model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-edge-ecod",
    tags=(BUILTIN_TAG, CLASSICAL_TAG, CPU_TAG),
    metadata={DESCRIPTION_KEY: "CPU-friendly baseline: edge statistics features + ECOD"},
)
def classical_edge_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-edge-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec(
            "edge_stats",
            {"canny_threshold1": 50, "canny_threshold2": 150, "sobel_ksize": 3},
        ),
    )
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name=VISION_ECOD_MODEL, model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-patch-stats-ecod",
    tags=(BUILTIN_TAG, CLASSICAL_TAG, CPU_TAG),
    metadata={DESCRIPTION_KEY: "CPU-friendly baseline: patch-grid statistics features + ECOD"},
)
def classical_patch_stats_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-patch-stats-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec(
            "patch_stats",
            {
                "grid": [4, 4],
                "stats": ["mean", "std", "skew", "kurt"],
                "resize_hw": [128, 128],
            },
        ),
    )
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name=VISION_ECOD_MODEL, model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-fft-lowfreq-ecod",
    tags=(BUILTIN_TAG, CLASSICAL_TAG, CPU_TAG),
    metadata={DESCRIPTION_KEY: "CPU-friendly baseline: FFT low-frequency energy ratios + ECOD"},
)
def classical_fft_lowfreq_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-fft-lowfreq-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("fft_lowfreq", {"size_hw": [64, 64], "radii": [4, 8, 16]}),
    )
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name=VISION_ECOD_MODEL, model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-lbp-loop",
    tags=(BUILTIN_TAG, CLASSICAL_TAG),
    metadata={DESCRIPTION_KEY: "Classical baseline: LBP features + LoOP"},
)
def classical_lbp_loop(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-lbp-loop"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("lbp", {"n_points": 8, "radius": 1.0, "method": "uniform"}),
    )
    model_kwargs.setdefault("n_neighbors", 15)
    cfg: WorkbenchConfig = replace(
        config, model=replace(config.model, name="vision_loop", model_kwargs=model_kwargs)
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-colorhist-mahalanobis",
    tags=(BUILTIN_TAG, CLASSICAL_TAG),
    metadata={DESCRIPTION_KEY: "Classical baseline: HSV color hist + Mahalanobis"},
)
def classical_colorhist_mahalanobis(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-colorhist-mahalanobis"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        FEATURE_EXTRACTOR_KEY,
        _feature_extractor_spec("color_hist", {"colorspace": "hsv", "bins": [16, 16, 16]}),
    )
    model_kwargs.setdefault("reg", 1e-6)
    cfg: WorkbenchConfig = replace(
        config,
        model=replace(config.model, name="vision_mahalanobis", model_kwargs=model_kwargs),
    )
    return workbench_service.run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-struct-iforest-synth",
    tags=(BUILTIN_TAG, CLASSICAL_TAG, SYNTHESIS_TAG),
    metadata={
        DESCRIPTION_KEY: (
            "One-click demo: generate a tiny synthetic dataset from a flat folder of normal images "
            "and run a structural-features IForest baseline on the resulting manifest."
        )
    },
)
def classical_struct_iforest_synth(config: WorkbenchConfig) -> dict[str, Any]:
    """Best-effort recipe that uses synthesis as a data source.

    This recipe interprets `config.dataset.root` as a directory containing normal images
    (flat folder). It generates a tiny dataset in `custom` layout + JSONL manifest, then
    runs `run_workbench(...)` with `dataset.name='manifest'`.
    """

    from tempfile import TemporaryDirectory

    from pyimgano.synthesize_cli import synthesize_dataset

    recipe_name = "classical-struct-iforest-synth"
    seed = 0 if config.seed is None else int(config.seed)

    in_dir = Path(config.dataset.root)
    if not in_dir.exists():
        raise ValueError(f"dataset.root directory does not exist: {in_dir}")

    if config.output.output_dir is not None:
        out_base = (
            Path(config.output.output_dir) / "__synthetic_datasets__" / f"{recipe_name}_{seed}"
        )
        out_base.mkdir(parents=True, exist_ok=True)
        synth_root = out_base / "data"
        synthesize_dataset(
            in_dir=in_dir,
            out_root=synth_root,
            category=SYNTHETIC_CATEGORY,
            preset=SCRATCH_PRESET,
            seed=seed,
            n_train=8 if config.dataset.limit_train is None else int(config.dataset.limit_train),
            n_test_normal=4,
            n_test_anomaly=(
                4 if config.dataset.limit_test is None else int(config.dataset.limit_test)
            ),
            manifest_path=synth_root / MANIFEST_FILENAME,
            absolute_paths=True,
        )
        manifest_path = synth_root / MANIFEST_FILENAME
    else:
        # CI-/demo-friendly fallback when output_dir isn't set.
        with TemporaryDirectory(prefix="pyimgano_synth_") as td:
            synth_root = Path(td) / "data"
            synthesize_dataset(
                in_dir=in_dir,
                out_root=synth_root,
                category=SYNTHETIC_CATEGORY,
                preset=SCRATCH_PRESET,
                seed=seed,
                n_train=8,
                n_test_normal=4,
                n_test_anomaly=4,
                manifest_path=synth_root / MANIFEST_FILENAME,
                absolute_paths=True,
            )
            manifest_path = synth_root / MANIFEST_FILENAME
            return _run_structural_iforest_synthetic_manifest(
                config=config,
                recipe_name=recipe_name,
                synth_root=synth_root,
                manifest_path=manifest_path,
            )

    return _run_structural_iforest_synthetic_manifest(
        config=config,
        recipe_name=recipe_name,
        synth_root=synth_root,
        manifest_path=manifest_path,
    )
