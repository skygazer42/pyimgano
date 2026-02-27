from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from pyimgano.recipes.registry import register_recipe
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.runner import run_workbench


@register_recipe(
    "classical-hog-ecod",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: HOG features + ECOD"},
)
def classical_hog_ecod(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-hog-ecod"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault("feature_extractor", {"name": "hog", "kwargs": {"resize_hw": [128, 128]}})
    cfg = replace(config, model=replace(config.model, name="vision_ecod", model_kwargs=model_kwargs))
    return run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-lbp-loop",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: LBP features + LoOP"},
)
def classical_lbp_loop(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-lbp-loop"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        "feature_extractor",
        {"name": "lbp", "kwargs": {"n_points": 8, "radius": 1.0, "method": "uniform"}},
    )
    model_kwargs.setdefault("n_neighbors", 15)
    cfg = replace(config, model=replace(config.model, name="vision_loop", model_kwargs=model_kwargs))
    return run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-colorhist-mahalanobis",
    tags=("builtin", "classical"),
    metadata={"description": "Classical baseline: HSV color hist + Mahalanobis"},
)
def classical_colorhist_mahalanobis(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "classical-colorhist-mahalanobis"
    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault(
        "feature_extractor",
        {"name": "color_hist", "kwargs": {"colorspace": "hsv", "bins": [16, 16, 16]}},
    )
    model_kwargs.setdefault("reg", 1e-6)
    cfg = replace(
        config,
        model=replace(config.model, name="vision_mahalanobis", model_kwargs=model_kwargs),
    )
    return run_workbench(config=cfg, recipe_name=recipe_name)


@register_recipe(
    "classical-struct-iforest-synth",
    tags=("builtin", "classical", "synthesis"),
    metadata={
        "description": (
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
        out_base = Path(config.output.output_dir) / "__synthetic_datasets__" / f"{recipe_name}_{seed}"
        out_base.mkdir(parents=True, exist_ok=True)
        synth_root = out_base / "data"
        synthesize_dataset(
            in_dir=in_dir,
            out_root=synth_root,
            category="synthetic",
            preset="scratch",
            seed=seed,
            n_train=8 if config.dataset.limit_train is None else int(config.dataset.limit_train),
            n_test_normal=4,
            n_test_anomaly=4 if config.dataset.limit_test is None else int(config.dataset.limit_test),
            manifest_path=synth_root / "manifest.jsonl",
            absolute_paths=True,
        )
        manifest_path = synth_root / "manifest.jsonl"
    else:
        # CI-/demo-friendly fallback when output_dir isn't set.
        with TemporaryDirectory(prefix="pyimgano_synth_") as td:
            synth_root = Path(td) / "data"
            synthesize_dataset(
                in_dir=in_dir,
                out_root=synth_root,
                category="synthetic",
                preset="scratch",
                seed=seed,
                n_train=8,
                n_test_normal=4,
                n_test_anomaly=4,
                manifest_path=synth_root / "manifest.jsonl",
                absolute_paths=True,
            )
            manifest_path = synth_root / "manifest.jsonl"

            model_kwargs = dict(config.model.model_kwargs)
            model_kwargs.setdefault(
                "feature_extractor", {"name": "structural", "kwargs": {"max_size": 512}}
            )
            cfg = replace(
                config,
                dataset=replace(
                    config.dataset,
                    name="manifest",
                    root=str(synth_root),
                    manifest_path=str(manifest_path),
                    category="synthetic",
                    input_mode="paths",
                ),
                model=replace(config.model, name="vision_iforest", model_kwargs=model_kwargs),
            )
            return run_workbench(config=cfg, recipe_name=recipe_name)

    model_kwargs = dict(config.model.model_kwargs)
    model_kwargs.setdefault("feature_extractor", {"name": "structural", "kwargs": {"max_size": 512}})
    cfg = replace(
        config,
        dataset=replace(
            config.dataset,
            name="manifest",
            root=str(synth_root),
            manifest_path=str(manifest_path),
            category="synthetic",
            input_mode="paths",
        ),
        model=replace(config.model, name="vision_iforest", model_kwargs=model_kwargs),
    )
    return run_workbench(config=cfg, recipe_name=recipe_name)
