from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pyimgano.recipes.registry import register_recipe
from pyimgano.reporting.environment import collect_environment
from pyimgano.reporting.report import save_run_report, stamp_report_payload
from pyimgano.reporting.runs import build_workbench_run_dir_name, build_workbench_run_paths, ensure_run_dir
from pyimgano.training.checkpointing import save_checkpoint
from pyimgano.training.runner import micro_finetune
from pyimgano.workbench.config import WorkbenchConfig


def _load_train_paths(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
) -> list[str]:
    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    split = load_benchmark_split(
        dataset=dataset,  # type: ignore[arg-type]
        root=root,
        category=category,
        resize=tuple(resize),
        load_masks=False,
    )
    return list(split.train_paths)


def _load_train_numpy(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
) -> list[np.ndarray]:
    from pyimgano.datasets import load_dataset

    ds = load_dataset(
        dataset,
        root,
        category=category,
        resize=tuple(resize),
        load_masks=False,
    )
    train_data = np.asarray(ds.get_train_data())
    return [np.asarray(train_data[i]) for i in range(int(train_data.shape[0]))]


def _create_detector(config: WorkbenchConfig) -> Any:
    import pyimgano.models  # noqa: F401
    from pyimgano.cli import _resolve_preset_kwargs
    from pyimgano.cli_common import build_model_kwargs
    from pyimgano.models.registry import create_model

    user_kwargs = dict(config.model.model_kwargs)
    if config.model.checkpoint_path is not None:
        user_kwargs.setdefault("checkpoint_path", str(config.model.checkpoint_path))

    preset_kwargs = _resolve_preset_kwargs(config.model.preset, config.model.name)

    auto_kwargs: dict[str, Any] = {
        "device": config.model.device,
        "contamination": float(config.model.contamination),
        "pretrained": bool(config.model.pretrained),
    }
    if config.seed is not None:
        auto_kwargs["random_seed"] = int(config.seed)
        auto_kwargs["random_state"] = int(config.seed)

    model_kwargs = build_model_kwargs(
        config.model.name,
        user_kwargs=user_kwargs,
        preset_kwargs=preset_kwargs,
        auto_kwargs=auto_kwargs,
    )
    return create_model(config.model.name, **model_kwargs)


@register_recipe(
    "micro-finetune-autoencoder",
    tags=("builtin", "training"),
    metadata={
        "description": "Micro-finetune recipe intended for small autoencoder-style models (best-effort).",
    },
)
def micro_finetune_autoencoder(config: WorkbenchConfig) -> dict[str, Any]:
    recipe_name = "micro-finetune-autoencoder"

    if not bool(config.output.save_run):
        raise ValueError(f"{recipe_name} requires output.save_run=true (needs checkpoint artifacts).")

    category_for_name = None if str(config.dataset.category).lower() == "all" else str(config.dataset.category)
    run_name = build_workbench_run_dir_name(
        dataset=str(config.dataset.name),
        recipe=recipe_name,
        model=str(config.model.name),
        category=category_for_name,
    )
    run_dir = ensure_run_dir(output_dir=config.output.output_dir, name=run_name)
    paths = build_workbench_run_paths(run_dir)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

    save_run_report(paths.environment_json, collect_environment())
    save_run_report(paths.config_json, {"config": asdict(config)})

    if config.dataset.input_mode == "paths":
        train_inputs: Sequence[Any] = _load_train_paths(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            category=str(config.dataset.category),
            resize=tuple(config.dataset.resize),
        )
    elif config.dataset.input_mode == "numpy":
        train_inputs = _load_train_numpy(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            category=str(config.dataset.category),
            resize=tuple(config.dataset.resize),
        )
    else:
        raise ValueError(
            f"Unknown input_mode: {config.dataset.input_mode!r}. Choose from: paths, numpy."
        )

    if config.dataset.limit_train is not None:
        train_inputs = list(train_inputs)[: int(config.dataset.limit_train)]

    detector = _create_detector(config)
    training = micro_finetune(detector, train_inputs, seed=config.seed)

    checkpoint_path = paths.checkpoints_dir / "model.pt"
    saved = save_checkpoint(detector, checkpoint_path)
    try:
        rel = saved.relative_to(paths.run_dir)
        checkpoint_rel = str(rel)
    except Exception:
        checkpoint_rel = str(saved)

    payload: dict[str, Any] = {
        "dataset": str(config.dataset.name),
        "category": str(config.dataset.category),
        "model": str(config.model.name),
        "recipe": recipe_name,
        "input_mode": str(config.dataset.input_mode),
        "device": str(config.model.device),
        "preset": config.model.preset,
        "resize": [int(config.dataset.resize[0]), int(config.dataset.resize[1])],
        "checkpoint": {"path": checkpoint_rel},
        "training": training,
        "run_dir": str(paths.run_dir),
    }
    payload = stamp_report_payload(payload)

    save_run_report(paths.report_json, payload)
    return payload

