from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pyimgano.reporting.runs import build_workbench_run_paths
from pyimgano.workbench.config import WorkbenchConfig


@dataclass(frozen=True)
class WorkbenchTrainingResult:
    detector: Any
    training_report: dict[str, Any] | None
    checkpoint_meta: dict[str, Any] | None


def _build_fit_kwargs(config: WorkbenchConfig) -> dict[str, Any]:
    fit_kwargs: dict[str, Any] = {}
    if config.training.epochs is not None:
        fit_kwargs["epochs"] = int(config.training.epochs)
    if config.training.lr is not None:
        fit_kwargs["lr"] = float(config.training.lr)
    return fit_kwargs


def _build_checkpoint_meta(*, saved: Path, run_dir: Path) -> dict[str, Any]:
    try:
        rel = saved.relative_to(run_dir)
        return {"path": rel.as_posix()}
    except Exception:
        return {"path": saved.as_posix()}


def run_workbench_training(
    *,
    detector: Any,
    train_inputs: Sequence[Any],
    config: WorkbenchConfig,
    category: str,
    run_dir: str | Path | None,
) -> WorkbenchTrainingResult:
    if bool(getattr(config, "training", None) and config.training.enabled):
        from pyimgano.training.checkpointing import save_checkpoint
        from pyimgano.training.runner import micro_finetune

        training_report = micro_finetune(
            detector,
            train_inputs,
            seed=config.seed,
            fit_kwargs=_build_fit_kwargs(config),
        )

        checkpoint_meta = None
        if run_dir is not None and bool(config.output.save_run):
            run_path = Path(run_dir)
            cat_ckpt_dir = build_workbench_run_paths(run_path).checkpoints_dir / str(category)
            cat_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = cat_ckpt_dir / str(config.training.checkpoint_name)
            checkpoint_meta = _build_checkpoint_meta(
                saved=save_checkpoint(detector, ckpt_path),
                run_dir=run_path,
            )

        return WorkbenchTrainingResult(
            detector=detector,
            training_report=training_report,
            checkpoint_meta=checkpoint_meta,
        )

    detector.fit(train_inputs)
    return WorkbenchTrainingResult(
        detector=detector,
        training_report=None,
        checkpoint_meta=None,
    )


__all__ = ["WorkbenchTrainingResult", "run_workbench_training"]
