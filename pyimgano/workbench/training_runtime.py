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
    if config.training.validation_fraction is not None:
        fit_kwargs["validation_fraction"] = float(config.training.validation_fraction)
    if config.training.early_stopping_patience is not None:
        fit_kwargs["early_stopping_patience"] = int(config.training.early_stopping_patience)
    if config.training.early_stopping_min_delta is not None:
        fit_kwargs["early_stopping_min_delta"] = float(config.training.early_stopping_min_delta)
    if config.training.max_steps is not None:
        fit_kwargs["max_steps"] = int(config.training.max_steps)
    if config.training.max_train_samples is not None:
        fit_kwargs["max_train_samples"] = int(config.training.max_train_samples)
    if config.training.batch_size is not None:
        fit_kwargs["batch_size"] = int(config.training.batch_size)
    if config.training.num_workers is not None:
        fit_kwargs["num_workers"] = int(config.training.num_workers)
    if config.training.weight_decay is not None:
        fit_kwargs["weight_decay"] = float(config.training.weight_decay)
    if config.training.optimizer_name is not None:
        fit_kwargs["optimizer_name"] = str(config.training.optimizer_name)
    if config.training.optimizer_momentum is not None:
        fit_kwargs["optimizer_momentum"] = float(config.training.optimizer_momentum)
    if config.training.optimizer_nesterov is not None:
        fit_kwargs["optimizer_nesterov"] = bool(config.training.optimizer_nesterov)
    if config.training.optimizer_dampening is not None:
        fit_kwargs["optimizer_dampening"] = float(config.training.optimizer_dampening)
    if config.training.adam_beta1 is not None:
        fit_kwargs["adam_beta1"] = float(config.training.adam_beta1)
    if config.training.adam_beta2 is not None:
        fit_kwargs["adam_beta2"] = float(config.training.adam_beta2)
    if config.training.adam_amsgrad is not None:
        fit_kwargs["adam_amsgrad"] = bool(config.training.adam_amsgrad)
    if config.training.optimizer_eps is not None:
        fit_kwargs["optimizer_eps"] = float(config.training.optimizer_eps)
    if config.training.rmsprop_alpha is not None:
        fit_kwargs["rmsprop_alpha"] = float(config.training.rmsprop_alpha)
    if config.training.rmsprop_centered is not None:
        fit_kwargs["rmsprop_centered"] = bool(config.training.rmsprop_centered)
    if config.training.scheduler_name is not None:
        fit_kwargs["scheduler_name"] = str(config.training.scheduler_name)
    if config.training.scheduler_milestones is not None:
        fit_kwargs["scheduler_milestones"] = [
            int(v) for v in config.training.scheduler_milestones
        ]
    if config.training.scheduler_step_size is not None:
        fit_kwargs["scheduler_step_size"] = int(config.training.scheduler_step_size)
    if config.training.scheduler_gamma is not None:
        fit_kwargs["scheduler_gamma"] = float(config.training.scheduler_gamma)
    if config.training.scheduler_t_max is not None:
        fit_kwargs["scheduler_t_max"] = int(config.training.scheduler_t_max)
    if config.training.scheduler_eta_min is not None:
        fit_kwargs["scheduler_eta_min"] = float(config.training.scheduler_eta_min)
    if config.training.scheduler_patience is not None:
        fit_kwargs["scheduler_patience"] = int(config.training.scheduler_patience)
    if config.training.scheduler_factor is not None:
        fit_kwargs["scheduler_factor"] = float(config.training.scheduler_factor)
    if config.training.scheduler_min_lr is not None:
        fit_kwargs["scheduler_min_lr"] = float(config.training.scheduler_min_lr)
    if config.training.scheduler_cooldown is not None:
        fit_kwargs["scheduler_cooldown"] = int(config.training.scheduler_cooldown)
    if config.training.scheduler_threshold is not None:
        fit_kwargs["scheduler_threshold"] = float(config.training.scheduler_threshold)
    if config.training.scheduler_threshold_mode is not None:
        fit_kwargs["scheduler_threshold_mode"] = str(config.training.scheduler_threshold_mode)
    if config.training.scheduler_eps is not None:
        fit_kwargs["scheduler_eps"] = float(config.training.scheduler_eps)
    if config.training.criterion_name is not None:
        fit_kwargs["criterion_name"] = str(config.training.criterion_name)
    if config.training.shuffle_train is not None:
        fit_kwargs["shuffle_train"] = bool(config.training.shuffle_train)
    if config.training.drop_last is not None:
        fit_kwargs["drop_last"] = bool(config.training.drop_last)
    if config.training.pin_memory is not None:
        fit_kwargs["pin_memory"] = bool(config.training.pin_memory)
    if config.training.persistent_workers is not None:
        fit_kwargs["persistent_workers"] = bool(config.training.persistent_workers)
    if config.training.validation_split_seed is not None:
        fit_kwargs["validation_split_seed"] = int(config.training.validation_split_seed)
    if config.training.warmup_epochs is not None:
        fit_kwargs["warmup_epochs"] = int(config.training.warmup_epochs)
    if config.training.warmup_start_factor is not None:
        fit_kwargs["warmup_start_factor"] = float(config.training.warmup_start_factor)
    if config.training.ema_enabled is not None:
        fit_kwargs["ema_enabled"] = bool(config.training.ema_enabled)
    if config.training.ema_decay is not None:
        fit_kwargs["ema_decay"] = float(config.training.ema_decay)
    if config.training.ema_start_epoch is not None:
        fit_kwargs["ema_start_epoch"] = int(config.training.ema_start_epoch)
    return fit_kwargs


def restore_training_checkpoint_if_requested(
    *,
    detector: Any,
    config: WorkbenchConfig,
) -> dict[str, Any] | None:
    resume_path = getattr(config.training, "resume_from_checkpoint", None)
    if resume_path is None:
        return None

    from pyimgano.workbench.checkpoint_restore import load_checkpoint_into_detector

    checkpoint_path = Path(str(resume_path))
    load_checkpoint_into_detector(detector, checkpoint_path)
    return {
        "requested_path": str(resume_path),
        "loaded": True,
        "path": checkpoint_path.as_posix(),
    }


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

        checkpoint_restore = restore_training_checkpoint_if_requested(
            detector=detector,
            config=config,
        )
        training_report = micro_finetune(
            detector,
            train_inputs,
            seed=config.seed,
            fit_kwargs=_build_fit_kwargs(config),
        )
        if checkpoint_restore is not None:
            training_report = dict(training_report or {})
            training_report["checkpoint_restore"] = checkpoint_restore

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
