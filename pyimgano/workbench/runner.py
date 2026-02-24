from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pyimgano.evaluation import evaluate_detector
from pyimgano.inference.api import infer
from pyimgano.models.registry import create_model
from pyimgano.reporting.environment import collect_environment
from pyimgano.reporting.report import save_jsonl_records, save_run_report, stamp_report_payload
from pyimgano.reporting.runs import (
    build_workbench_run_dir_name,
    build_workbench_run_paths,
    ensure_run_dir,
)
from pyimgano.workbench.adaptation import apply_tiling, build_postprocess
from pyimgano.workbench.calibration import calibrate_detector_threshold
from pyimgano.workbench.config import WorkbenchConfig
from pyimgano.workbench.maps import save_anomaly_map_npy


def _load_split_paths(
    *,
    config: WorkbenchConfig,
    category: str,
    load_masks: bool,
) -> tuple[
    list[str],
    list[str],
    list[str],
    np.ndarray,
    np.ndarray | None,
    str | None,
    list[Mapping[str, Any] | None] | None,
]:
    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    dataset = str(config.dataset.name)
    if dataset.lower() == "manifest":
        if config.dataset.input_mode != "paths":
            raise ValueError("dataset.name='manifest' currently supports only dataset.input_mode='paths'.")
        if config.dataset.manifest_path is None:
            raise ValueError("dataset.manifest_path is required when dataset.name='manifest'.")

        from pyimgano.datasets.manifest import ManifestSplitPolicy, load_manifest_benchmark_split

        sp = config.dataset.split_policy
        seed = (
            int(sp.seed)
            if sp.seed is not None
            else (int(config.seed) if config.seed is not None else 0)
        )
        policy = ManifestSplitPolicy(
            mode=str(sp.mode),
            scope=str(sp.scope),
            seed=seed,
            test_normal_fraction=float(sp.test_normal_fraction),
        )
        split = load_manifest_benchmark_split(
            manifest_path=str(config.dataset.manifest_path),
            root_fallback=str(config.dataset.root),
            category=str(category),
            resize=tuple(config.dataset.resize),
            load_masks=bool(load_masks),
            split_policy=policy,
        )
        calibration = list(split.calibration_paths) if split.calibration_paths else list(split.train_paths)
        return (
            list(split.train_paths),
            calibration,
            list(split.test_paths),
            np.asarray(split.test_labels),
            split.test_masks,
            split.pixel_skip_reason,
            split.test_meta,
        )

    split = load_benchmark_split(
        dataset=dataset,  # type: ignore[arg-type]
        root=str(config.dataset.root),
        category=str(category),
        resize=tuple(config.dataset.resize),
        load_masks=bool(load_masks),
    )
    train_paths = list(split.train_paths)
    calibration_paths = list(train_paths)
    return (
        train_paths,
        calibration_paths,
        list(split.test_paths),
        np.asarray(split.test_labels),
        split.test_masks,
        None,
        None,
    )


def _load_split_numpy(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray | None]:
    from pyimgano.datasets import load_dataset

    ds = load_dataset(
        dataset,
        root,
        category=category,
        resize=tuple(resize),
        load_masks=True,
    )
    train_data = np.asarray(ds.get_train_data())
    test_data, test_labels, test_masks = ds.get_test_data()
    test_arr = np.asarray(test_data)

    train_inputs = [np.asarray(train_data[i]) for i in range(int(train_data.shape[0]))]
    test_inputs = [np.asarray(test_arr[i]) for i in range(int(test_arr.shape[0]))]
    return train_inputs, test_inputs, np.asarray(test_labels), test_masks


def _maybe_resize_maps_to_masks(maps: Sequence[np.ndarray], masks: np.ndarray) -> np.ndarray:
    import cv2

    if masks.ndim != 3:
        raise ValueError(f"Expected masks (N,H,W), got {masks.shape}")

    out: list[np.ndarray] = []
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got {arr.shape}")
        target_h, target_w = int(masks[i].shape[0]), int(masks[i].shape[1])
        if arr.shape != (target_h, target_w):
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        out.append(np.asarray(arr, dtype=np.float32))
    return np.stack(out, axis=0)


def _create_detector(config: WorkbenchConfig) -> Any:
    import pyimgano.models  # noqa: F401
    from pyimgano.cli import _resolve_preset_kwargs
    from pyimgano.cli_common import build_model_kwargs

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


def _run_category(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
    category: str,
    run_dir: Path | None,
) -> dict[str, Any]:
    if config.dataset.input_mode == "paths":
        train_inputs, calibration_inputs, test_inputs, test_labels, test_masks, pixel_skip_reason, test_meta = _load_split_paths(
            config=config,
            category=str(category),
            load_masks=True,
        )
        input_format = None
    elif config.dataset.input_mode == "numpy":
        if str(config.dataset.name).lower() == "manifest":
            raise ValueError(
                "dataset.name='manifest' currently supports only dataset.input_mode='paths'."
            )
        train_inputs, test_inputs, test_labels, test_masks = _load_split_numpy(
            dataset=str(config.dataset.name),
            root=str(config.dataset.root),
            category=str(category),
            resize=tuple(config.dataset.resize),
        )
        calibration_inputs = list(train_inputs)
        pixel_skip_reason = None
        test_meta = None
        input_format = "rgb_u8_hwc"
    else:
        raise ValueError(
            f"Unknown input_mode: {config.dataset.input_mode!r}. Choose from: paths, numpy."
        )

    if config.dataset.limit_train is not None:
        train_inputs = list(train_inputs)[: int(config.dataset.limit_train)]
        calibration_inputs = list(calibration_inputs)[: int(config.dataset.limit_train)]
    if config.dataset.limit_test is not None:
        test_inputs = list(test_inputs)[: int(config.dataset.limit_test)]
        test_labels = np.asarray(test_labels)[: int(config.dataset.limit_test)]
        if test_masks is not None:
            test_masks = np.asarray(test_masks)[: int(config.dataset.limit_test)]

    detector = _create_detector(config)
    detector = apply_tiling(detector, config.adaptation.tiling)

    training_report = None
    checkpoint_meta = None
    if bool(getattr(config, "training", None) and config.training.enabled):
        from pyimgano.training.checkpointing import save_checkpoint
        from pyimgano.training.runner import micro_finetune

        fit_kwargs: dict[str, Any] = {}
        if config.training.epochs is not None:
            fit_kwargs["epochs"] = int(config.training.epochs)
        if config.training.lr is not None:
            fit_kwargs["lr"] = float(config.training.lr)

        training_report = micro_finetune(
            detector,
            train_inputs,
            seed=config.seed,
            fit_kwargs=fit_kwargs,
        )

        if run_dir is not None and bool(config.output.save_run):
            cat_ckpt_dir = build_workbench_run_paths(run_dir).checkpoints_dir / str(category)
            cat_ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = cat_ckpt_dir / str(config.training.checkpoint_name)
            saved = save_checkpoint(detector, ckpt_path)
            try:
                rel = saved.relative_to(run_dir)
                checkpoint_meta = {"path": str(rel)}
            except Exception:
                checkpoint_meta = {"path": str(saved)}
    else:
        detector.fit(train_inputs)
    threshold = calibrate_detector_threshold(detector, calibration_inputs, input_format=input_format)

    postprocess = build_postprocess(config.adaptation.postprocess)
    include_maps = bool(config.adaptation.save_maps or (postprocess is not None))

    results = infer(
        detector,
        test_inputs,
        input_format=input_format,
        include_maps=include_maps,
        postprocess=postprocess,
    )

    scores = np.asarray([r.score for r in results], dtype=np.float32)
    maps_list = [r.anomaly_map for r in results] if include_maps else None

    pixel_scores = None
    if test_masks is not None and maps_list is not None and all(m is not None for m in maps_list):
        maps_arr = [np.asarray(m, dtype=np.float32) for m in maps_list if m is not None]
        pixel_scores = _maybe_resize_maps_to_masks(maps_arr, np.asarray(test_masks))

    eval_results = evaluate_detector(
        np.asarray(test_labels),
        scores,
        threshold=float(threshold),
        find_best_threshold=False,
        pixel_labels=test_masks,
        pixel_scores=pixel_scores,
    )

    threshold_used = float(eval_results["threshold"])

    payload: dict[str, Any] = {
        "dataset": str(config.dataset.name),
        "category": str(category),
        "model": str(config.model.name),
        "recipe": str(recipe_name),
        "seed": (int(config.seed) if config.seed is not None else None),
        "input_mode": str(config.dataset.input_mode),
        "device": str(config.model.device),
        "preset": config.model.preset,
        "resize": [int(config.dataset.resize[0]), int(config.dataset.resize[1])],
        "threshold": threshold_used,
        "results": eval_results,
    }
    if pixel_skip_reason is not None:
        payload["pixel_metrics_status"] = {
            "enabled": False,
            "reason": str(pixel_skip_reason),
        }
    if training_report is not None:
        payload["training"] = training_report
    if checkpoint_meta is not None:
        payload["checkpoint"] = checkpoint_meta
    payload = stamp_report_payload(payload)

    if run_dir is not None and bool(config.output.save_run):
        paths = build_workbench_run_paths(run_dir)
        cat_dir = paths.categories_dir / str(category)
        cat_dir.mkdir(parents=True, exist_ok=True)

        save_run_report(cat_dir / "report.json", payload)

        map_paths: list[str | None] | None = None
        if bool(config.adaptation.save_maps) and maps_list is not None:
            map_paths = []
            for i, (item, m) in enumerate(zip(test_inputs, maps_list)):
                if m is None:
                    map_paths.append(None)
                    continue
                in_path = str(item) if isinstance(item, (str, Path)) else f"numpy[{i}]"
                saved = save_anomaly_map_npy(
                    paths.artifacts_dir,
                    index=int(i),
                    input_path=in_path,
                    anomaly_map=np.asarray(m, dtype=np.float32),
                )
                try:
                    rel = saved.relative_to(paths.run_dir)
                    map_paths.append(str(rel))
                except Exception:
                    map_paths.append(str(saved))

        records: list[dict[str, Any]] = []
        y_true = np.asarray(test_labels).astype(int).tolist()
        pred = (scores >= float(threshold_used)).astype(int).tolist()
        for i, item in enumerate(test_inputs):
            input_value = str(item) if isinstance(item, (str, Path)) else f"numpy[{i}]"
            rec: dict[str, Any] = {
                "index": int(i),
                "dataset": str(config.dataset.name),
                "category": str(category),
                "input": input_value,
                "y_true": int(y_true[i]),
                "score": float(scores[i]),
                "threshold": float(threshold_used),
                "pred": int(pred[i]),
            }
            if test_meta is not None:
                meta = test_meta[i]
                if meta is not None:
                    rec["meta"] = dict(meta)
            if map_paths is not None and map_paths[i] is not None and maps_list is not None:
                m = maps_list[i]
                if m is not None:
                    arr = np.asarray(m)
                    rec["anomaly_map"] = {
                        "path": str(map_paths[i]),
                        "shape": [int(d) for d in arr.shape],
                        "dtype": str(arr.dtype),
                    }
            records.append(rec)

        if bool(config.output.per_image_jsonl):
            save_jsonl_records(cat_dir / "per_image.jsonl", records)

    return payload


def run_workbench(
    *,
    config: WorkbenchConfig,
    recipe_name: str,
) -> dict[str, Any]:
    if bool(config.adaptation.save_maps) and not bool(config.output.save_run):
        raise ValueError("adaptation.save_maps requires output.save_run=true.")
    if bool(config.training.enabled) and not bool(config.output.save_run):
        raise ValueError("training.enabled requires output.save_run=true.")

    run_dir = None
    paths = None
    if bool(config.output.save_run):
        category_for_name = None if str(config.dataset.category).lower() == "all" else str(config.dataset.category)
        name = build_workbench_run_dir_name(
            dataset=str(config.dataset.name),
            recipe=str(recipe_name),
            model=str(config.model.name),
            category=category_for_name,
        )
        run_dir = ensure_run_dir(output_dir=config.output.output_dir, name=name)
        paths = build_workbench_run_paths(run_dir)
        paths.categories_dir.mkdir(parents=True, exist_ok=True)
        paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

        save_run_report(paths.environment_json, collect_environment())
        save_run_report(paths.config_json, {"config": asdict(config)})

    dataset = str(config.dataset.name)
    root = str(config.dataset.root)
    category = str(config.dataset.category)

    if category.lower() != "all":
        payload = _run_category(
            config=config,
            recipe_name=recipe_name,
            category=category,
            run_dir=run_dir,
        )
        if run_dir is not None and paths is not None:
            payload["run_dir"] = str(paths.run_dir)
            save_run_report(paths.report_json, payload)

        return payload

    # All categories: aggregate per-category runs into a single report.
    if dataset.lower() == "manifest":
        if config.dataset.manifest_path is None:
            raise ValueError("dataset.manifest_path is required when dataset.name='manifest'.")
        from pyimgano.datasets.manifest import list_manifest_categories

        categories = list_manifest_categories(config.dataset.manifest_path)
    else:
        from pyimgano.datasets.catalog import list_dataset_categories

        categories = list_dataset_categories(dataset=dataset, root=root)

    per_category: dict[str, Any] = {}
    for cat in categories:
        per_category[str(cat)] = _run_category(
            config=config,
            recipe_name=recipe_name,
            category=str(cat),
            run_dir=run_dir,
        )

    def _safe_float(value: Any) -> float | None:
        try:
            v = float(value)
        except Exception:
            return None
        if not np.isfinite(v):
            return None
        return v

    metrics_to_average = ["auroc", "average_precision"]
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in metrics_to_average:
        vals: list[float] = []
        for cat in categories:
            res = per_category[str(cat)].get("results", {})
            v = _safe_float(res.get(key, None))
            if v is not None:
                vals.append(v)
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            means[key] = float(np.mean(arr))
            stds[key] = float(np.std(arr))

    payload = {
        "dataset": dataset,
        "category": "all",
        "model": str(config.model.name),
        "recipe": str(recipe_name),
        "seed": (int(config.seed) if config.seed is not None else None),
        "input_mode": str(config.dataset.input_mode),
        "device": str(config.model.device),
        "preset": config.model.preset,
        "resize": [int(config.dataset.resize[0]), int(config.dataset.resize[1])],
        "categories": categories,
        "mean_metrics": means,
        "std_metrics": stds,
        "per_category": per_category,
    }
    payload = stamp_report_payload(payload)

    if run_dir is not None and paths is not None:
        payload["run_dir"] = str(paths.run_dir)
        save_run_report(paths.report_json, payload)

    return payload


def build_infer_config_payload(*, config: WorkbenchConfig, report: Mapping[str, Any]) -> dict[str, Any]:
    """Build a minimal, JSON-friendly payload describing how to run inference.

    Intended for `pyimgano-train --export-infer-config`.
    """

    model_payload: dict[str, Any] = {
        "name": str(config.model.name),
        "preset": config.model.preset,
        "device": str(config.model.device),
        "pretrained": bool(config.model.pretrained),
        "contamination": float(config.model.contamination),
        "model_kwargs": dict(config.model.model_kwargs),
        "checkpoint_path": (str(config.model.checkpoint_path) if config.model.checkpoint_path is not None else None),
    }

    adaptation_payload: dict[str, Any] = {
        "tiling": {
            "tile_size": config.adaptation.tiling.tile_size,
            "stride": config.adaptation.tiling.stride,
            "score_reduce": config.adaptation.tiling.score_reduce,
            "score_topk": float(config.adaptation.tiling.score_topk),
            "map_reduce": config.adaptation.tiling.map_reduce,
        },
        "postprocess": (config.adaptation.postprocess.__dict__ if config.adaptation.postprocess is not None else None),
        "save_maps": bool(config.adaptation.save_maps),
    }

    out: dict[str, Any] = {
        "model": model_payload,
        "adaptation": adaptation_payload,
    }

    # Prefer the report's run_dir (if present) to keep the payload portable across
    # output_dir overrides.
    run_dir = report.get("run_dir", None)
    if run_dir is not None:
        out["from_run"] = str(run_dir)

    # Preserve threshold/checkpoint when present (single-category or category-selected exports).
    if "threshold" in report:
        out["threshold"] = report.get("threshold")
    if "checkpoint" in report:
        out["checkpoint"] = report.get("checkpoint")
    if "category" in report:
        out["category"] = report.get("category")
    if "per_category" in report:
        out["per_category"] = report.get("per_category")

    return stamp_report_payload(out)
