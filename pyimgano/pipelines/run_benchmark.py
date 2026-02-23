from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from pyimgano.evaluation import evaluate_detector
from pyimgano.models.registry import MODEL_REGISTRY, create_model
from pyimgano.reporting.report import save_jsonl_records, save_run_report, stamp_report_payload
from pyimgano.reporting.runs import build_run_dir_name, build_run_paths, ensure_run_dir

from .mvtec_visa import load_benchmark_split


ScoreThresholdStrategy = Literal["train_quantile", "test_optimal_f1", "median"]


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    root: str
    category: str
    model: str
    input_mode: str = "paths"
    seed: int | None = None
    device: str = "cpu"
    preset: str | None = None
    pretrained: bool = True
    contamination: float = 0.1
    resize: tuple[int, int] = (256, 256)
    model_kwargs: dict[str, Any] | None = None
    cache_dir: str | None = None
    load_detector_path: str | None = None
    save_detector_path: str | None = None
    score_threshold_strategy: ScoreThresholdStrategy = "train_quantile"
    calibration_quantile: float | None = None
    limit_train: int | None = None
    limit_test: int | None = None


def list_dataset_categories(*, dataset: str, root: str) -> list[str]:
    """Compatibility wrapper for category discovery."""

    from pyimgano.datasets.catalog import list_dataset_categories as _list

    return _list(dataset=dataset, root=root)


def _default_calibration_quantile(detector: Any, *, fallback: float = 0.995) -> float:
    contamination = getattr(detector, "contamination", None)
    try:
        if contamination is not None:
            cf = float(contamination)
            if 0.0 < cf < 0.5:
                return 1.0 - cf
    except Exception:
        pass
    return float(fallback)


def _seed_everything(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch
    except Exception:
        return

    try:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        return


def _calibrate_score_threshold(
    detector: Any,
    train_inputs: Sequence[str],
    *,
    strategy: ScoreThresholdStrategy,
    calibration_quantile: float | None,
) -> float | None:
    if strategy == "test_optimal_f1":
        return None
    if strategy == "median":
        scores = np.asarray(detector.decision_function(list(train_inputs)), dtype=np.float64)
        if scores.size == 0:
            raise ValueError("Unable to calibrate threshold: empty train score set.")
        return float(np.median(scores))

    if strategy != "train_quantile":
        raise ValueError(
            f"Unknown score_threshold_strategy: {strategy!r}. "
            "Choose from: train_quantile, test_optimal_f1, median."
        )

    q = (
        float(calibration_quantile)
        if calibration_quantile is not None
        else _default_calibration_quantile(detector)
    )
    if not 0.0 < q < 1.0:
        raise ValueError(f"calibration_quantile must be in (0,1), got {q}")

    scores = np.asarray(detector.decision_function(list(train_inputs)), dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Unable to calibrate threshold: empty train score set.")
    return float(np.quantile(scores, q))


def _merge_and_filter_model_kwargs(
    model_name: str,
    *,
    model_kwargs: dict[str, Any],
    auto_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Merge caller-provided kwargs with auto defaults and filter to accepted kwargs.

    This keeps the pipeline runner robust across heterogeneous model constructors.
    """

    try:
        entry = MODEL_REGISTRY.info(model_name)
    except Exception as exc:
        raise ValueError(f"Unknown model: {model_name!r}") from exc

    import inspect

    sig = inspect.signature(entry.constructor)
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    merged = dict(model_kwargs)
    for key, value in auto_defaults.items():
        merged.setdefault(key, value)

    if accepts_var_kwargs:
        return merged

    return {k: v for k, v in merged.items() if k in accepted}


def run_benchmark_category(
    *,
    config: RunConfig,
    save_run: bool = True,
    per_image_jsonl: bool = True,
    output_dir: str | Path | None = None,
    write_top_level: bool = True,
) -> dict[str, Any]:
    """Run a single category benchmark and optionally write artifacts."""

    import time

    total_start = time.perf_counter()
    timing: dict[str, float] = {
        "load_data_s": 0.0,
        "create_model_s": 0.0,
        "load_detector_s": 0.0,
        "fit_s": 0.0,
        "score_test_s": 0.0,
        "calibrate_s": 0.0,
        "evaluate_s": 0.0,
        "total_s": 0.0,
    }

    if config.seed is not None:
        _seed_everything(int(config.seed))

    save_detector_requested = config.save_detector_path is not None
    save_detector_auto = str(config.save_detector_path).lower() == "auto"

    load_detector_requested = config.load_detector_path is not None

    run_dir = None
    paths = None
    if bool(save_run) or (save_detector_requested and save_detector_auto):
        name = build_run_dir_name(dataset=config.dataset, model=config.model)
        run_dir = ensure_run_dir(output_dir=output_dir, name=name)
        paths = build_run_paths(run_dir)
        paths.categories_dir.mkdir(parents=True, exist_ok=True)

        if bool(save_run) and bool(write_top_level):
            from pyimgano.reporting.environment import collect_environment

            save_run_report(paths.run_dir / "environment.json", collect_environment())

    load_start = time.perf_counter()
    if config.input_mode == "paths":
        split = load_benchmark_split(
            dataset=config.dataset,  # type: ignore[arg-type]
            root=config.root,
            category=config.category,
            resize=tuple(config.resize),
            load_masks=True,
        )
        train_inputs: list[Any] = list(split.train_paths)
        test_inputs: list[Any] = list(split.test_paths)
        test_labels = np.asarray(split.test_labels)
        test_masks = split.test_masks
    elif config.input_mode == "numpy":
        from pyimgano.datasets import load_dataset

        ds = load_dataset(
            config.dataset,
            config.root,
            category=config.category,
            resize=tuple(config.resize),
            load_masks=True,
        )
        train_data = np.asarray(ds.get_train_data())
        test_data, test_labels, test_masks = ds.get_test_data()
        test_data_arr = np.asarray(test_data)
        train_inputs = [train_data[i] for i in range(int(train_data.shape[0]))]
        test_inputs = [
            test_data_arr[i] for i in range(int(test_data_arr.shape[0]))
        ]
    else:
        raise ValueError(
            f"Unknown input_mode: {config.input_mode!r}. Choose from: paths, numpy."
        )
    timing["load_data_s"] = float(time.perf_counter() - load_start)

    if config.limit_train is not None:
        train_inputs = train_inputs[: int(config.limit_train)]
    if config.limit_test is not None:
        limit = int(config.limit_test)
        test_inputs = test_inputs[:limit]
        test_labels = np.asarray(test_labels)[:limit]
        if test_masks is not None:
            test_masks = np.asarray(test_masks)[:limit]

    auto_defaults: dict[str, Any] = {
        "device": config.device,
        "contamination": float(config.contamination),
        "pretrained": bool(config.pretrained),
    }
    if config.seed is not None:
        auto_defaults["random_seed"] = int(config.seed)
        auto_defaults["random_state"] = int(config.seed)

    if load_detector_requested:
        load_detector_start = time.perf_counter()
        from pyimgano.serialization import load_detector

        detector = load_detector(str(config.load_detector_path))
        timing["load_detector_s"] = float(time.perf_counter() - load_detector_start)
    else:
        create_model_start = time.perf_counter()
        detector = create_model(
            config.model,
            **_merge_and_filter_model_kwargs(
                config.model,
                model_kwargs=dict(config.model_kwargs or {}),
                auto_defaults=auto_defaults,
            ),
        )
        timing["create_model_s"] = float(time.perf_counter() - create_model_start)

        # Fit and score.
        fit_start = time.perf_counter()
        detector.fit(train_inputs)
        timing["fit_s"] = float(time.perf_counter() - fit_start)

    if save_detector_requested:
        from pyimgano.serialization import save_detector

        if save_detector_auto:
            if paths is None:
                raise ValueError("save_detector_path=auto requires an output directory.")
            detector_path = paths.run_dir / "detector.pkl"
        else:
            detector_path = Path(str(config.save_detector_path))

        save_detector(detector_path, detector)

    if config.input_mode == "paths" and config.cache_dir is not None:
        if hasattr(detector, "set_feature_cache"):
            detector.set_feature_cache(config.cache_dir)

    score_start = time.perf_counter()
    scores = np.asarray(detector.decision_function(test_inputs), dtype=np.float64)
    timing["score_test_s"] = float(time.perf_counter() - score_start)

    calibrate_start = time.perf_counter()
    calibrated_threshold = _calibrate_score_threshold(
        detector,
        train_inputs,
        strategy=config.score_threshold_strategy,
        calibration_quantile=config.calibration_quantile,
    )
    timing["calibrate_s"] = float(time.perf_counter() - calibrate_start)

    eval_start = time.perf_counter()
    if calibrated_threshold is None:
        results = evaluate_detector(
            test_labels,
            scores,
            threshold=None,
            find_best_threshold=True,
            pixel_labels=test_masks,
            pixel_scores=None,
        )
    else:
        results = evaluate_detector(
            test_labels,
            scores,
            threshold=float(calibrated_threshold),
            find_best_threshold=False,
            pixel_labels=test_masks,
            pixel_scores=None,
        )
    timing["evaluate_s"] = float(time.perf_counter() - eval_start)

    threshold_used = float(results["threshold"])

    # Optional: compute pixel metrics using the existing helper.
    # We call into `evaluate_split` to preserve alignment / resizing behavior.
    # It will re-fit the detector, but only when pixel metrics are requested in CLI.
    timing["total_s"] = float(time.perf_counter() - total_start)

    payload: dict[str, Any] = {
        "dataset": config.dataset,
        "category": config.category,
        "model": config.model,
        "input_mode": config.input_mode,
        "device": config.device,
        "preset": config.preset,
        "resize": list(config.resize),
        "score_threshold_strategy": config.score_threshold_strategy,
        "calibration_quantile": config.calibration_quantile,
        "threshold": threshold_used,
        "calibrated_threshold": calibrated_threshold,
        "results": results,
        "timing": dict(timing),
    }
    if load_detector_requested:
        payload["loaded_detector_path"] = str(config.load_detector_path)
    if save_detector_requested:
        payload["detector_path"] = str(detector_path)
    payload = stamp_report_payload(payload)

    if save_run:
        if paths is None:
            raise RuntimeError("internal error: expected run paths when save_run=True")

        cat_dir = paths.categories_dir / str(config.category)
        cat_dir.mkdir(parents=True, exist_ok=True)

        save_run_report(cat_dir / "report.json", payload)

        if per_image_jsonl:
            y_true = np.asarray(test_labels).astype(int).tolist()
            pred = (scores >= float(threshold_used)).astype(int).tolist()

            records: list[dict[str, Any]] = []
            for i, item in enumerate(test_inputs):
                if isinstance(item, (str, Path)):
                    input_value = str(item)
                else:
                    input_value = f"numpy[{i}]"
                rec = {
                    "index": int(i),
                    "dataset": str(config.dataset),
                    "category": str(config.category),
                    "input": input_value,
                    "y_true": int(y_true[i]),
                    "score": float(scores[i]),
                    "threshold": float(threshold_used),
                    "pred": int(pred[i]),
                }
                records.append(rec)

            save_jsonl_records(cat_dir / "per_image.jsonl", records)

        if write_top_level:
            save_run_report(paths.config_json, {"config": dict(config.__dict__)})
            save_run_report(paths.report_json, payload)
            payload["run_dir"] = str(run_dir)

    return payload


def run_benchmark(
    *,
    dataset: str,
    root: str,
    category: str,
    model: str,
    input_mode: str = "paths",
    seed: int | None = None,
    device: str = "cpu",
    preset: str | None = None,
    pretrained: bool = True,
    contamination: float = 0.1,
    resize: tuple[int, int] = (256, 256),
    model_kwargs: dict[str, Any] | None = None,
    score_threshold_strategy: ScoreThresholdStrategy = "train_quantile",
    calibration_quantile: float | None = None,
    limit_train: int | None = None,
    limit_test: int | None = None,
    save_run: bool = True,
    per_image_jsonl: bool = True,
    cache_dir: str | Path | None = None,
    load_detector_path: str | Path | None = None,
    save_detector_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a benchmark for a single category or for all categories."""

    if load_detector_path is not None and str(save_detector_path).lower() == "auto":
        raise ValueError("--save-detector auto conflicts with --load-detector.")

    if str(category).lower() == "all" and save_detector_path is not None:
        raise ValueError("--save-detector is not supported with --category all.")
    if str(category).lower() == "all" and load_detector_path is not None:
        raise ValueError("--load-detector is not supported with --category all.")

    if str(category).lower() != "all":
        cfg = RunConfig(
            dataset=str(dataset),
            root=str(root),
            category=str(category),
            model=str(model),
            input_mode=str(input_mode),
            seed=(int(seed) if seed is not None else None),
            device=str(device),
            preset=(str(preset) if preset is not None else None),
            pretrained=bool(pretrained),
            contamination=float(contamination),
            resize=(int(resize[0]), int(resize[1])),
            model_kwargs=dict(model_kwargs or {}),
            cache_dir=(str(cache_dir) if cache_dir is not None else None),
            load_detector_path=(str(load_detector_path) if load_detector_path is not None else None),
            save_detector_path=(str(save_detector_path) if save_detector_path is not None else None),
            score_threshold_strategy=score_threshold_strategy,
            calibration_quantile=calibration_quantile,
            limit_train=limit_train,
            limit_test=limit_test,
        )
        return run_benchmark_category(
            config=cfg,
            save_run=save_run,
            per_image_jsonl=per_image_jsonl,
            output_dir=output_dir,
        )

    categories = list_dataset_categories(dataset=dataset, root=root)
    per_category: dict[str, Any] = {}

    name = build_run_dir_name(dataset=str(dataset), model=str(model))
    run_dir = ensure_run_dir(output_dir=output_dir, name=name) if save_run else None
    paths = build_run_paths(run_dir) if run_dir is not None else None
    if paths is not None:
        paths.categories_dir.mkdir(parents=True, exist_ok=True)
        from pyimgano.reporting.environment import collect_environment

        save_run_report(paths.run_dir / "environment.json", collect_environment())

    for cat in categories:
        cfg = RunConfig(
            dataset=str(dataset),
            root=str(root),
            category=str(cat),
            model=str(model),
            input_mode=str(input_mode),
            seed=(int(seed) if seed is not None else None),
            device=str(device),
            preset=(str(preset) if preset is not None else None),
            pretrained=bool(pretrained),
            contamination=float(contamination),
            resize=(int(resize[0]), int(resize[1])),
            model_kwargs=dict(model_kwargs or {}),
            cache_dir=(str(cache_dir) if cache_dir is not None else None),
            score_threshold_strategy=score_threshold_strategy,
            calibration_quantile=calibration_quantile,
            limit_train=limit_train,
            limit_test=limit_test,
        )

        cat_output_dir = None
        if paths is not None:
            cat_output_dir = paths.run_dir  # run_benchmark_category writes under categories/<cat>/

        result = run_benchmark_category(
            config=cfg,
            save_run=bool(paths is not None),
            per_image_jsonl=per_image_jsonl,
            output_dir=cat_output_dir,
            write_top_level=False,
        )
        per_category[str(cat)] = result

    # Aggregate means for common metrics.
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
        "dataset": str(dataset),
        "category": "all",
        "model": str(model),
        "input_mode": str(input_mode),
        "device": str(device),
        "preset": (str(preset) if preset is not None else None),
        "resize": [int(resize[0]), int(resize[1])],
        "score_threshold_strategy": score_threshold_strategy,
        "calibration_quantile": calibration_quantile,
        "categories": categories,
        "mean_metrics": means,
        "std_metrics": stds,
        "per_category": per_category,
    }
    payload = stamp_report_payload(payload)

    if paths is not None:
        save_run_report(paths.report_json, payload)
        save_run_report(
            paths.config_json,
            {
                "config": {
                    "dataset": str(dataset),
                    "root": str(root),
                    "category": "all",
                    "model": str(model),
                    "input_mode": str(input_mode),
                    "seed": (int(seed) if seed is not None else None),
                    "device": str(device),
                    "preset": (str(preset) if preset is not None else None),
                    "pretrained": bool(pretrained),
                    "contamination": float(contamination),
                    "resize": [int(resize[0]), int(resize[1])],
                    "model_kwargs": dict(model_kwargs or {}),
                    "score_threshold_strategy": score_threshold_strategy,
                    "calibration_quantile": calibration_quantile,
                    "limit_train": limit_train,
                    "limit_test": limit_test,
                }
            },
        )
        payload["run_dir"] = str(paths.run_dir)

    return payload
