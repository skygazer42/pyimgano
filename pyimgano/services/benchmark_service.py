from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

import pyimgano.services.dataset_split_service as dataset_split_service
from pyimgano.models.registry import create_model
from pyimgano.services.model_options import (
    enforce_checkpoint_requirement,
    resolve_model_options,
    resolve_requested_model,
)


@dataclass(frozen=True)
class PixelPostprocessConfig:
    normalize_method: str = "minmax"
    percentile_range: tuple[float, float] = (1.0, 99.0)
    gaussian_sigma: float = 0.0
    morph_open_ksize: int = 0
    morph_close_ksize: int = 0
    component_threshold: float | None = None
    min_component_area: int = 0


@dataclass(frozen=True)
class BenchmarkRunRequest:
    dataset: str
    root: str
    category: str
    model: str
    manifest_path: str | None = None
    input_mode: str = "paths"
    seed: int | None = None
    device: str = "cpu"
    preset: str | None = None
    pretrained: bool = False
    contamination: float = 0.1
    resize: tuple[int, int] = (256, 256)
    model_kwargs: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    calibration_quantile: float | None = None
    limit_train: int | None = None
    limit_test: int | None = None
    manifest_split_seed: int | None = None
    manifest_test_normal_fraction: float = 0.2
    save_run: bool = True
    per_image_jsonl: bool = True
    cache_dir: str | None = None
    load_detector_path: str | None = None
    save_detector_path: str | None = None
    output_dir: str | None = None
    pixel: bool = False
    pixel_segf1: bool = False
    pixel_threshold_strategy: str | None = None
    pixel_normal_quantile: float = 0.999
    pixel_calibration_fraction: float = 0.2
    pixel_calibration_seed: int = 0
    pixel_postprocess: PixelPostprocessConfig | None = None
    pixel_aupro_limit: float = 0.3
    pixel_aupro_thresholds: int = 200
    default_knn_backend: Callable[[], str] | None = None


@dataclass(frozen=True)
class SuiteRunRequest:
    suite: str
    dataset: str
    root: str
    category: str
    manifest_path: str | None = None
    input_mode: str = "paths"
    seed: int | None = None
    device: str = "cpu"
    pretrained: bool = False
    contamination: float = 0.1
    resize: tuple[int, int] = (256, 256)
    calibration_quantile: float | None = None
    limit_train: int | None = None
    limit_test: int | None = None
    manifest_split_seed: int | None = None
    manifest_test_normal_fraction: float = 0.2
    pixel: bool = False
    pixel_segf1: bool = False
    pixel_threshold_strategy: str | None = None
    pixel_normal_quantile: float = 0.999
    pixel_calibration_fraction: float = 0.2
    pixel_calibration_seed: int = 0
    pixel_postprocess: PixelPostprocessConfig | None = None
    pixel_aupro_limit: float = 0.3
    pixel_aupro_thresholds: int = 200
    save_run: bool = True
    per_image_jsonl: bool = True
    cache_dir: str | None = None
    output_dir: str | None = None
    max_models: int | None = None
    include_baselines: Sequence[str] | None = None
    exclude_baselines: Sequence[str] | None = None
    continue_on_error: bool = True
    sweep: str | None = None
    sweep_max_variants: int | None = None


def _run_benchmark_pipeline(**kwargs: Any) -> dict[str, Any]:
    from pyimgano.pipelines.run_benchmark import run_benchmark

    return run_benchmark(**kwargs)


def _run_suite_pipeline(**kwargs: Any) -> dict[str, Any]:
    from pyimgano.pipelines.run_suite import run_baseline_suite

    return run_baseline_suite(**kwargs)


def _normalize_name_filters(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None

    normalized: list[str] = []
    for item in values:
        for name in str(item).split(","):
            stripped = name.strip()
            if stripped:
                normalized.append(stripped)

    return normalized or None


def build_pixel_postprocess(
    config: PixelPostprocessConfig | None,
):
    if config is None:
        return None

    from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

    return AnomalyMapPostprocess(
        normalize=True,
        normalize_method=str(config.normalize_method),
        percentile_range=(
            float(config.percentile_range[0]),
            float(config.percentile_range[1]),
        ),
        gaussian_sigma=float(config.gaussian_sigma),
        morph_open_ksize=int(config.morph_open_ksize),
        morph_close_ksize=int(config.morph_close_ksize),
        component_threshold=config.component_threshold,
        min_component_area=int(config.min_component_area),
    )


def _resolve_model_run_options(
    request: BenchmarkRunRequest,
) -> tuple[str, dict[str, Any], Any]:
    if request.cache_dir is not None and str(request.input_mode) != "paths":
        raise ValueError("--cache-dir requires --input-mode paths.")

    model_name, preset_model_auto_kwargs, entry = resolve_requested_model(str(request.model))

    auto_kwargs: dict[str, Any] = dict(preset_model_auto_kwargs)
    auto_kwargs.update(
        {
            "device": str(request.device),
            "contamination": float(request.contamination),
            "pretrained": bool(request.pretrained),
        }
    )
    if request.seed is not None:
        auto_kwargs["random_seed"] = int(request.seed)
        auto_kwargs["random_state"] = int(request.seed)

    model_kwargs = resolve_model_options(
        model_name=model_name,
        preset=(str(request.preset) if request.preset is not None else None),
        user_kwargs=dict(request.model_kwargs or {}),
        auto_kwargs=auto_kwargs,
        checkpoint_path=(
            str(request.checkpoint_path) if request.checkpoint_path is not None else None
        ),
        default_knn_backend=request.default_knn_backend,
    )

    enforce_checkpoint_requirement(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

    return model_name, model_kwargs, entry


def _evaluate_pixel_split(
    detector: Any,
    split: Any,
    request: BenchmarkRunRequest,
) -> dict[str, Any]:
    from pyimgano.pipelines.mvtec_visa import evaluate_split

    return evaluate_split(
        detector,
        split,
        compute_pixel_scores=True,
        postprocess=build_pixel_postprocess(request.pixel_postprocess),
        pro_integration_limit=float(request.pixel_aupro_limit),
        pro_num_thresholds=int(request.pixel_aupro_thresholds),
        pixel_segf1=bool(request.pixel_segf1),
        pixel_threshold_strategy=request.pixel_threshold_strategy,
        pixel_normal_quantile=float(request.pixel_normal_quantile),
        calibration_fraction=float(request.pixel_calibration_fraction),
        calibration_seed=int(request.pixel_calibration_seed),
        score_calibration_quantile=(
            float(request.calibration_quantile)
            if request.calibration_quantile is not None
            else None
        ),
    )


def _run_pixel_benchmark_request(request: BenchmarkRunRequest) -> dict[str, Any]:
    if request.save_detector_path is not None:
        raise ValueError("--save-detector is only supported without --pixel.")
    if request.load_detector_path is not None:
        raise ValueError("--load-detector is only supported without --pixel.")
    if request.cache_dir is not None:
        raise ValueError("--cache-dir is only supported without --pixel.")
    if str(request.input_mode) != "paths":
        raise ValueError("--input-mode currently supports only 'paths' when using --pixel.")
    if str(request.category).lower() == "all":
        raise ValueError("--category all is not yet supported with --pixel.")

    model_name, model_kwargs, _entry = _resolve_model_run_options(request)
    detector = create_model(model_name, **model_kwargs)

    loaded_split = dataset_split_service.load_benchmark_style_split(
        dataset=str(request.dataset),
        root=str(request.root),
        manifest_path=(str(request.manifest_path) if request.manifest_path is not None else None),
        category=str(request.category),
        resize=(int(request.resize[0]), int(request.resize[1])),
        load_masks=True,
        seed=(int(request.seed) if request.seed is not None else None),
        manifest_split_seed=(
            int(request.manifest_split_seed) if request.manifest_split_seed is not None else None
        ),
        manifest_test_normal_fraction=float(request.manifest_test_normal_fraction),
    )
    split = loaded_split.split
    pixel_skip_reason = loaded_split.pixel_skip_reason

    from pyimgano.pipelines.mvtec_visa import BenchmarkSplit

    train_paths = list(split.train_paths)
    test_paths = list(split.test_paths)
    test_labels = np.asarray(split.test_labels)
    test_masks = None if split.test_masks is None else np.asarray(split.test_masks)
    if request.limit_train is not None:
        train_paths = train_paths[: int(request.limit_train)]
    if request.limit_test is not None:
        limit = int(request.limit_test)
        test_paths = test_paths[:limit]
        test_labels = test_labels[:limit]
        if test_masks is not None:
            test_masks = test_masks[:limit]
    split = BenchmarkSplit(
        train_paths=train_paths,
        test_paths=test_paths,
        test_labels=test_labels,
        test_masks=test_masks,
    )

    results = _evaluate_pixel_split(detector, split, request)
    payload: dict[str, Any] = {
        "dataset": str(request.dataset),
        "category": str(request.category),
        "model": str(request.model),
        "preset": (str(request.preset) if request.preset is not None else None),
        "input_mode": str(request.input_mode),
        "device": str(request.device),
        "resize": [int(request.resize[0]), int(request.resize[1])],
        "dataset_summary": {
            "train_count": len(train_paths),
            "test_count": len(test_paths),
            "test_anomaly_count": int(np.sum(test_labels == 1)),
        },
        "threshold_provenance": {
            "strategy": (
                "train_quantile"
                if request.calibration_quantile is not None
                else "test_optimal_f1"
            ),
            "calibration_quantile": (
                float(request.calibration_quantile)
                if request.calibration_quantile is not None
                else None
            ),
        },
        "results": results,
    }
    if pixel_skip_reason is not None:
        payload["pixel_metrics_status"] = {
            "enabled": False,
            "reason": str(pixel_skip_reason),
        }

    from pyimgano.reporting.report import (
        save_jsonl_records,
        save_run_report,
        stamp_report_payload,
    )

    payload = stamp_report_payload(payload)
    if request.save_run:
        from pyimgano.reporting.environment import collect_environment
        from pyimgano.reporting.runs import build_run_dir_name, build_run_paths, ensure_run_dir

        run_dir = ensure_run_dir(
            output_dir=request.output_dir,
            name=build_run_dir_name(
                dataset=str(request.dataset),
                model=str(model_name),
                category=str(request.category),
            ),
        )
        paths = build_run_paths(run_dir)
        category_dir = paths.categories_dir / str(request.category)
        category_dir.mkdir(parents=True, exist_ok=True)
        payload["run_dir"] = str(run_dir)

        config = {
            "dataset": str(request.dataset),
            "root": str(request.root),
            "manifest_path": request.manifest_path,
            "category": str(request.category),
            "model": str(model_name),
            "requested_model": str(request.model),
            "input_mode": str(request.input_mode),
            "seed": request.seed,
            "device": str(request.device),
            "preset": request.preset,
            "pretrained": bool(request.pretrained),
            "contamination": float(request.contamination),
            "resize": [int(request.resize[0]), int(request.resize[1])],
            "model_kwargs": model_kwargs,
            "calibration_quantile": request.calibration_quantile,
            "limit_train": request.limit_train,
            "limit_test": request.limit_test,
            "pixel": True,
            "pixel_segf1": bool(request.pixel_segf1),
            "pixel_threshold_strategy": request.pixel_threshold_strategy,
            "pixel_normal_quantile": float(request.pixel_normal_quantile),
            "pixel_calibration_fraction": float(request.pixel_calibration_fraction),
            "pixel_calibration_seed": int(request.pixel_calibration_seed),
            "pixel_postprocess": (
                asdict(request.pixel_postprocess)
                if request.pixel_postprocess is not None
                else None
            ),
            "pixel_aupro_limit": float(request.pixel_aupro_limit),
            "pixel_aupro_thresholds": int(request.pixel_aupro_thresholds),
        }
        save_run_report(paths.run_dir / "environment.json", collect_environment())
        save_run_report(paths.config_json, {"config": config})
        save_run_report(paths.report_json, payload)
        save_run_report(category_dir / "report.json", payload)

        if request.per_image_jsonl:
            sample_scores = np.asarray(
                detector.decision_function(test_paths), dtype=np.float64
            ).reshape(-1)
            if sample_scores.size != len(test_paths):
                raise ValueError(
                    "Detector score count does not match test paths: "
                    f"{sample_scores.size} != {len(test_paths)}."
                )
            threshold = float(results["threshold"])
            records = [
                {
                    "index": index,
                    "dataset": str(request.dataset),
                    "category": str(request.category),
                    "input": str(path),
                    "y_true": int(test_labels[index]),
                    "score": float(sample_scores[index]),
                    "threshold": threshold,
                    "pred": int(sample_scores[index] >= threshold),
                }
                for index, path in enumerate(test_paths)
            ]
            save_jsonl_records(category_dir / "per_image.jsonl", records)

    return payload


def run_benchmark_request(request: BenchmarkRunRequest) -> dict[str, Any]:
    if bool(request.pixel):
        return _run_pixel_benchmark_request(request)

    model_name, model_kwargs, _entry = _resolve_model_run_options(request)

    return _run_benchmark_pipeline(
        dataset=str(request.dataset),
        root=str(request.root),
        manifest_path=(str(request.manifest_path) if request.manifest_path is not None else None),
        category=str(request.category),
        model=str(model_name),
        input_mode=str(request.input_mode),
        seed=(int(request.seed) if request.seed is not None else None),
        device=str(request.device),
        preset=(str(request.preset) if request.preset is not None else None),
        pretrained=bool(request.pretrained),
        contamination=float(request.contamination),
        resize=(int(request.resize[0]), int(request.resize[1])),
        model_kwargs=model_kwargs,
        calibration_quantile=(
            float(request.calibration_quantile)
            if request.calibration_quantile is not None
            else None
        ),
        limit_train=(int(request.limit_train) if request.limit_train is not None else None),
        limit_test=(int(request.limit_test) if request.limit_test is not None else None),
        manifest_split_seed=(
            int(request.manifest_split_seed) if request.manifest_split_seed is not None else None
        ),
        manifest_test_normal_fraction=float(request.manifest_test_normal_fraction),
        save_run=bool(request.save_run),
        per_image_jsonl=bool(request.per_image_jsonl),
        cache_dir=(str(request.cache_dir) if request.cache_dir is not None else None),
        load_detector_path=(
            str(request.load_detector_path) if request.load_detector_path is not None else None
        ),
        save_detector_path=(
            str(request.save_detector_path) if request.save_detector_path is not None else None
        ),
        output_dir=(str(request.output_dir) if request.output_dir is not None else None),
    )


def run_suite_request(request: SuiteRunRequest) -> dict[str, Any]:
    return _run_suite_pipeline(
        suite=str(request.suite),
        dataset=str(request.dataset),
        root=str(request.root),
        manifest_path=(str(request.manifest_path) if request.manifest_path is not None else None),
        category=str(request.category),
        input_mode=str(request.input_mode),
        seed=(int(request.seed) if request.seed is not None else None),
        device=str(request.device),
        pretrained=bool(request.pretrained),
        contamination=float(request.contamination),
        resize=(int(request.resize[0]), int(request.resize[1])),
        calibration_quantile=(
            float(request.calibration_quantile)
            if request.calibration_quantile is not None
            else None
        ),
        limit_train=(int(request.limit_train) if request.limit_train is not None else None),
        limit_test=(int(request.limit_test) if request.limit_test is not None else None),
        manifest_split_seed=(
            int(request.manifest_split_seed) if request.manifest_split_seed is not None else None
        ),
        manifest_test_normal_fraction=float(request.manifest_test_normal_fraction),
        pixel=bool(request.pixel),
        pixel_segf1=bool(request.pixel_segf1),
        pixel_threshold_strategy=(
            str(request.pixel_threshold_strategy)
            if request.pixel_threshold_strategy is not None
            else None
        ),
        pixel_normal_quantile=float(request.pixel_normal_quantile),
        pixel_calibration_fraction=float(request.pixel_calibration_fraction),
        pixel_calibration_seed=int(request.pixel_calibration_seed),
        pixel_postprocess=build_pixel_postprocess(request.pixel_postprocess),
        pixel_aupro_limit=float(request.pixel_aupro_limit),
        pixel_aupro_thresholds=int(request.pixel_aupro_thresholds),
        save_run=bool(request.save_run),
        per_image_jsonl=bool(request.per_image_jsonl),
        cache_dir=(str(request.cache_dir) if request.cache_dir is not None else None),
        output_dir=(str(request.output_dir) if request.output_dir is not None else None),
        max_models=(int(request.max_models) if request.max_models is not None else None),
        include_baselines=_normalize_name_filters(request.include_baselines),
        exclude_baselines=_normalize_name_filters(request.exclude_baselines),
        continue_on_error=bool(request.continue_on_error),
        sweep=(str(request.sweep) if request.sweep is not None else None),
        sweep_max_variants=(
            int(request.sweep_max_variants) if request.sweep_max_variants is not None else None
        ),
    )


__all__ = [
    "BenchmarkRunRequest",
    "PixelPostprocessConfig",
    "SuiteRunRequest",
    "build_pixel_postprocess",
    "run_benchmark_request",
    "run_suite_request",
]
