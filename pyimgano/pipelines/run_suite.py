"""Baseline suite runner for `pyimgano-benchmark --suite ...`.

The suite runner is designed for industrial baseline selection:
- Use curated presets (model + kwargs) where possible.
- Keep running even when optional extras are missing (mark as skipped).
- Write per-baseline run artifacts under a single suite output directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyimgano.baselines.suites import Baseline, get_baseline_suite, resolve_suite_baselines
from pyimgano.models.registry import create_model
from pyimgano.reporting.report import save_jsonl_records, save_run_report, stamp_report_payload
from pyimgano.reporting.runs import build_run_dir_name, ensure_run_dir
from pyimgano.utils.extras import extra_roots, extras_install_hint

from .run_benchmark import (
    ScoreThresholdStrategy,
    _calibrate_score_threshold,
    _merge_and_filter_model_kwargs,
)


@dataclass(frozen=True)
class _SuiteSplit:
    train_paths: list[str]
    test_paths: list[str]
    test_labels: np.ndarray
    test_masks: np.ndarray | None
    pixel_skip_reason: str | None = None


def _override_known_kwargs(
    kwargs: dict[str, Any], *, resize: tuple[int, int], device: str, pretrained: bool
) -> dict[str, Any]:
    """Apply suite-level overrides to common kwarg shapes.

    We keep this conservative: only override keys that are very likely to exist
    and be semantically consistent across detectors/presets.
    """

    out = dict(kwargs)

    # Pixel/template baselines frequently accept `resize_hw`.
    if (
        "resize_hw" in out
        and isinstance(out["resize_hw"], (list, tuple))
        and len(out["resize_hw"]) == 2
    ):
        out["resize_hw"] = [int(resize[0]), int(resize[1])]

    # Many deep/vision models accept `device` and `pretrained`.
    if "device" in out:
        out["device"] = str(device)
    if "pretrained" in out:
        out["pretrained"] = bool(pretrained)

    # Common nested pattern: vision_embedding_core presets.
    if isinstance(out.get("embedding_kwargs"), Mapping):
        ek = dict(out["embedding_kwargs"])
        if "device" in ek:
            ek["device"] = str(device)
        if "pretrained" in ek:
            ek["pretrained"] = bool(pretrained)
        out["embedding_kwargs"] = ek

    return out


def _deep_merge(base: Any, override: Any) -> Any:
    """Deep merge `override` into `base` (dicts only); scalars/lists replace.

    This keeps sweep specs JSON-friendly while supporting nested overrides like:
    {"feature_extractor": {"kwargs": {"max_size": 256}}}.
    """

    if isinstance(base, Mapping) and isinstance(override, Mapping):
        out: dict[str, Any] = dict(base)
        for k, v in dict(override).items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return override


def _format_variant_name(baseline_name: str, variant: str) -> str:
    safe = "".join([c if c.isalnum() or c in ("_", "-", ".") else "_" for c in str(variant)])
    safe = safe.strip("_")
    if not safe:
        safe = "variant"
    return f"{baseline_name}__{safe}"


def _load_split(
    *,
    dataset: str,
    root: str,
    manifest_path: str | None,
    category: str,
    resize: tuple[int, int],
    seed: int | None,
    manifest_split_seed: int | None,
    manifest_test_normal_fraction: float,
    load_masks: bool,
) -> _SuiteSplit:
    ds = str(dataset).lower()
    if ds == "manifest":
        from pyimgano.datasets.manifest import ManifestSplitPolicy, load_manifest_benchmark_split

        mp = str(root) if manifest_path is None else str(manifest_path)
        root_fallback = None if manifest_path is None else str(root)
        split_seed = (
            int(manifest_split_seed)
            if manifest_split_seed is not None
            else (int(seed) if seed is not None else 0)
        )
        policy = ManifestSplitPolicy(
            seed=split_seed,
            test_normal_fraction=float(manifest_test_normal_fraction),
        )
        ms = load_manifest_benchmark_split(
            manifest_path=mp,
            root_fallback=root_fallback,
            category=str(category),
            resize=(int(resize[0]), int(resize[1])),
            load_masks=bool(load_masks),
            split_policy=policy,
        )
        return _SuiteSplit(
            train_paths=list(ms.train_paths),
            test_paths=list(ms.test_paths),
            test_labels=np.asarray(ms.test_labels),
            test_masks=(np.asarray(ms.test_masks) if ms.test_masks is not None else None),
            pixel_skip_reason=ms.pixel_skip_reason,
        )

    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    split = load_benchmark_split(
        dataset=ds,  # type: ignore[arg-type]
        root=str(root),
        category=str(category),
        resize=(int(resize[0]), int(resize[1])),
        load_masks=bool(load_masks),
    )
    return _SuiteSplit(
        train_paths=list(split.train_paths),
        test_paths=list(split.test_paths),
        test_labels=np.asarray(split.test_labels),
        test_masks=(np.asarray(split.test_masks) if split.test_masks is not None else None),
        pixel_skip_reason=None,
    )


def _can_import_root(module_root: str) -> bool:
    """Best-effort root-module existence check (no import side effects)."""

    import importlib.util

    return importlib.util.find_spec(str(module_root)) is not None


def _extra_available(extra: str) -> bool:
    return all(_can_import_root(root) for root in extra_roots(str(extra)))


def _missing_extras_hint_for_baseline(baseline: Baseline) -> str | None:
    """Return a user-facing extras hint if this baseline is likely missing deps."""

    requires_extras = tuple(getattr(baseline, "requires_extras", ()))
    if requires_extras:
        missing = [e for e in requires_extras if not _extra_available(e)]
        if missing:
            return extras_install_hint(missing)
        return None

    if not bool(baseline.optional):
        return None

    # Back-compat heuristic mapping for older presets that did not set
    # `requires_extras`. Keep it simple and rely on runtime ImportError
    # messages for details.
    name = str(baseline.name)
    model = str(baseline.model)
    desc = str(baseline.description)
    text = " ".join([name, model, desc]).lower()

    # skimage-backed template baselines
    if any(
        k in text for k in ["ssim", "phase_correlation", "phase-correlation"]
    ) and not _can_import_root("skimage"):
        return "pip install 'pyimgano[skimage]'"

    # torchvision embeddings / deep baselines
    if any(
        k in text
        for k in [
            "torchvision",
            "torchscript",
            "patchcore",
            "embedding_core",
            "vision_embedding_core",
        ]
    ) and not _can_import_root("torch"):
        return "pip install 'pyimgano[torch]'"

    return None


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _rank_table(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    # Sort descending by metric value when available.
    def _sort_key(r: dict[str, Any]) -> tuple[int, float]:
        v = _safe_float(r.get(key))
        if v is None:
            return (1, 0.0)
        return (0, -float(v))

    return sorted(rows, key=_sort_key)


def _run_one_on_split(
    *,
    baseline: Baseline,
    split: _SuiteSplit,
    dataset: str,
    category: str,
    input_mode: str,
    seed: int | None,
    device: str,
    pretrained: bool,
    contamination: float,
    resize: tuple[int, int],
    model_kwargs: dict[str, Any],
    score_threshold_strategy: ScoreThresholdStrategy,
    calibration_quantile: float | None,
    limit_train: int | None,
    limit_test: int | None,
    cache_dir: str | None,
    pixel: bool,
    pixel_segf1: bool,
    pixel_threshold_strategy: str | None,
    pixel_normal_quantile: float,
    pixel_calibration_fraction: float,
    pixel_calibration_seed: int,
    pixel_postprocess=None,
    pixel_aupro_limit: float = 0.3,
    pixel_aupro_thresholds: int = 200,
    save_run: bool,
    per_image_jsonl: bool,
    output_dir: Path | None,
) -> dict[str, Any]:
    import time

    # Apply suite-level overrides (device/pretrained/resize).
    model_kwargs = _override_known_kwargs(
        dict(model_kwargs),
        resize=resize,
        device=str(device),
        pretrained=bool(pretrained),
    )

    train_inputs = list(split.train_paths)
    test_inputs = list(split.test_paths)
    test_labels = np.asarray(split.test_labels)
    test_masks = split.test_masks

    if limit_train is not None:
        train_inputs = train_inputs[: int(limit_train)]
    if limit_test is not None:
        lim = int(limit_test)
        test_inputs = test_inputs[:lim]
        test_labels = np.asarray(test_labels)[:lim]
        if test_masks is not None:
            test_masks = np.asarray(test_masks)[:lim]

    timing: dict[str, float] = {
        "create_model_s": 0.0,
        "fit_s": 0.0,
        "score_test_s": 0.0,
        "calibrate_s": 0.0,
        "pixel_s": 0.0,
        "evaluate_s": 0.0,
        "total_s": 0.0,
    }
    t0 = time.perf_counter()

    auto_defaults: dict[str, Any] = {
        "device": str(device),
        "contamination": float(contamination),
        "pretrained": bool(pretrained),
    }
    if seed is not None:
        auto_defaults["random_seed"] = int(seed)
        auto_defaults["random_state"] = int(seed)

    create_start = time.perf_counter()
    detector = create_model(
        str(baseline.model),
        **_merge_and_filter_model_kwargs(
            str(baseline.model),
            model_kwargs=dict(model_kwargs),
            auto_defaults=auto_defaults,
        ),
    )
    timing["create_model_s"] = float(time.perf_counter() - create_start)

    if input_mode == "paths" and cache_dir is not None:
        if hasattr(detector, "set_feature_cache"):
            detector.set_feature_cache(str(cache_dir))

    fit_start = time.perf_counter()
    detector.fit(train_inputs)
    timing["fit_s"] = float(time.perf_counter() - fit_start)

    score_start = time.perf_counter()
    scores = np.asarray(detector.decision_function(test_inputs), dtype=np.float64)
    timing["score_test_s"] = float(time.perf_counter() - score_start)

    calib_start = time.perf_counter()
    calibrated_threshold, threshold_provenance = _calibrate_score_threshold(
        detector,
        train_inputs,
        strategy=score_threshold_strategy,
        calibration_quantile=calibration_quantile,
    )
    timing["calibrate_s"] = float(time.perf_counter() - calib_start)

    pixel_scores = None
    pixel_threshold = None
    pixel_status: dict[str, Any] | None = None
    if bool(pixel):
        if test_masks is None:
            pixel_status = {"enabled": False, "reason": "No ground-truth masks available."}
        elif split.pixel_skip_reason is not None:
            pixel_status = {"enabled": False, "reason": str(split.pixel_skip_reason)}
        else:
            from pyimgano.pipelines.mvtec_visa import (
                _compute_pixel_scores_from_detector,
                _extract_raw_maps_from_detector,
            )

            pixel_start = time.perf_counter()
            try:
                pixel_scores = _compute_pixel_scores_from_detector(
                    detector,
                    list(test_inputs),
                    np.asarray(test_masks),
                    postprocess=pixel_postprocess,
                )
                pixel_status = {"enabled": True, "reason": None}
            except Exception as exc:  # noqa: BLE001 - pixel maps are optional
                pixel_scores = None
                pixel_status = {"enabled": False, "reason": f"Failed computing anomaly maps: {exc}"}
            timing["pixel_s"] = float(time.perf_counter() - pixel_start)

            if bool(pixel_segf1) and pixel_scores is not None:
                strat = (
                    "normal_pixel_quantile"
                    if pixel_threshold_strategy is None
                    else str(pixel_threshold_strategy)
                )
                if strat not in ("normal_pixel_quantile", "supervised_segf1"):
                    raise ValueError(
                        "Unsupported pixel_threshold_strategy. "
                        "Supported: normal_pixel_quantile, supervised_segf1."
                    )

                if strat == "normal_pixel_quantile":
                    from pyimgano.calibration.pixel_threshold import (
                        calibrate_normal_pixel_quantile_threshold,
                    )
                    from pyimgano.utils.splits import split_train_calibration

                    train_for_cal, cal_paths = split_train_calibration(
                        list(train_inputs),
                        calibration_fraction=float(pixel_calibration_fraction),
                        seed=int(pixel_calibration_seed),
                    )
                    calib_paths = cal_paths if cal_paths else list(train_for_cal)
                    if not calib_paths:
                        calib_paths = list(train_inputs)

                    raw_maps = _extract_raw_maps_from_detector(detector, list(calib_paths))
                    vals: list[np.ndarray] = []
                    for m in raw_maps:
                        arr = np.asarray(m, dtype=np.float32)
                        if arr.ndim != 2:
                            raise ValueError(
                                f"Expected 2D anomaly map for calibration, got {arr.shape}"
                            )
                        if pixel_postprocess is not None:
                            arr = np.asarray(pixel_postprocess(arr), dtype=np.float32)
                        vals.append(arr.reshape(-1))
                    pixel_threshold = calibrate_normal_pixel_quantile_threshold(
                        np.concatenate(vals, axis=0),
                        q=float(pixel_normal_quantile),
                    )
                else:
                    from pyimgano.calibration.pixel_threshold_supervised import (
                        calibrate_supervised_segf1_threshold,
                    )

                    pixel_threshold = calibrate_supervised_segf1_threshold(
                        np.asarray(pixel_scores, dtype=np.float32),
                        np.asarray(test_masks),
                    )

    from pyimgano.evaluation import evaluate_detector

    eval_start = time.perf_counter()
    if calibrated_threshold is None:
        results = evaluate_detector(
            test_labels,
            scores,
            threshold=None,
            find_best_threshold=True,
            pixel_labels=(np.asarray(test_masks) if pixel_scores is not None else None),
            pixel_scores=(
                np.asarray(pixel_scores, dtype=np.float32) if pixel_scores is not None else None
            ),
            pixel_threshold=(float(pixel_threshold) if pixel_threshold is not None else None),
            pro_integration_limit=float(pixel_aupro_limit),
            pro_num_thresholds=int(pixel_aupro_thresholds),
        )
    else:
        results = evaluate_detector(
            test_labels,
            scores,
            threshold=float(calibrated_threshold),
            find_best_threshold=False,
            pixel_labels=(np.asarray(test_masks) if pixel_scores is not None else None),
            pixel_scores=(
                np.asarray(pixel_scores, dtype=np.float32) if pixel_scores is not None else None
            ),
            pixel_threshold=(float(pixel_threshold) if pixel_threshold is not None else None),
            pro_integration_limit=float(pixel_aupro_limit),
            pro_num_thresholds=int(pixel_aupro_thresholds),
        )
    timing["evaluate_s"] = float(time.perf_counter() - eval_start)

    timing["total_s"] = float(time.perf_counter() - t0)

    dataset_summary: dict[str, Any] = {
        "train_count": int(len(train_inputs)),
        "test_count": int(len(test_inputs)),
        "test_anomaly_count": int(np.sum(np.asarray(test_labels) == 1)),
        "test_anomaly_ratio": (
            float(np.mean(np.asarray(test_labels) == 1)) if len(test_inputs) > 0 else None
        ),
    }
    if pixel_status is not None:
        dataset_summary["pixel_metrics_status"] = dict(pixel_status)

    threshold_provenance_payload = dict(threshold_provenance)
    threshold_provenance_payload.setdefault("contamination", float(contamination))
    if calibration_quantile is not None:
        threshold_provenance_payload.setdefault(
            "calibration_quantile_requested", float(calibration_quantile)
        )

    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "category": str(category),
        "model": str(baseline.name),
        "model_resolved": str(baseline.model),
        "input_mode": str(input_mode),
        "device": str(device),
        "resize": [int(resize[0]), int(resize[1])],
        "results": results,
        "dataset_summary": dataset_summary,
        "threshold_provenance": threshold_provenance_payload,
        "timing": timing,
    }
    payload = stamp_report_payload(payload)

    if save_run and output_dir is not None:
        cat_dir = output_dir / "categories" / str(category)
        cat_dir.mkdir(parents=True, exist_ok=True)
        save_run_report(output_dir / "report.json", payload)
        save_run_report(
            output_dir / "config.json",
            {
                "config": {
                    "dataset": str(dataset),
                    "root": None,
                    "category": str(category),
                    "model": str(baseline.name),
                    "model_resolved": str(baseline.model),
                    "input_mode": str(input_mode),
                    "seed": (int(seed) if seed is not None else None),
                    "device": str(device),
                    "pretrained": bool(pretrained),
                    "contamination": float(contamination),
                    "resize": [int(resize[0]), int(resize[1])],
                    "model_kwargs": dict(model_kwargs),
                    "score_threshold_strategy": str(score_threshold_strategy),
                    "calibration_quantile": calibration_quantile,
                    "limit_train": limit_train,
                    "limit_test": limit_test,
                    "pixel": bool(pixel),
                }
            },
        )

        if per_image_jsonl:
            thr = float(payload["results"]["threshold"])
            pred = (scores >= thr).astype(int).tolist()
            y_true = np.asarray(test_labels).astype(int).tolist()
            records: list[dict[str, Any]] = []
            for i, p in enumerate(test_inputs):
                records.append(
                    {
                        "index": int(i),
                        "dataset": str(dataset),
                        "category": str(category),
                        "input": str(p),
                        "y_true": int(y_true[i]),
                        "score": float(scores[i]),
                        "threshold": float(thr),
                        "pred": int(pred[i]),
                    }
                )
            save_jsonl_records(cat_dir / "per_image.jsonl", records)

        payload["run_dir"] = str(output_dir)

    return payload


def run_baseline_suite(
    *,
    suite: str,
    dataset: str,
    root: str,
    manifest_path: str | None,
    category: str,
    input_mode: str = "paths",
    seed: int | None = None,
    device: str = "cpu",
    pretrained: bool = False,
    contamination: float = 0.1,
    resize: tuple[int, int] = (256, 256),
    score_threshold_strategy: ScoreThresholdStrategy = "train_quantile",
    calibration_quantile: float | None = None,
    limit_train: int | None = None,
    limit_test: int | None = None,
    manifest_split_seed: int | None = None,
    manifest_test_normal_fraction: float = 0.2,
    pixel: bool = False,
    pixel_segf1: bool = False,
    pixel_threshold_strategy: str | None = None,
    pixel_normal_quantile: float = 0.999,
    pixel_calibration_fraction: float = 0.2,
    pixel_calibration_seed: int = 0,
    pixel_postprocess=None,
    pixel_aupro_limit: float = 0.3,
    pixel_aupro_thresholds: int = 200,
    save_run: bool = True,
    per_image_jsonl: bool = True,
    cache_dir: str | None = None,
    output_dir: str | Path | None = None,
    max_models: int | None = None,
    include_baselines: Sequence[str] | None = None,
    exclude_baselines: Sequence[str] | None = None,
    continue_on_error: bool = True,
    sweep: str | None = None,
    sweep_max_variants: int | None = None,
) -> dict[str, Any]:
    """Run a baseline suite and return an aggregated JSON payload."""

    if sweep_max_variants is not None and int(sweep_max_variants) < 0:
        raise ValueError("--suite-sweep-max-variants must be >= 0.")

    suite_obj = get_baseline_suite(str(suite))
    baselines_all = resolve_suite_baselines(str(suite))
    baselines = list(baselines_all)

    suite_available = {str(b.name) for b in baselines_all}

    include_set: set[str] | None = None
    if include_baselines:
        include_set = {str(x).strip() for x in include_baselines if str(x).strip()}

    exclude_set: set[str] = set()
    if exclude_baselines:
        exclude_set = {str(x).strip() for x in exclude_baselines if str(x).strip()}

    if include_set:
        unknown = sorted(include_set - suite_available)
        if unknown:
            raise ValueError(f"Unknown --suite-include baselines: {', '.join(unknown)}")
        baselines = [b for b in baselines if str(b.name) in include_set]

    if exclude_set:
        unknown = sorted(exclude_set - suite_available)
        if unknown:
            raise ValueError(f"Unknown --suite-exclude baselines: {', '.join(unknown)}")
        baselines = [b for b in baselines if str(b.name) not in exclude_set]

    if max_models is not None:
        baselines = baselines[: int(max_models)]

    if bool(pixel) and str(category).lower() == "all":
        raise ValueError("--category all is not supported with --pixel in suite mode.")
    if bool(pixel) and str(input_mode) != "paths":
        raise ValueError("--input-mode currently supports only 'paths' with --pixel in suite mode.")

    suite_dir: Path | None = None
    if save_run:
        if output_dir is None:
            name = build_run_dir_name(
                dataset=str(dataset),
                model=f"suite_{str(suite_obj.name)}",
                category=(str(category) if category is not None else None),
            )
            suite_dir = ensure_run_dir(output_dir=None, name=name)
        else:
            suite_dir = Path(output_dir)
            suite_dir.mkdir(parents=True, exist_ok=True)

        from pyimgano.reporting.environment import collect_environment

        save_run_report(suite_dir / "environment.json", collect_environment())

    # Load split once for the common single-category case.
    split: _SuiteSplit | None = None
    if str(category).lower() != "all" and str(input_mode) == "paths":
        split = _load_split(
            dataset=str(dataset),
            root=str(root),
            manifest_path=(str(manifest_path) if manifest_path is not None else None),
            category=str(category),
            resize=(int(resize[0]), int(resize[1])),
            seed=(int(seed) if seed is not None else None),
            manifest_split_seed=(
                int(manifest_split_seed) if manifest_split_seed is not None else None
            ),
            manifest_test_normal_fraction=float(manifest_test_normal_fraction),
            load_masks=bool(pixel),
        )

    rows: list[dict[str, Any]] = []
    per_baseline: dict[str, Any] = {}
    skipped: dict[str, Any] = {}

    sweep_payload: dict[str, Any] | None = None
    sweep_plan = None
    if sweep is not None:
        from pyimgano.baselines.sweeps import resolve_sweep as _resolve_sweep

        sweep_plan = _resolve_sweep(str(sweep))
        sweep_payload = {"name": str(sweep_plan.name), "description": str(sweep_plan.description)}

    for b in baselines:
        baseline_key = str(b.name)

        hint = _missing_extras_hint_for_baseline(b)
        if hint is not None:
            skipped[baseline_key] = {
                "status": "skipped",
                "reason": f"Missing optional deps. Install with: {hint}",
            }
            continue

        baseline_dir = None
        if suite_dir is not None:
            baseline_dir = suite_dir / "models" / baseline_key
            baseline_dir.mkdir(parents=True, exist_ok=True)

        variants: list[tuple[str, dict[str, Any]]] = [("base", {})]
        if sweep_plan is not None:
            for v in sweep_plan.variants_by_entry.get(baseline_key, ()):
                variants.append((str(v.name), dict(v.override)))
            if sweep_max_variants is not None:
                variants = variants[: 1 + int(sweep_max_variants)]

        for variant_name, variant_override in variants:
            variant_key = (
                baseline_key
                if variant_name == "base"
                else _format_variant_name(baseline_key, variant_name)
            )

            variant_dir = None
            if suite_dir is not None:
                if variant_name == "base":
                    variant_dir = baseline_dir
                else:
                    variant_dir = (
                        baseline_dir / "variants" / str(variant_name)
                        if baseline_dir is not None
                        else None
                    )
                    if variant_dir is not None:
                        variant_dir.mkdir(parents=True, exist_ok=True)

            try:
                merged_variant_kwargs = dict(_deep_merge(dict(b.kwargs), dict(variant_override)))

                if split is None:
                    # Fallback: delegate to the existing single-run pipeline (supports --category all).
                    from pyimgano.pipelines.run_benchmark import run_benchmark

                    payload = run_benchmark(
                        dataset=str(dataset),
                        root=str(root),
                        manifest_path=(str(manifest_path) if manifest_path is not None else None),
                        category=str(category),
                        model=str(b.model),
                        input_mode=str(input_mode),
                        seed=(int(seed) if seed is not None else None),
                        device=str(device),
                        preset=None,
                        pretrained=bool(pretrained),
                        contamination=float(contamination),
                        resize=(int(resize[0]), int(resize[1])),
                        model_kwargs=_override_known_kwargs(
                            merged_variant_kwargs,
                            resize=resize,
                            device=str(device),
                            pretrained=bool(pretrained),
                        ),
                        score_threshold_strategy=score_threshold_strategy,
                        calibration_quantile=(
                            float(calibration_quantile)
                            if calibration_quantile is not None
                            else None
                        ),
                        limit_train=(int(limit_train) if limit_train is not None else None),
                        limit_test=(int(limit_test) if limit_test is not None else None),
                        manifest_split_seed=(
                            int(manifest_split_seed) if manifest_split_seed is not None else None
                        ),
                        manifest_test_normal_fraction=float(manifest_test_normal_fraction),
                        save_run=bool(save_run),
                        per_image_jsonl=bool(per_image_jsonl),
                        cache_dir=(str(cache_dir) if cache_dir is not None else None),
                        load_detector_path=None,
                        save_detector_path=None,
                        output_dir=(variant_dir if variant_dir is not None else None),
                    )
                else:
                    payload = _run_one_on_split(
                        baseline=b,
                        split=split,
                        dataset=str(dataset),
                        category=str(category),
                        input_mode=str(input_mode),
                        seed=(int(seed) if seed is not None else None),
                        device=str(device),
                        pretrained=bool(pretrained),
                        contamination=float(contamination),
                        resize=(int(resize[0]), int(resize[1])),
                        model_kwargs=merged_variant_kwargs,
                        score_threshold_strategy=score_threshold_strategy,
                        calibration_quantile=(
                            float(calibration_quantile)
                            if calibration_quantile is not None
                            else None
                        ),
                        limit_train=(int(limit_train) if limit_train is not None else None),
                        limit_test=(int(limit_test) if limit_test is not None else None),
                        cache_dir=(str(cache_dir) if cache_dir is not None else None),
                        pixel=bool(pixel),
                        pixel_segf1=bool(pixel_segf1),
                        pixel_threshold_strategy=pixel_threshold_strategy,
                        pixel_normal_quantile=float(pixel_normal_quantile),
                        pixel_calibration_fraction=float(pixel_calibration_fraction),
                        pixel_calibration_seed=int(pixel_calibration_seed),
                        pixel_postprocess=pixel_postprocess,
                        pixel_aupro_limit=float(pixel_aupro_limit),
                        pixel_aupro_thresholds=int(pixel_aupro_thresholds),
                        save_run=bool(save_run),
                        per_image_jsonl=bool(per_image_jsonl),
                        output_dir=variant_dir,
                    )

                per_baseline[variant_key] = payload

                res = payload.get("results", {})
                row = {
                    "name": variant_key,
                    "base_name": baseline_key,
                    "variant": str(variant_name),
                    "model": str(b.model),
                    "optional": bool(b.optional),
                    "auroc": _safe_float(res.get("auroc")),
                    "average_precision": _safe_float(res.get("average_precision")),
                }
                pixel_metrics = res.get("pixel_metrics")
                if isinstance(pixel_metrics, Mapping):
                    row["pixel_auroc"] = _safe_float(pixel_metrics.get("pixel_auroc"))
                    row["pixel_average_precision"] = _safe_float(
                        pixel_metrics.get("pixel_average_precision")
                    )
                    row["aupro"] = _safe_float(pixel_metrics.get("aupro"))
                    row["pixel_segf1"] = _safe_float(pixel_metrics.get("pixel_segf1"))

                if isinstance(payload.get("run_dir"), str):
                    row["run_dir"] = str(payload["run_dir"])
                rows.append(row)

            except ImportError as exc:
                skipped[variant_key] = {"status": "skipped", "reason": str(exc)}
                if not bool(continue_on_error):
                    raise
            except Exception as exc:  # noqa: BLE001 - keep suite running
                skipped[variant_key] = {"status": "error", "reason": str(exc)}
                if not bool(continue_on_error):
                    raise

    summary = {
        "by_auroc": _rank_table(rows, key="auroc"),
        "by_ap": _rank_table(rows, key="average_precision"),
    }
    if bool(pixel):
        summary["by_pixel_auroc"] = _rank_table(rows, key="pixel_auroc")
        summary["by_aupro"] = _rank_table(rows, key="aupro")

    payload: dict[str, Any] = {
        "suite": str(suite_obj.name),
        "suite_description": str(suite_obj.description),
        "sweep": sweep_payload,
        "dataset": str(dataset),
        "root": str(root),
        "manifest_path": (str(manifest_path) if manifest_path is not None else None),
        "category": str(category),
        "input_mode": str(input_mode),
        "seed": (int(seed) if seed is not None else None),
        "device": str(device),
        "pretrained": bool(pretrained),
        "contamination": float(contamination),
        "resize": [int(resize[0]), int(resize[1])],
        "pixel": bool(pixel),
        "pixel_segf1": bool(pixel_segf1),
        "score_threshold_strategy": str(score_threshold_strategy),
        "calibration_quantile": calibration_quantile,
        "baselines": [str(b.name) for b in baselines],
        "rows": rows,
        "summary": summary,
        "skipped": skipped,
    }
    payload = stamp_report_payload(payload)

    if suite_dir is not None:
        save_run_report(suite_dir / "report.json", payload)
        save_run_report(
            suite_dir / "config.json",
            {
                "config": {
                    "suite": str(suite_obj.name),
                    "dataset": str(dataset),
                    "root": str(root),
                    "manifest_path": (str(manifest_path) if manifest_path is not None else None),
                    "category": str(category),
                    "input_mode": str(input_mode),
                    "seed": (int(seed) if seed is not None else None),
                    "device": str(device),
                    "pretrained": bool(pretrained),
                    "contamination": float(contamination),
                    "resize": [int(resize[0]), int(resize[1])],
                    "suite_max_models": max_models,
                    "suite_include": sorted(include_set) if include_set else None,
                    "suite_exclude": sorted(exclude_set) if exclude_set else None,
                    "suite_continue_on_error": bool(continue_on_error),
                    "suite_sweep": (str(sweep) if sweep is not None else None),
                    "suite_sweep_max_variants": (
                        int(sweep_max_variants) if sweep_max_variants is not None else None
                    ),
                    "score_threshold_strategy": str(score_threshold_strategy),
                    "calibration_quantile": calibration_quantile,
                    "limit_train": limit_train,
                    "limit_test": limit_test,
                    "pixel": bool(pixel),
                    "pixel_segf1": bool(pixel_segf1),
                }
            },
        )
        payload["run_dir"] = str(suite_dir)

    return payload


__all__ = [
    "run_baseline_suite",
]
