from __future__ import annotations

import time
import zlib
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pyimgano.calibration.pixel_threshold import calibrate_normal_pixel_quantile_threshold
from pyimgano.evaluation import evaluate_detector
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

from .types import Corruption


def _stable_u32(name: str) -> int:
    return int(zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF)


def _split_fit_calibration(
    items: Sequence[Any],
    *,
    calibration_fraction: float,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    n = int(len(items))
    if n == 0:
        return [], []
    if n < 2:
        return list(items), []

    frac = float(calibration_fraction)
    if frac <= 0.0:
        return list(items), []
    if frac >= 1.0:
        # Keep at least one train sample.
        return list(items[:-1]), [items[-1]]

    n_cal = int(np.ceil(n * frac))
    n_cal = max(1, min(n - 1, n_cal))

    rng = np.random.default_rng(int(seed))
    cal_idx = set(rng.choice(n, size=n_cal, replace=False).tolist())

    fit: list[Any] = []
    cal: list[Any] = []
    for i, item in enumerate(items):
        if i in cal_idx:
            cal.append(item)
        else:
            fit.append(item)
    return fit, cal


def _call_decision_function(detector: Any, inputs: Sequence[Any]) -> NDArray:
    try:
        scores = detector.decision_function(inputs)
        return np.asarray(scores, dtype=np.float64).reshape(-1)
    except Exception as exc:
        if inputs and isinstance(inputs[0], np.ndarray):
            try:
                batch = np.stack([np.asarray(x) for x in inputs], axis=0)
            except Exception:
                raise exc
            scores = detector.decision_function(batch)
            return np.asarray(scores, dtype=np.float64).reshape(-1)
        raise


def _extract_raw_maps(detector: Any, inputs: Sequence[Any]) -> list[np.ndarray]:
    if hasattr(detector, "predict_anomaly_map"):
        maps = None
        try:
            maps = detector.predict_anomaly_map(inputs)
        except Exception:
            if inputs and isinstance(inputs[0], np.ndarray):
                try:
                    batch = np.stack([np.asarray(x) for x in inputs], axis=0)
                except Exception:
                    maps = None
                else:
                    try:
                        maps = detector.predict_anomaly_map(batch)
                    except Exception:
                        maps = None

        if maps is not None:
            arr = np.asarray(maps)
            if arr.ndim != 3:
                raise ValueError(f"Expected predict_anomaly_map to return (N,H,W), got {arr.shape}")
            if arr.shape[0] != len(inputs):
                raise ValueError(
                    "predict_anomaly_map first dimension must match inputs length, "
                    f"got {arr.shape[0]} != {len(inputs)}"
                )
            return [np.asarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]

    if hasattr(detector, "get_anomaly_map"):
        out: list[np.ndarray] = []
        for item in inputs:
            out.append(np.asarray(detector.get_anomaly_map(item), dtype=np.float32))
        return out

    raise ValueError("Detector does not expose predict_anomaly_map or get_anomaly_map.")


def _align_maps_to_masks(
    maps: Sequence[np.ndarray],
    masks: NDArray,
    *,
    postprocess: Optional[AnomalyMapPostprocess],
) -> NDArray:
    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")
    if len(maps) != int(masks.shape[0]):
        raise ValueError("maps length must match masks batch size")

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to align anomaly maps to masks.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    out: list[np.ndarray] = []
    for i, m in enumerate(maps):
        arr = np.asarray(m, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got {arr.shape}")

        target_h, target_w = int(masks[i].shape[0]), int(masks[i].shape[1])
        if arr.shape != (target_h, target_w):
            arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        if postprocess is not None:
            arr = np.asarray(postprocess(arr), dtype=np.float32)

        out.append(arr.astype(np.float32, copy=False))

    return np.stack(out, axis=0)


def _evaluate_condition(
    detector: Any,
    *,
    inputs: Sequence[Any],
    labels: NDArray,
    masks: Optional[NDArray],
    pixel_threshold: Optional[float],
    postprocess: Optional[AnomalyMapPostprocess],
    pro_integration_limit: float,
    pro_num_thresholds: int,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    scores = _call_decision_function(detector, inputs)
    maps = None
    if masks is not None and (hasattr(detector, "predict_anomaly_map") or hasattr(detector, "get_anomaly_map")):
        raw = _extract_raw_maps(detector, inputs)
        maps = _align_maps_to_masks(raw, masks, postprocess=postprocess)
    t1 = time.perf_counter()

    results = evaluate_detector(
        np.asarray(labels),
        np.asarray(scores),
        pixel_labels=masks,
        pixel_scores=maps,
        pixel_threshold=pixel_threshold,
        pro_integration_limit=pro_integration_limit,
        pro_num_thresholds=pro_num_thresholds,
    )

    latency_ms = (t1 - t0) * 1000.0 / max(1, int(len(inputs)))
    return {
        "latency_ms_per_image": float(latency_ms),
        "results": results,
    }


def run_robustness_benchmark(
    detector: Any,
    *,
    train_images: Sequence[Any],
    test_images: Sequence[Any],
    test_labels: NDArray,
    test_masks: Optional[NDArray],
    corruptions: Sequence[Corruption],
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    seed: int = 0,
    pixel_segf1: bool = True,
    pixel_threshold_strategy: Optional[str] = "normal_pixel_quantile",
    pixel_normal_quantile: float = 0.999,
    calibration_fraction: float = 0.2,
    calibration_seed: int = 0,
    postprocess: Optional[AnomalyMapPostprocess] = None,
    pro_integration_limit: float = 0.3,
    pro_num_thresholds: int = 200,
) -> dict[str, Any]:
    """Run clean + corruption robustness evaluation with a fixed pixel threshold."""

    if not train_images:
        raise ValueError("train_images must be non-empty.")
    if not test_images:
        raise ValueError("test_images must be non-empty.")

    if bool(pixel_segf1) and test_masks is None:
        raise ValueError("pixel_segf1=True requires test_masks to be provided.")

    if corruptions:
        if not test_images or not isinstance(test_images[0], np.ndarray):
            raise ValueError(
                "Corruptions require numpy image inputs. "
                "Provide test_images as RGB uint8 numpy arrays or pass corruptions=[]."
            )

    fit_images, cal_images = _split_fit_calibration(
        list(train_images),
        calibration_fraction=float(calibration_fraction),
        seed=int(calibration_seed),
    )

    detector.fit(fit_images)

    pixel_threshold = None
    if bool(pixel_segf1):
        strategy = "normal_pixel_quantile" if pixel_threshold_strategy is None else str(pixel_threshold_strategy)
        if strategy != "normal_pixel_quantile":
            raise ValueError(
                "Unsupported pixel_threshold_strategy. "
                "Supported: normal_pixel_quantile. "
                f"Got: {pixel_threshold_strategy!r}"
            )

        calibration_inputs = cal_images if cal_images else fit_images
        raw_maps = _extract_raw_maps(detector, calibration_inputs)
        vals: list[np.ndarray] = []
        for m in raw_maps:
            arr = np.asarray(m, dtype=np.float32)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D anomaly map for calibration, got {arr.shape}")
            if postprocess is not None:
                arr = np.asarray(postprocess(arr), dtype=np.float32)
            vals.append(arr.reshape(-1))

        pixel_threshold = calibrate_normal_pixel_quantile_threshold(
            np.concatenate(vals, axis=0),
            q=float(pixel_normal_quantile),
        )

    report: dict[str, Any] = {
        "pixel_threshold_strategy": pixel_threshold_strategy,
        "pixel_normal_quantile": float(pixel_normal_quantile),
        "clean": _evaluate_condition(
            detector,
            inputs=test_images,
            labels=test_labels,
            masks=test_masks,
            pixel_threshold=pixel_threshold,
            postprocess=postprocess,
            pro_integration_limit=float(pro_integration_limit),
            pro_num_thresholds=int(pro_num_thresholds),
        ),
        "corruptions": {},
    }

    for corr in corruptions:
        name = str(getattr(corr, "name", "corruption"))
        by_sev: dict[str, Any] = {}
        for sev in severities:
            sev_int = int(sev)
            cond_seed = (int(seed) + _stable_u32(name) + 1009 * sev_int) & 0xFFFFFFFF
            rng = np.random.default_rng(cond_seed)

            out_images: list[np.ndarray] = []
            out_masks_list: list[np.ndarray] | None = [] if test_masks is not None else None

            for i, img in enumerate(test_images):
                m = None
                if test_masks is not None:
                    m = np.asarray(test_masks[i])

                out_img, out_m = corr(img, m, severity=sev_int, rng=rng)
                out_images.append(np.asarray(out_img, dtype=np.uint8))

                if out_masks_list is not None:
                    if out_m is None:
                        out_m = m
                    out_masks_list.append((np.asarray(out_m) > 0).astype(np.uint8, copy=False))

            out_masks = None if out_masks_list is None else np.stack(out_masks_list, axis=0)
            by_sev[f"severity_{sev_int}"] = _evaluate_condition(
                detector,
                inputs=out_images,
                labels=test_labels,
                masks=out_masks,
                pixel_threshold=pixel_threshold,
                postprocess=postprocess,
                pro_integration_limit=float(pro_integration_limit),
                pro_num_thresholds=int(pro_num_thresholds),
            )

        report["corruptions"][name] = by_sev

    return report
