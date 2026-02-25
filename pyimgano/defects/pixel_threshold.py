from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from pyimgano.calibration.pixel_threshold import calibrate_normal_pixel_quantile_threshold
from pyimgano.defects.roi import clamp_roi_xyxy_norm, roi_mask_from_xyxy_norm


def _stack_maps(calibration_maps: Iterable[np.ndarray]) -> np.ndarray:
    maps = [np.asarray(m, dtype=np.float32) for m in calibration_maps]
    if not maps:
        raise ValueError("calibration_maps must be non-empty for quantile calibration.")
    first_shape = maps[0].shape
    if any(m.shape != first_shape for m in maps):
        shapes = ", ".join(str(m.shape) for m in maps[:4])
        suffix = "" if len(maps) <= 4 else ", ..."
        raise ValueError(
            "All calibration_maps must have the same shape. "
            f"Got: {shapes}{suffix}"
        )
    return np.stack(maps, axis=0)


def resolve_pixel_threshold(
    *,
    pixel_threshold: float | None,
    pixel_threshold_strategy: str,
    infer_config_pixel_threshold: float | None = None,
    calibration_maps: Iterable[np.ndarray] | None = None,
    pixel_normal_quantile: float = 0.999,
    infer_config_source: str = "infer_config",
    roi_xyxy_norm: Sequence[float] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve a pixel threshold for defects export with provenance.

    Priority order:
    1) explicit CLI pixel_threshold (source=explicit)
    2) infer-config pixel threshold (source=infer_config)
    3) normal-pixel quantile calibration from maps (source=train_dir)
    """

    strategy = str(pixel_threshold_strategy)

    if pixel_threshold is not None:
        thr = float(pixel_threshold)
        return thr, {"method": "fixed", "source": "explicit", "value": thr}

    if infer_config_pixel_threshold is not None:
        thr = float(infer_config_pixel_threshold)
        src = str(infer_config_source) if infer_config_source else "infer_config"
        return thr, {"method": "fixed", "source": src, "value": thr}

    if strategy == "infer_config":
        raise ValueError(
            "pixel_threshold_strategy=infer_config requires infer_config_pixel_threshold to be provided."
        )

    if strategy == "fixed":
        raise ValueError(
            "pixel_threshold_strategy=fixed requires a fixed threshold.\n"
            "Provide pixel_threshold explicitly or infer_config_pixel_threshold."
        )

    if strategy != "normal_pixel_quantile":
        raise ValueError(f"Unsupported pixel_threshold_strategy: {pixel_threshold_strategy!r}")

    if calibration_maps is None:
        raise ValueError(
            "pixel threshold strategy normal_pixel_quantile requires calibration_maps."
        )

    stack = _stack_maps(calibration_maps)
    q = float(pixel_normal_quantile)

    roi_used = False
    roi_px_per_map = None
    roi_reason = None
    if roi_xyxy_norm is not None:
        roi_xyxy_norm = clamp_roi_xyxy_norm(roi_xyxy_norm)
        roi_mask = roi_mask_from_xyxy_norm((int(stack.shape[1]), int(stack.shape[2])), roi_xyxy_norm)
        roi_px_per_map = int(np.sum(roi_mask > 0))
        if roi_px_per_map > 0:
            roi_used = True
            thr = calibrate_normal_pixel_quantile_threshold(stack[:, roi_mask > 0], q=q)
        else:
            roi_reason = "roi_empty"
            thr = calibrate_normal_pixel_quantile_threshold(stack, q=q)
    else:
        thr = calibrate_normal_pixel_quantile_threshold(stack, q=q)

    calibration_pixel_count = int(stack.size)
    if roi_used and roi_px_per_map is not None:
        calibration_pixel_count = int(int(stack.shape[0]) * int(roi_px_per_map))

    return (
        float(thr),
        {
            "method": "normal_pixel_quantile",
            "source": "train_dir",
            "q": float(q),
            "calibration_map_count": int(stack.shape[0]),
            "calibration_pixel_count": int(calibration_pixel_count),
            "roi_used": bool(roi_used),
            "roi_xyxy_norm": (list(roi_xyxy_norm) if roi_xyxy_norm is not None else None),
            "roi_pixel_count": (int(roi_px_per_map) if roi_px_per_map is not None else None),
            "roi_reason": str(roi_reason) if roi_reason is not None else None,
        },
    )
