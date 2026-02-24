from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from pyimgano.calibration.pixel_threshold import calibrate_normal_pixel_quantile_threshold


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
) -> tuple[float, dict[str, Any]]:
    """Resolve a pixel threshold for defects export with provenance.

    Priority order:
    1) explicit CLI pixel_threshold (source=explicit)
    2) infer-config pixel threshold (source=infer_config)
    3) normal-pixel quantile calibration from maps (source=train_dir)
    """

    if pixel_threshold is not None:
        thr = float(pixel_threshold)
        return thr, {"method": "fixed", "source": "explicit", "value": thr}

    if infer_config_pixel_threshold is not None:
        thr = float(infer_config_pixel_threshold)
        return thr, {"method": "fixed", "source": "infer_config", "value": thr}

    strategy = str(pixel_threshold_strategy)
    if strategy != "normal_pixel_quantile":
        raise ValueError(
            "pixel threshold is required but not provided.\n"
            "Provide --pixel-threshold, set it in infer_config.json, or use "
            "--pixel-threshold-strategy=normal_pixel_quantile with calibration maps."
        )

    if calibration_maps is None:
        raise ValueError(
            "pixel threshold strategy normal_pixel_quantile requires calibration_maps."
        )

    stack = _stack_maps(calibration_maps)
    q = float(pixel_normal_quantile)
    thr = calibrate_normal_pixel_quantile_threshold(stack, q=q)

    return (
        float(thr),
        {
            "method": "normal_pixel_quantile",
            "source": "train_dir",
            "q": float(q),
            "calibration_map_count": int(stack.shape[0]),
            "calibration_pixel_count": int(stack.size),
        },
    )

