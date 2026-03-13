from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pyimgano.models.registry import create_model
from pyimgano.robustness.corruptions import (
    apply_blur,
    apply_geo_jitter,
    apply_glare,
    apply_jpeg,
    apply_lighting,
)
from pyimgano.services.benchmark_service import PixelPostprocessConfig, build_pixel_postprocess
import pyimgano.services.dataset_split_service as dataset_split_service
from pyimgano.services.model_options import (
    enforce_checkpoint_requirement,
    resolve_model_options,
    resolve_requested_model,
)


@dataclass(frozen=True)
class RobustnessRunRequest:
    dataset: str
    root: str
    category: str
    model: str
    resize: tuple[int, int] = (256, 256)
    input_mode: str = "numpy"
    preset: str | None = None
    device: str = "cpu"
    contamination: float = 0.1
    pretrained: bool = False
    model_kwargs: dict[str, Any] | None = None
    checkpoint_path: str | None = None
    pixel_segf1: bool = True
    pixel_normal_quantile: float = 0.999
    pixel_calibration_fraction: float = 0.2
    pixel_calibration_seed: int = 0
    pixel_postprocess: PixelPostprocessConfig | None = None
    corruptions: str = "lighting,jpeg,blur,glare,geo_jitter"
    severities: Sequence[int] = (1, 2, 3, 4, 5)
    seed: int = 0
    limit_train: int | None = None
    limit_test: int | None = None


@dataclass(frozen=True)
class _NamedCorruption:
    name: str
    fn: Callable[..., tuple[NDArray, Optional[NDArray]]]

    def __call__(
        self,
        image: NDArray,
        mask: Optional[NDArray],
        *,
        severity: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray, Optional[NDArray]]:
        return self.fn(image, mask=mask, severity=int(severity), rng=rng)


def _run_robustness_benchmark(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from pyimgano.robustness.benchmark import run_robustness_benchmark

    return run_robustness_benchmark(*args, **kwargs)


def _load_u8_rgb(path: str, *, resize: tuple[int, int]) -> NDArray:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to load images.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = int(resize[0]), int(resize[1])
    if h > 0 and w > 0 and img_rgb.shape[:2] != (h, w):
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(img_rgb, dtype=np.uint8)


def _apply_resize_to_masks(
    masks: Optional[NDArray], *, resize: tuple[int, int]
) -> Optional[NDArray]:
    if masks is None:
        return None

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shape (N,H,W), got {masks.shape}")

    target_h, target_w = int(resize[0]), int(resize[1])
    if target_h <= 0 or target_w <= 0:
        return masks

    if masks.shape[1:] == (target_h, target_w):
        return masks

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "opencv-python is required to resize masks.\n"
            "Install it via:\n  pip install 'opencv-python'\n"
            f"Original error: {exc}"
        ) from exc

    out: list[np.ndarray] = []
    for i in range(int(masks.shape[0])):
        m = np.asarray(masks[i])
        resized = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        out.append((np.asarray(resized) > 0).astype(np.uint8, copy=False))
    return np.stack(out, axis=0)


def _resolve_corruptions(names: str) -> list[_NamedCorruption]:
    requested = [n.strip() for n in str(names).split(",") if n.strip()]
    available: dict[str, _NamedCorruption] = {
        "lighting": _NamedCorruption("lighting", apply_lighting),
        "jpeg": _NamedCorruption("jpeg", apply_jpeg),
        "blur": _NamedCorruption("blur", apply_blur),
        "glare": _NamedCorruption("glare", apply_glare),
        "geo_jitter": _NamedCorruption("geo_jitter", apply_geo_jitter),
    }

    out: list[_NamedCorruption] = []
    for name in requested:
        if name not in available:
            raise ValueError(
                f"Unknown corruption: {name!r}. Supported: {', '.join(sorted(available.keys()))}"
            )
        out.append(available[name])
    if not out:
        raise ValueError("--corruptions resolved to an empty list.")
    return out


def run_robustness_request(request: RobustnessRunRequest) -> dict[str, Any]:
    import pyimgano.models  # noqa: F401

    resize_hw = (int(request.resize[0]), int(request.resize[1]))
    loaded_split = dataset_split_service.load_benchmark_style_split(
        dataset=str(request.dataset),
        root=str(request.root),
        category=str(request.category),
        resize=resize_hw,
        load_masks=True,
    )
    split = loaded_split.split

    user_kwargs = dict(request.model_kwargs or {})
    model_name, preset_model_auto_kwargs, _entry = resolve_requested_model(str(request.model))

    auto_kwargs: dict[str, Any] = dict(preset_model_auto_kwargs)
    auto_kwargs.update(
        {
            "device": str(request.device),
            "contamination": float(request.contamination),
            "pretrained": bool(request.pretrained),
        }
    )

    model_kwargs = resolve_model_options(
        model_name=model_name,
        preset=(str(request.preset) if request.preset is not None else None),
        user_kwargs=user_kwargs,
        auto_kwargs=auto_kwargs,
        checkpoint_path=(
            str(request.checkpoint_path) if request.checkpoint_path is not None else None
        ),
    )
    enforce_checkpoint_requirement(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )

    train_paths = list(split.train_paths)
    test_paths = list(split.test_paths)
    if request.limit_train is not None:
        train_paths = train_paths[: int(request.limit_train)]
    if request.limit_test is not None:
        test_paths = test_paths[: int(request.limit_test)]

    test_labels = np.asarray(split.test_labels[: len(test_paths)])
    test_masks = _apply_resize_to_masks(
        None if split.test_masks is None else np.asarray(split.test_masks[: len(test_paths)]),
        resize=resize_hw,
    )

    requested_corruptions = str(request.corruptions)
    requested_severities = [int(s) for s in list(request.severities)]
    input_mode = str(request.input_mode)

    if input_mode == "numpy":
        train_inputs = [_load_u8_rgb(p, resize=resize_hw) for p in train_paths]
        test_inputs = [_load_u8_rgb(p, resize=resize_hw) for p in test_paths]
        corruptions: Sequence[_NamedCorruption] = _resolve_corruptions(requested_corruptions)
        corruption_mode = "full"
        corruptions_skipped_reason = None
    else:
        train_inputs = train_paths
        test_inputs = test_paths
        corruptions = []
        corruption_mode = "clean_only"
        corruptions_skipped_reason = (
            "input_mode=paths: corruptions are skipped because corruptions require numpy inputs."
        )

    detector = create_model(model_name, **model_kwargs)

    notes: list[str] = []
    pixel_segf1_enabled = bool(request.pixel_segf1)
    if pixel_segf1_enabled and test_masks is None:
        pixel_segf1_enabled = False
        notes.append("pixel_segf1 disabled because dataset split has no masks.")

    supports_maps = hasattr(detector, "predict_anomaly_map") or hasattr(
        detector, "get_anomaly_map"
    )
    if pixel_segf1_enabled and not supports_maps:
        pixel_segf1_enabled = False
        notes.append(
            "pixel_segf1 disabled because detector does not expose "
            "predict_anomaly_map() or get_anomaly_map()."
        )

    if not pixel_segf1_enabled:
        test_masks = None

    report = _run_robustness_benchmark(
        detector,
        train_images=train_inputs,
        test_images=test_inputs,
        test_labels=test_labels,
        test_masks=test_masks,
        corruptions=corruptions,
        severities=requested_severities,
        seed=int(request.seed),
        pixel_segf1=pixel_segf1_enabled,
        pixel_threshold_strategy="normal_pixel_quantile",
        pixel_normal_quantile=float(request.pixel_normal_quantile),
        calibration_fraction=float(request.pixel_calibration_fraction),
        calibration_seed=int(request.pixel_calibration_seed),
        postprocess=build_pixel_postprocess(request.pixel_postprocess),
    )
    report["input_mode"] = input_mode
    report["corruption_mode"] = corruption_mode
    report["requested_corruptions"] = requested_corruptions
    report["requested_severities"] = requested_severities
    if corruptions_skipped_reason is not None:
        report["corruptions_skipped_reason"] = corruptions_skipped_reason
    if notes:
        report["notes"] = list(notes)

    return {
        "dataset": str(request.dataset),
        "category": str(request.category),
        "model": str(request.model),
        "robustness": report,
    }


__all__ = ["RobustnessRunRequest", "run_robustness_request"]
