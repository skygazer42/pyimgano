from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from pyimgano.evaluation import evaluate_detector
from pyimgano.models import create_model
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess
from pyimgano.utils.datasets import load_dataset


DatasetName = Literal["mvtec", "mvtec_ad", "visa"]


@dataclass(frozen=True)
class BenchmarkSplit:
    train_paths: list[str]
    test_paths: list[str]
    test_labels: NDArray
    test_masks: Optional[NDArray]


def load_benchmark_split(
    *,
    dataset: DatasetName,
    root: str,
    category: str,
    resize: Optional[Tuple[int, int]] = (256, 256),
    load_masks: bool = True,
) -> BenchmarkSplit:
    """Load a (train_paths, test_paths, labels, masks) split for MVTec/VisA-like datasets."""

    ds = load_dataset(dataset, root, category=category, resize=resize, load_masks=load_masks)
    train_paths = ds.get_train_paths()
    test_paths, labels, masks = ds.get_test_paths()
    return BenchmarkSplit(
        train_paths=list(train_paths),
        test_paths=list(test_paths),
        test_labels=np.asarray(labels),
        test_masks=masks,
    )


def build_default_detector(
    *,
    model: str = "vision_patchcore",
    device: str = "cpu",
    contamination: float = 0.1,
    pretrained: bool = True,
    **kwargs,
):
    """Create a detector with sensible defaults for MVTec/VisA-style benchmarks."""

    if model == "vision_patchcore":
        return create_model(
            model,
            device=device,
            contamination=contamination,
            pretrained=pretrained,
            **kwargs,
        )

    # Classical models: rely on the default ImagePreprocessor feature extractor.
    return create_model(
        model,
        contamination=contamination,
        **kwargs,
    )


def evaluate_split(
    detector,
    split: BenchmarkSplit,
    *,
    pixel_scores: Optional[NDArray] = None,
    compute_pixel_scores: bool = True,
    postprocess: Optional[AnomalyMapPostprocess] = None,
) -> dict:
    """Fit on split.train_paths, score split.test_paths, and return evaluation dict."""

    detector.fit(split.train_paths)
    scores = detector.decision_function(split.test_paths)
    pixel_scores_used = pixel_scores

    if (
        pixel_scores_used is None
        and compute_pixel_scores
        and split.test_masks is not None
        and (
            hasattr(detector, "predict_anomaly_map")
            or hasattr(detector, "get_anomaly_map")
        )
    ):
        try:
            pixel_scores_used = _compute_pixel_scores_from_detector(
                detector,
                split.test_paths,
                split.test_masks,
                postprocess=postprocess,
            )
        except Exception:
            # Pixel maps are optional; if a model doesn't support them (or returns
            # incompatible sizes), we still want image-level metrics.
            pixel_scores_used = None

    return evaluate_detector(
        split.test_labels,
        scores,
        pixel_labels=split.test_masks,
        pixel_scores=pixel_scores_used,
    )


def _compute_pixel_scores_from_detector(
    detector,
    paths: list[str],
    masks: NDArray,
    *,
    postprocess: Optional[AnomalyMapPostprocess] = None,
) -> NDArray:
    """Compute per-image anomaly maps and align them to the GT mask size."""

    if masks.ndim != 3:
        raise ValueError(f"Expected masks to have shape (N, H, W), got {masks.shape}")

    if hasattr(detector, "predict_anomaly_map"):
        raw_maps = np.asarray(detector.predict_anomaly_map(paths))
        if raw_maps.ndim != 3:
            raise ValueError(f"Expected predict_anomaly_map to return (N, H, W), got {raw_maps.shape}")
        if raw_maps.shape[0] != len(paths):
            raise ValueError(
                "predict_anomaly_map first dimension must match paths length, "
                f"got {raw_maps.shape[0]} != {len(paths)}"
            )
        raw_maps_list = [raw_maps[i] for i in range(raw_maps.shape[0])]
    elif hasattr(detector, "get_anomaly_map"):
        raw_maps_list = [np.asarray(detector.get_anomaly_map(path)) for path in paths]
    else:
        raise ValueError("Detector does not expose get_anomaly_map or predict_anomaly_map")

    maps: list[np.ndarray] = []
    for i, anomaly_map in enumerate(raw_maps_list):
        if anomaly_map.ndim != 2:
            raise ValueError(f"Expected 2D anomaly map, got shape {anomaly_map.shape}")

        target_h, target_w = int(masks[i].shape[0]), int(masks[i].shape[1])
        if anomaly_map.shape != (target_h, target_w):
            anomaly_map = cv2.resize(
                anomaly_map,
                (target_w, target_h),
                interpolation=cv2.INTER_CUBIC,
            )

        if postprocess is not None:
            anomaly_map = postprocess(anomaly_map)

        maps.append(anomaly_map.astype(np.float32, copy=False))

    return np.stack(maps, axis=0)
