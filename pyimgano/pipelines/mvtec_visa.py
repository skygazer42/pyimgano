from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pyimgano.evaluation import evaluate_detector
from pyimgano.models import create_model
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
) -> dict:
    """Fit on split.train_paths, score split.test_paths, and return evaluation dict."""

    detector.fit(split.train_paths)
    scores = detector.decision_function(split.test_paths)
    return evaluate_detector(
        split.test_labels,
        scores,
        pixel_labels=split.test_masks,
        pixel_scores=pixel_scores,
    )

