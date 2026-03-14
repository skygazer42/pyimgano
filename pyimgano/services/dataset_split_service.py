from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedBenchmarkSplit:
    split: Any
    pixel_skip_reason: str | None = None


def load_benchmark_style_split(
    *,
    dataset: str,
    root: str,
    category: str,
    resize: tuple[int, int],
    load_masks: bool,
    manifest_path: str | None = None,
    seed: int | None = None,
    manifest_split_seed: int | None = None,
    manifest_test_normal_fraction: float = 0.2,
) -> LoadedBenchmarkSplit:
    ds = str(dataset).lower()
    resize_hw = (int(resize[0]), int(resize[1]))

    if ds == "manifest":
        from pyimgano.datasets.manifest import ManifestSplitPolicy, load_manifest_benchmark_split
        from pyimgano.pipelines.mvtec_visa import BenchmarkSplit

        resolved_manifest_path = str(root) if manifest_path is None else str(manifest_path)
        root_fallback = None if manifest_path is None else str(root)
        if manifest_split_seed is not None:
            split_seed = int(manifest_split_seed)
        elif seed is not None:
            split_seed = int(seed)
        else:
            split_seed = 0
        policy = ManifestSplitPolicy(
            seed=split_seed,
            test_normal_fraction=float(manifest_test_normal_fraction),
        )
        manifest_split = load_manifest_benchmark_split(
            manifest_path=resolved_manifest_path,
            root_fallback=root_fallback,
            category=str(category),
            resize=resize_hw,
            load_masks=bool(load_masks),
            split_policy=policy,
        )
        split = BenchmarkSplit(
            train_paths=list(manifest_split.train_paths),
            test_paths=list(manifest_split.test_paths),
            test_labels=np.asarray(manifest_split.test_labels),
            test_masks=manifest_split.test_masks,
        )
        return LoadedBenchmarkSplit(
            split=split,
            pixel_skip_reason=manifest_split.pixel_skip_reason,
        )

    from pyimgano.pipelines.mvtec_visa import load_benchmark_split

    return LoadedBenchmarkSplit(
        split=load_benchmark_split(
            dataset=str(dataset),
            root=str(root),
            category=str(category),
            resize=resize_hw,
            load_masks=bool(load_masks),
        ),
        pixel_skip_reason=None,
    )


__all__ = ["LoadedBenchmarkSplit", "load_benchmark_style_split"]
