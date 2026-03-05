"""Industrial classical presets (JSON-ready).

These presets are intended to be copied into config files or used from the CLI.

Design goals:
- deterministic by default
- no implicit network downloads
- works well on small/medium industrial datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelPreset:
    """A JSON-friendly model preset.

    - `model` must be a registered model name.
    - `kwargs` must be JSON-serializable (dict/lists/scalars).
    """

    name: str
    model: str
    kwargs: Mapping[str, Any]
    description: str
    optional: bool = False
    # Optional extras required to run this preset (used for suite skip hints).
    requires_extras: tuple[str, ...] = ()


def _structural_feature_extractor(*, max_size: int = 512) -> dict[str, Any]:
    return {"name": "structural", "kwargs": {"max_size": int(max_size)}}


def _edge_stats_feature_extractor(
    *,
    canny_threshold1: int = 50,
    canny_threshold2: int = 150,
    sobel_ksize: int = 3,
) -> dict[str, Any]:
    return {
        "name": "edge_stats",
        "kwargs": {
            "canny_threshold1": int(canny_threshold1),
            "canny_threshold2": int(canny_threshold2),
            "sobel_ksize": int(sobel_ksize),
        },
    }


def _patch_stats_feature_extractor(
    *,
    grid_rc: tuple[int, int] = (4, 4),
    stats: tuple[str, ...] = ("mean", "std", "skew", "kurt"),
    resize_hw: tuple[int, int] = (128, 128),
) -> dict[str, Any]:
    return {
        "name": "patch_stats",
        "kwargs": {
            "grid": [int(grid_rc[0]), int(grid_rc[1])],
            "stats": [str(s) for s in stats],
            "resize_hw": [int(resize_hw[0]), int(resize_hw[1])],
        },
    }


def _color_hist_feature_extractor(
    *,
    colorspace: str = "hsv",
    bins: tuple[int, int, int] = (16, 16, 16),
) -> dict[str, Any]:
    return {
        "name": "color_hist",
        "kwargs": {
            "colorspace": str(colorspace),
            "bins": [int(bins[0]), int(bins[1]), int(bins[2])],
        },
    }


def _fft_lowfreq_feature_extractor(
    *,
    size_hw: tuple[int, int] = (64, 64),
    radii: tuple[int, ...] = (4, 8, 16),
) -> dict[str, Any]:
    return {
        "name": "fft_lowfreq",
        "kwargs": {
            "size_hw": [int(size_hw[0]), int(size_hw[1])],
            "radii": [int(r) for r in radii],
        },
    }


def _lbp_feature_extractor(
    *,
    n_points: int = 8,
    radius: float = 1.0,
    method: str = "uniform",
) -> dict[str, Any]:
    return {
        "name": "lbp",
        "kwargs": {"n_points": int(n_points), "radius": float(radius), "method": str(method)},
    }


def _hog_feature_extractor(
    *,
    resize_hw: tuple[int, int] = (128, 128),
    orientations: int = 9,
) -> dict[str, Any]:
    return {
        "name": "hog",
        "kwargs": {
            "resize_hw": [int(resize_hw[0]), int(resize_hw[1])],
            "orientations": int(orientations),
        },
    }


def _gabor_bank_feature_extractor(
    *,
    resize_hw: tuple[int, int] = (128, 128),
    frequencies: tuple[float, ...] = (0.1, 0.2, 0.3),
    thetas: tuple[float, ...] = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345),
) -> dict[str, Any]:
    # Use explicit theta floats (pi/4 multiples) to keep this JSON-friendly.
    return {
        "name": "gabor_bank",
        "kwargs": {
            "resize_hw": [int(resize_hw[0]), int(resize_hw[1])],
            "frequencies": [float(f) for f in frequencies],
            "thetas": [float(t) for t in thetas],
        },
    }


def _torchvision_embedding_extractor(
    *,
    backbone: str = "resnet18",
    pretrained: bool = False,
    pool: str = "avg",
) -> dict[str, Any]:
    # `pretrained=False` is critical to avoid implicit weight downloads.
    return {
        "name": "torchvision_backbone",
        "kwargs": {
            "backbone": str(backbone),
            "pretrained": bool(pretrained),
            "pool": str(pool),
        },
    }


def _core_score_standardizer(
    *,
    base_detector: str,
    base_kwargs: Mapping[str, Any] | None = None,
    method: str = "rank",
) -> dict[str, Any]:
    """JSON-friendly kwargs for `core_score_standardizer`."""

    return {
        "base_detector": str(base_detector),
        "base_kwargs": dict(base_kwargs or {}),
        "method": str(method),
    }


INDUSTRIAL_CLASSICAL_PRESETS: dict[str, ModelPreset] = {
    "industrial-structural-ecod": ModelPreset(
        name="industrial-structural-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_ecod",
        },
        description="Fast industrial baseline: structural features -> ECOD.",
    ),
    "industrial-structural-iforest": ModelPreset(
        name="industrial-structural-iforest",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_iforest",
            "core_kwargs": {"n_estimators": 200, "n_jobs": 1},
        },
        description="Robust baseline: structural features -> IsolationForest.",
    ),
    "industrial-structural-mst": ModelPreset(
        name="industrial-structural-mst",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _structural_feature_extractor(max_size=512),
            "core_detector": "core_mst_outlier",
            "core_kwargs": {"metric": "euclidean", "score_mode": "max"},
        },
        description="Graph baseline: structural features -> MST outlier score.",
    ),
    # ------------------------------------------------------------------
    # Additional CPU-friendly feature pipeline baselines (core-only).
    "industrial-edge-ecod": ModelPreset(
        name="industrial-edge-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _edge_stats_feature_extractor(),
            "core_detector": "core_ecod",
        },
        description="Fast baseline: edge statistics features (Canny/Sobel) -> ECOD.",
    ),
    "industrial-patch-stats-ecod": ModelPreset(
        name="industrial-patch-stats-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _patch_stats_feature_extractor(
                grid_rc=(4, 4), resize_hw=(128, 128)
            ),
            "core_detector": "core_ecod",
        },
        description="Fast baseline: patch-grid statistics features -> ECOD (good for texture/spot defects).",
    ),
    "industrial-color-hist-ecod": ModelPreset(
        name="industrial-color-hist-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _color_hist_feature_extractor(colorspace="hsv", bins=(16, 16, 16)),
            "core_detector": "core_ecod",
        },
        description="Fast baseline: HSV color histogram features -> ECOD (good for color/contamination anomalies).",
    ),
    "industrial-fft-lowfreq-ecod": ModelPreset(
        name="industrial-fft-lowfreq-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _fft_lowfreq_feature_extractor(size_hw=(64, 64), radii=(4, 8, 16)),
            "core_detector": "core_ecod",
        },
        description="Fast baseline: FFT low-frequency energy ratios -> ECOD (good for blur/texture shift).",
    ),
    # ------------------------------------------------------------------
    # Optional (skimage) classical texture baselines (still CPU-friendly).
    "industrial-lbp-ecod": ModelPreset(
        name="industrial-lbp-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _lbp_feature_extractor(n_points=8, radius=1.0, method="uniform"),
            "core_detector": "core_ecod",
        },
        description="Texture baseline: LBP histogram features -> ECOD.",
        optional=True,
        requires_extras=("skimage",),
    ),
    "industrial-hog-ecod": ModelPreset(
        name="industrial-hog-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _hog_feature_extractor(resize_hw=(128, 128), orientations=9),
            "core_detector": "core_ecod",
        },
        description="Texture baseline: HOG features -> ECOD.",
        optional=True,
        requires_extras=("skimage",),
    ),
    "industrial-gabor-ecod": ModelPreset(
        name="industrial-gabor-ecod",
        model="vision_feature_pipeline",
        kwargs={
            "feature_extractor": _gabor_bank_feature_extractor(),
            "core_detector": "core_ecod",
        },
        description="Texture baseline: Gabor filter bank statistics -> ECOD.",
        optional=True,
        requires_extras=("skimage",),
    ),
    "industrial-pixel-mean-absdiff-map": ModelPreset(
        name="industrial-pixel-mean-absdiff-map",
        model="vision_pixel_mean_absdiff_map",
        kwargs={
            "resize_hw": [384, 512],
            "color": "gray",
            "reduction": "topk_mean",
            "topk": 0.01,
        },
        description="Aligned template baseline: per-pixel mean template abs-diff anomaly map (fast CPU, pixel_map).",
    ),
    "industrial-pixel-gaussian-map": ModelPreset(
        name="industrial-pixel-gaussian-map",
        model="vision_pixel_gaussian_map",
        kwargs={
            "resize_hw": [384, 512],
            "color": "gray",
            "channel_reduce": "max",
            "reduction": "topk_mean",
            "topk": 0.01,
            "std_floor": 1.0,
        },
        description="Aligned template baseline: per-pixel mean+std z-score anomaly map (robust to noise/illumination, pixel_map).",
    ),
    "industrial-pixel-mad-map": ModelPreset(
        name="industrial-pixel-mad-map",
        model="vision_pixel_mad_map",
        kwargs={
            "resize_hw": [384, 512],
            "color": "gray",
            "channel_reduce": "max",
            "reduction": "topk_mean",
            "topk": 0.01,
            "mad_floor": 1.0,
            "max_train_images": 128,
            "random_state": 0,
        },
        description="Aligned template baseline: robust per-pixel median+MAD z-score anomaly map (noisy-normal friendly, pixel_map).",
    ),
    "industrial-ssim-template-map": ModelPreset(
        name="industrial-ssim-template-map",
        model="ssim_template_map",
        kwargs={
            "n_templates": 1,
            "resize_hw": [384, 512],
            "random_state": 42,
            "reduction": "topk_mean",
            "topk": 0.01,
        },
        description="Pixel-first template baseline: SSIM map vs best template (anomaly map = 1 - SSIM).",
        optional=True,  # requires scikit-image (pyimgano[skimage])
        requires_extras=("skimage",),
    ),
    "industrial-ssim-struct-map": ModelPreset(
        name="industrial-ssim-struct-map",
        model="ssim_struct_map",
        kwargs={
            "n_templates": 1,
            "resize_hw": [384, 512],
            "canny_threshold1": 50,
            "canny_threshold2": 150,
            "random_state": 42,
            "reduction": "topk_mean",
            "topk": 0.01,
        },
        description="Pixel-first template baseline: SSIM on edge maps (Canny) for illumination-robust template inspection.",
        optional=True,  # requires scikit-image (pyimgano[skimage])
        requires_extras=("skimage",),
    ),
    "industrial-template-ncc-map": ModelPreset(
        name="industrial-template-ncc-map",
        model="vision_template_ncc_map",
        kwargs={
            "n_templates": 1,
            "resize_hw": [384, 512],
            "window_hw": [11, 11],
            "random_state": 42,
            "reduction": "topk_mean",
            "topk": 0.01,
        },
        description="Pixel-first template baseline: local NCC similarity vs best template → anomaly map (aligned inspection).",
    ),
    "industrial-phase-correlation-map": ModelPreset(
        name="industrial-phase-correlation-map",
        model="vision_phase_correlation_map",
        kwargs={
            "n_templates": 1,
            "resize_hw": [384, 512],
            "random_state": 42,
            "reduction": "topk_mean",
            "topk": 0.01,
            "upsample_factor": 1,
        },
        description="Pixel-first template baseline: phase-correlation alignment vs best template + abs-diff anomaly map (misalignment tolerant).",
        optional=True,  # requires scikit-image (pyimgano[skimage])
        requires_extras=("skimage",),
    ),
    "industrial-embed-ecod": ModelPreset(
        name="industrial-embed-ecod",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_ecod",
        },
        description="Embeddings route: torchvision backbone embeddings -> ECOD.",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-embed-knn-cosine": ModelPreset(
        name="industrial-embed-knn-cosine",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_knn_cosine",
            "core_kwargs": {"n_neighbors": 5, "method": "largest", "normalize": True},
        },
        description="Embeddings route: torchvision backbone embeddings -> cosine kNN distance.",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-embed-mahalanobis-shrinkage": ModelPreset(
        name="industrial-embed-mahalanobis-shrinkage",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_mahalanobis_shrinkage",
            "core_kwargs": {"assume_centered": False},
        },
        description="Embeddings route: torchvision backbone embeddings -> Mahalanobis (Ledoit-Wolf shrinkage).",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-embed-mahalanobis-shrinkage-rank": ModelPreset(
        name="industrial-embed-mahalanobis-shrinkage-rank",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_score_standardizer",
            "core_kwargs": _core_score_standardizer(
                base_detector="core_mahalanobis_shrinkage",
                base_kwargs={"assume_centered": False},
                method="rank",
            ),
        },
        description="Recommended: embeddings -> Mahalanobis shrinkage -> rank standardization ([0,1]).",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-embed-lid": ModelPreset(
        name="industrial-embed-lid",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "torchvision_backbone",
            "embedding_kwargs": _torchvision_embedding_extractor(backbone="resnet18")["kwargs"],
            "core_detector": "core_lid",
            "core_kwargs": {"n_neighbors": 20},
        },
        description="Embeddings route: torchvision backbone embeddings -> LID (kNN statistic).",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-openclip-knn": ModelPreset(
        name="industrial-openclip-knn",
        model="vision_embedding_core",
        kwargs={
            "embedding_extractor": "openclip_embed",
            # `pretrained=None` to avoid implicit downloads.
            "embedding_kwargs": {"model_name": "ViT-B-32", "pretrained": None, "device": "cpu"},
            "core_detector": "core_knn",
            "core_kwargs": {"method": "largest", "n_neighbors": 5},
        },
        description="Optional: OpenCLIP embeddings -> kNN distance (requires open_clip_torch).",
        optional=True,
        requires_extras=("clip",),
    ),
    "industrial-patchcore-lite-map": ModelPreset(
        name="industrial-patchcore-lite-map",
        model="vision_patchcore_lite_map",
        kwargs={
            "backbone": "resnet18",
            "node": "layer3",
            "pretrained": False,
            "device": "cpu",
            "image_size": 224,
            "knn_backend": "sklearn",
            "metric": "euclidean",
            "n_neighbors": 1,
            "coreset_sampling_ratio": 0.2,
            "aggregation_method": "topk_mean",
            "aggregation_topk": 0.01,
            "random_seed": 0,
        },
        description="Deep pixel-map baseline: conv patch embeddings -> memory bank kNN distance map (PatchCore-lite-map).",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
    "industrial-patch-embedding-core-map": ModelPreset(
        name="industrial-patch-embedding-core-map",
        model="vision_patch_embedding_core_map",
        kwargs={
            "backbone": "resnet18",
            "node": "layer3",
            "pretrained": False,
            "device": "cpu",
            "image_size": 224,
            "core_detector": "core_dtc",
            "core_kwargs": {},
            "aggregation_method": "topk_mean",
            "aggregation_topk": 0.01,
        },
        description="Deep+classical hybrid: conv patch embeddings -> core detector -> anomaly map (generic industrial baseline).",
        optional=True,  # requires torch/torchvision (pyimgano[torch])
        requires_extras=("torch",),
    ),
}


def list_industrial_classical_presets() -> list[str]:
    return sorted(INDUSTRIAL_CLASSICAL_PRESETS.keys())


def get_industrial_classical_preset(name: str) -> ModelPreset:
    key = str(name).strip()
    if key not in INDUSTRIAL_CLASSICAL_PRESETS:
        available = ", ".join(list_industrial_classical_presets()) or "<none>"
        raise KeyError(f"Unknown preset {name!r}. Available: {available}")
    return INDUSTRIAL_CLASSICAL_PRESETS[key]


__all__ = [
    "ModelPreset",
    "INDUSTRIAL_CLASSICAL_PRESETS",
    "list_industrial_classical_presets",
    "get_industrial_classical_preset",
]
