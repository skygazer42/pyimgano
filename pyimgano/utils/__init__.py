"""Utility helpers for PyImgAno."""

from __future__ import annotations

import warnings

from .image_ops import (
    Compose,
    ImagePreprocessor,
    center_crop,
    load_image,
    normalize_array,
    random_horizontal_flip,
    resize_image,
    to_numpy,
)
from .image_ops_cv import (
    AugmentationPipeline,
    add_gaussian_noise,
    adjust_brightness_contrast,
    apply_augmentations,
    canny_edges,
    clahe_equalization,
    dilate,
    erode,
    find_contours,
    gaussian_blur,
    laplacian_edges,
    morphological_close,
    morphological_open,
    motion_blur,
    random_brightness_contrast,
    random_crop_and_resize,
    random_gaussian_noise,
    random_rotation,
    sharpen,
    sobel_edges,
    to_gray,
    to_gray_equalized,
)

__all__ = [
    "Compose",
    "ImagePreprocessor",
    "center_crop",
    "load_image",
    "normalize_array",
    "random_horizontal_flip",
    "resize_image",
    "to_numpy",
    "to_gray",
    "to_gray_equalized",
    "gaussian_blur",
    "sharpen",
    "add_gaussian_noise",
    "adjust_brightness_contrast",
    "motion_blur",
    "clahe_equalization",
    "canny_edges",
    "sobel_edges",
    "laplacian_edges",
    "find_contours",
    "erode",
    "dilate",
    "morphological_open",
    "morphological_close",
    "random_rotation",
    "random_crop_and_resize",
    "random_brightness_contrast",
    "random_gaussian_noise",
    "apply_augmentations",
    "AugmentationPipeline",
]

from .augmentation import (
    AUGMENTATION_REGISTRY,
    build_augmentation_pipeline,
    list_augmentations,
    register_augmentation,
    resolve_augmentation,
)
from .defect_ops import (
    adaptive_threshold,
    background_subtraction,
    defect_preprocess_pipeline,
    difference_of_gaussian,
    enhance_edges,
    gabor_filter_bank,
    local_binary_pattern,
    multi_scale_defect_map,
    normalize_illumination,
    top_hat,
    bottom_hat,
)

__all__ += [
    "AUGMENTATION_REGISTRY",
    "register_augmentation",
    "resolve_augmentation",
    "build_augmentation_pipeline",
    "list_augmentations",
]

__all__ += [
    "normalize_illumination",
    "background_subtraction",
    "adaptive_threshold",
    "top_hat",
    "bottom_hat",
    "difference_of_gaussian",
    "gabor_filter_bank",
    "enhance_edges",
    "local_binary_pattern",
    "multi_scale_defect_map",
    "defect_preprocess_pipeline",
]

# Dataset utilities
from .datasets import (
    MVTecDataset,
    MVTecLOCODataset,
    MVTecAD2Dataset,
    BTADDataset,
    VisADataset,
    CustomDataset,
    ManifestDataset,
    load_dataset,
)

__all__ += [
    "MVTecDataset",
    "MVTecLOCODataset",
    "MVTecAD2Dataset",
    "BTADDataset",
    "VisADataset",
    "CustomDataset",
    "ManifestDataset",
    "load_dataset",
]

# Advanced visualization utilities
try:
    from .advanced_viz import (
        create_evaluation_report,
        plot_anomaly_heatmap,
        plot_confusion_matrix,
        plot_feature_space_tsne,
        plot_multi_model_comparison,
        plot_pr_curve,
        plot_roc_curve,
        plot_score_distribution,
        plot_threshold_analysis,
    )

    __all__ += [
        "plot_roc_curve",
        "plot_pr_curve",
        "plot_confusion_matrix",
        "plot_score_distribution",
        "plot_feature_space_tsne",
        "plot_anomaly_heatmap",
        "plot_multi_model_comparison",
        "plot_threshold_analysis",
        "create_evaluation_report",
    ]
except ModuleNotFoundError as exc:
    warnings.warn(
        "Optional dependency missing for advanced visualization utilities. "
        "Install with: pip install pyimgano[viz] (or pip install seaborn). "
        f"Original error: {exc}",
        RuntimeWarning,
    )

# Model management utilities
from .model_utils import (
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
    get_model_info,
    export_model_config,
    profile_model,
    ModelRegistry,
    compare_models,
)

__all__ += [
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    "get_model_info",
    "export_model_config",
    "profile_model",
    "ModelRegistry",
    "compare_models",
]

# Experiment tracking utilities
from .experiment_tracker import (
    Experiment,
    ExperimentTracker,
    track_experiment,
)

__all__ += [
    "Experiment",
    "ExperimentTracker",
    "track_experiment",
]
