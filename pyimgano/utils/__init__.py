"""Utility helpers for PyImgAno (import-light).

This package historically re-exported many helpers at import time. That made
`import pyimgano.utils` (and any import of submodules such as
`pyimgano.utils.optional_deps`) implicitly pull in heavy optional dependencies
like `cv2`, `torch`, and `torchvision`.

Industrial constraint: keep discovery and config validation cheap. We therefore
lazy-load most exports on first attribute access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # ---------------------------------------------------------------------
    # image_ops
    "Compose": ("image_ops", "Compose"),
    "ImagePreprocessor": ("image_ops", "ImagePreprocessor"),
    "center_crop": ("image_ops", "center_crop"),
    "load_image": ("image_ops", "load_image"),
    "normalize_array": ("image_ops", "normalize_array"),
    "random_horizontal_flip": ("image_ops", "random_horizontal_flip"),
    "resize_image": ("image_ops", "resize_image"),
    "to_numpy": ("image_ops", "to_numpy"),
    # ---------------------------------------------------------------------
    # image_ops_cv (imports cv2; keep lazy)
    "AugmentationPipeline": ("image_ops_cv", "AugmentationPipeline"),
    "add_gaussian_noise": ("image_ops_cv", "add_gaussian_noise"),
    "adjust_brightness_contrast": ("image_ops_cv", "adjust_brightness_contrast"),
    "apply_augmentations": ("image_ops_cv", "apply_augmentations"),
    "canny_edges": ("image_ops_cv", "canny_edges"),
    "clahe_equalization": ("image_ops_cv", "clahe_equalization"),
    "dilate": ("image_ops_cv", "dilate"),
    "erode": ("image_ops_cv", "erode"),
    "find_contours": ("image_ops_cv", "find_contours"),
    "gaussian_blur": ("image_ops_cv", "gaussian_blur"),
    "laplacian_edges": ("image_ops_cv", "laplacian_edges"),
    "morphological_close": ("image_ops_cv", "morphological_close"),
    "morphological_open": ("image_ops_cv", "morphological_open"),
    "motion_blur": ("image_ops_cv", "motion_blur"),
    "random_brightness_contrast": ("image_ops_cv", "random_brightness_contrast"),
    "random_crop_and_resize": ("image_ops_cv", "random_crop_and_resize"),
    "random_gaussian_noise": ("image_ops_cv", "random_gaussian_noise"),
    "random_rotation": ("image_ops_cv", "random_rotation"),
    "sharpen": ("image_ops_cv", "sharpen"),
    "sobel_edges": ("image_ops_cv", "sobel_edges"),
    "to_gray": ("image_ops_cv", "to_gray"),
    "to_gray_equalized": ("image_ops_cv", "to_gray_equalized"),
    # ---------------------------------------------------------------------
    # augmentation (may import torchvision; keep lazy)
    "AUGMENTATION_REGISTRY": ("augmentation", "AUGMENTATION_REGISTRY"),
    "register_augmentation": ("augmentation", "register_augmentation"),
    "resolve_augmentation": ("augmentation", "resolve_augmentation"),
    "build_augmentation_pipeline": ("augmentation", "build_augmentation_pipeline"),
    "list_augmentations": ("augmentation", "list_augmentations"),
    # ---------------------------------------------------------------------
    # defect_ops (imports cv2; keep lazy)
    "normalize_illumination": ("defect_ops", "normalize_illumination"),
    "background_subtraction": ("defect_ops", "background_subtraction"),
    "adaptive_threshold": ("defect_ops", "adaptive_threshold"),
    "top_hat": ("defect_ops", "top_hat"),
    "bottom_hat": ("defect_ops", "bottom_hat"),
    "difference_of_gaussian": ("defect_ops", "difference_of_gaussian"),
    "gabor_filter_bank": ("defect_ops", "gabor_filter_bank"),
    "enhance_edges": ("defect_ops", "enhance_edges"),
    "local_binary_pattern": ("defect_ops", "local_binary_pattern"),
    "multi_scale_defect_map": ("defect_ops", "multi_scale_defect_map"),
    "defect_preprocess_pipeline": ("defect_ops", "defect_preprocess_pipeline"),
    # ---------------------------------------------------------------------
    # datasets (kept lazy to avoid IO / heavy deps at import time)
    "MVTecDataset": ("datasets", "MVTecDataset"),
    "MVTecLOCODataset": ("datasets", "MVTecLOCODataset"),
    "MVTecAD2Dataset": ("datasets", "MVTecAD2Dataset"),
    "BTADDataset": ("datasets", "BTADDataset"),
    "VisADataset": ("datasets", "VisADataset"),
    "CustomDataset": ("datasets", "CustomDataset"),
    "ManifestDataset": ("datasets", "ManifestDataset"),
    "load_dataset": ("datasets", "load_dataset"),
    # ---------------------------------------------------------------------
    # advanced_viz (optional extra; keep lazy)
    "plot_anomaly_heatmap": ("advanced_viz", "plot_anomaly_heatmap"),
    "plot_confusion_matrix": ("advanced_viz", "plot_confusion_matrix"),
    "plot_feature_space_tsne": ("advanced_viz", "plot_feature_space_tsne"),
    "plot_multi_model_comparison": ("advanced_viz", "plot_multi_model_comparison"),
    "plot_pr_curve": ("advanced_viz", "plot_pr_curve"),
    "plot_roc_curve": ("advanced_viz", "plot_roc_curve"),
    "plot_score_distribution": ("advanced_viz", "plot_score_distribution"),
    "plot_threshold_analysis": ("advanced_viz", "plot_threshold_analysis"),
    "create_evaluation_report": ("advanced_viz", "create_evaluation_report"),
    # ---------------------------------------------------------------------
    # model_utils
    "save_model": ("model_utils", "save_model"),
    "load_model": ("model_utils", "load_model"),
    "save_checkpoint": ("model_utils", "save_checkpoint"),
    "load_checkpoint": ("model_utils", "load_checkpoint"),
    "get_model_info": ("model_utils", "get_model_info"),
    "export_model_config": ("model_utils", "export_model_config"),
    "profile_model": ("model_utils", "profile_model"),
    "ModelRegistry": ("model_utils", "ModelRegistry"),
    "compare_models": ("model_utils", "compare_models"),
    # ---------------------------------------------------------------------
    # experiment_tracker
    "Experiment": ("experiment_tracker", "Experiment"),
    "ExperimentTracker": ("experiment_tracker", "ExperimentTracker"),
    "track_experiment": ("experiment_tracker", "track_experiment"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr = target
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        # Make optional extras errors actionable without importing them by default.
        if module_name in {"advanced_viz"}:
            raise ImportError(
                "Optional dependency missing for advanced visualization utilities.\n"
                "Install with:\n  pip install 'pyimgano[viz]'\n"
                f"Original error: {exc}"
            ) from exc
        raise

    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - tooling convenience
    return sorted(set(globals()) | set(__all__))

