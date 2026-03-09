"""
Image preprocessing and enhancement utilities.

This module provides comprehensive image preprocessing capabilities including:
- Edge detection (Canny, Sobel, Laplacian, etc.)
- Morphological operations (erosion, dilation, opening, closing)
- Filters (Gaussian, bilateral, median, etc.)
- Color space conversions
- Normalization and standardization
- Preprocessing pipelines
- Advanced operations (FFT, texture analysis, segmentation, etc.)
- Image augmentation (rotation, flip, color jitter, noise, blur, weather effects, etc.)
"""

from .advanced_operations import (  # Frequency domain; Texture analysis; Color space; Enhancement; Denoising; Feature extraction; Advanced morphology; Segmentation; Pyramids; Enums
    ColorSpace,
    CornerDetector,
    MorphologicalAdvanced,
    ThresholdMethod,
    anisotropic_diffusion,
    apply_advanced_morphology,
    apply_fft,
    apply_gabor_filter,
    apply_ifft,
    apply_threshold,
    compute_glcm_features,
    compute_lbp,
    contrast_stretching,
    convert_color_space,
    detect_corners,
    equalize_color_histogram,
    extract_hog_features,
    frequency_filter,
    gamma_correction,
    gaussian_pyramid,
    laplacian_pyramid,
    non_local_means_denoising,
    retinex_msr,
    retinex_ssr,
    watershed_segmentation,
)
from .augmentation_pipeline import (  # Pipeline classes; Transform classes; Preset pipelines
    AugmentationPipeline,
    ColorJitter,
    Compose,
    DefocusBlur,
    ElasticTransform,
    GaussianNoise,
    GridMask,
    MotionBlur,
    OneOf,
    RandomApply,
    RandomCutout,
    RandomFlip,
    RandomFog,
    RandomPerspective,
    RandomRain,
    RandomRotate,
    RandomScale,
    RandomShadow,
    RandomShear,
    RandomSnow,
    RandomTranslate,
    SaltPepperNoise,
    get_anomaly_augmentation,
    get_heavy_augmentation,
    get_industrial_camera_robust_augmentation,
    get_industrial_surface_defect_synthesis_augmentation,
    get_light_augmentation,
    get_medium_augmentation,
    get_weather_augmentation,
)
from .background import estimate_background_rolling_ball, subtract_background_rolling_ball
from .catalog import PreprocessingScheme, list_preprocessing_schemes, resolve_preprocessing_scheme
from .enhancer import (
    AdvancedImageEnhancer,
    ImageEnhancer,
    PreprocessingPipeline,
    apply_filter,
    edge_detection,
    morphological_operation,
    normalize_image,
)
from .guided_filter import guided_filter
from .industrial_presets import (
    IlluminationContrastKnobs,
    apply_illumination_contrast,
    defect_amplification,
    gray_world_white_balance,
    homomorphic_filter,
    max_rgb_white_balance,
    retinex_illumination_normalization,
    shading_correction,
)
from .mixin import PreprocessingMixin
from .retinex import RetinexConfig, msrcr_lite
from .tiling import tile_apply

__all__ = [
    # Main classes
    "ImageEnhancer",
    "AdvancedImageEnhancer",
    "PreprocessingPipeline",
    "PreprocessingMixin",
    # Basic operations
    "edge_detection",
    "morphological_operation",
    "apply_filter",
    "normalize_image",
    # Frequency domain
    "apply_fft",
    "apply_ifft",
    "frequency_filter",
    # Texture analysis
    "apply_gabor_filter",
    "compute_lbp",
    "compute_glcm_features",
    # Color space
    "convert_color_space",
    "equalize_color_histogram",
    # Enhancement
    "gamma_correction",
    "contrast_stretching",
    "retinex_ssr",
    "retinex_msr",
    "RetinexConfig",
    "msrcr_lite",
    # Denoising
    "non_local_means_denoising",
    "anisotropic_diffusion",
    "guided_filter",
    "estimate_background_rolling_ball",
    "subtract_background_rolling_ball",
    "tile_apply",
    # Feature extraction
    "extract_hog_features",
    "detect_corners",
    # Advanced morphology
    "apply_advanced_morphology",
    # Segmentation
    "apply_threshold",
    "watershed_segmentation",
    # Pyramids
    "gaussian_pyramid",
    "laplacian_pyramid",
    # Enums
    "ColorSpace",
    "ThresholdMethod",
    "CornerDetector",
    "MorphologicalAdvanced",
    # Augmentation pipelines
    "Compose",
    "OneOf",
    "RandomApply",
    "AugmentationPipeline",
    # Augmentation transforms
    "RandomRotate",
    "RandomFlip",
    "RandomScale",
    "RandomTranslate",
    "RandomShear",
    "RandomPerspective",
    "ColorJitter",
    "GaussianNoise",
    "SaltPepperNoise",
    "MotionBlur",
    "DefocusBlur",
    "RandomRain",
    "RandomFog",
    "RandomSnow",
    "RandomShadow",
    "RandomCutout",
    "GridMask",
    "ElasticTransform",
    # Preset augmentation pipelines
    "get_light_augmentation",
    "get_medium_augmentation",
    "get_heavy_augmentation",
    "get_weather_augmentation",
    "get_anomaly_augmentation",
    "get_industrial_camera_robust_augmentation",
    "get_industrial_surface_defect_synthesis_augmentation",
    # Industrial presets
    "gray_world_white_balance",
    "max_rgb_white_balance",
    "homomorphic_filter",
    "IlluminationContrastKnobs",
    "apply_illumination_contrast",
    "shading_correction",
    "retinex_illumination_normalization",
    "defect_amplification",
    "PreprocessingScheme",
    "list_preprocessing_schemes",
    "resolve_preprocessing_scheme",
]
