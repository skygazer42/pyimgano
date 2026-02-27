"""
Models module providing unified factory and registry interfaces.

This module auto-imports all available models and registers them
in the MODEL_REGISTRY for dynamic model creation.
"""

import contextlib
import io
import warnings
from importlib import import_module
from typing import Iterable

from .baseCv import BaseVisionDeepDetector
from .baseml import BaseVisionDetector
from .registry import MODEL_REGISTRY, create_model, list_models, register_model


def _auto_import(modules: Iterable[str]) -> None:
    """
    Auto-import modules to trigger registry decorators.

    Parameters
    ----------
    modules : Iterable[str]
        Module names to import
    """
    for module_name in modules:
        try:
            # Some third-party model backends emit user-facing "please install ..."
            # messages via `print()` at import time. Keep `import pyimgano.models`
            # quiet by default; dependency errors are surfaced when constructing
            # the model.
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - Log import failures
            warnings.warn(
                f"Failed to load model module {module_name!r}: {exc}",
                RuntimeWarning,
            )


_auto_import(
    [
        # Classical ML algorithms
        "abod",  # Angle-Based Outlier Detection
        "cblof",  # Cluster-Based Local Outlier Factor
        "cof",  # Connectivity-based Outlier Factor
        "copod",  # Copula-based Outlier Detection (ICDM 2020)
        "crossmad",  # Cross-Modal Anomaly Detection (CVPR 2025)
        "dbscan",  # Density-Based Spatial Clustering
        "ecod",  # Empirical Cumulative Outlier Detection (TKDE 2022)
        "extra_trees_density",  # Random trees embedding density baseline
        "feature_bagging",  # Feature Bagging ensemble method
        "gmm",  # Gaussian Mixture Model density baseline
        "hbos",  # Histogram-Based Outlier Score
        "iforest",  # Isolation Forest
        "rrcf",  # Random cut forest
        "hst",  # Half-space trees
        "inne",  # Isolation using Nearest Neighbors
        "Isolationforest",  # Isolation Forest
        "knn",  # K-Nearest Neighbors
        "knn_degree",  # epsilon-graph degree
        "kpca",  # Kernel Principal Component Analysis
        "k_means",  # K-Means clustering-based detection
        "kde",  # Kernel Density Estimation density baseline
        "kde_ratio",  # KDE density-contrast (dual bandwidth)
        "lid",  # Local Intrinsic Dimensionality (kNN-distance statistic)
        "loci",  # Local Correlation Integral
        "loop",  # Local Outlier Probability
        "loda",  # Lightweight On-line Detector of Anomalies
        "lof",  # Local Outlier Factor
        "ldof",  # Local Distance-based Outlier Factor
        "neighborhood_entropy",  # kNN distance entropy baseline
        "odin",  # kNN indegree (ODIN)
        "lscp",  # Locally Selective Combination in Parallel
        "rgraph",  # R-Graph: robust graph-based outlier detection
        "lmdd",  # LMDD
        "mad",  # Median Absolute Deviation
        "rzscore",  # Robust z-score (median + MAD)
        "mcd",  # Minimum Covariance Determinant
        "mahalanobis",  # Mahalanobis distance baseline
        "mst_outlier",  # MST-based outlier baseline
        "dtc",  # Distance to centroid baseline
        "cook_distance",  # Influence score (PCA residual + leverage)
        "studentized_residual",  # Robust standardized PCA residual
        "dcorr",  # Distance-correlation influence
        "elliptic_envelope",  # Robust covariance / Mahalanobis baseline
        "ocsvm",  # One-Class Support Vector Machine
        "pca",  # Principal Component Analysis
        "pca_md",  # PCA + Mahalanobis distance
        "qmcd",  # Quantile-based MCD
        "rod",  # Rotation-based Outlier Detection
        "random_projection_knn",  # RP + kNN distance
        "sampling",  # Sampling-based outlier detection
        "sod",  # Subspace Outlier Detection
        "sos",  # Stochastic Outlier Selection
        "suod",  # Scalable Unsupervised Outlier Detection
        # Deep learning algorithms
        "ae",  # Autoencoder
        "ae1svm",  # Autoencoder with One-Class SVM
        "alad",  # Adversarially Learned Anomaly Detection
        "ast",  # Anomaly-aware Student-Teacher (ICCV 2023)
        "bayesianpf",  # Bayesian Prompt Flow (CVPR 2025)
        "bgad",  # Background-guided Anomaly Detection (CVPR 2023)
        "cflow",  # Conditional Normalizing Flows (WACV 2022)
        "csflow",  # Cross-scale Flows (WACV 2022)
        "cutpaste",  # CutPaste self-supervised learning (CVPR 2021)
        "deep_svdd",  # Deep Support Vector Data Description
        "devnet",  # Deviation Networks (KDD 2019)
        "dfm",  # Discriminative Feature Modeling
        "differnet",  # DifferNet learnable difference detector (WACV 2023)
        "draem",  # Discriminatively Reconstructed Embedding (ICCV 2021)
        "dsr",  # Deep Spectral Residual (WACV 2023)
        "dst",  # Double Student-Teacher (ICCV 2023)
        "efficientad",  # EfficientAD
        "fastflow",  # FastFlow normalizing flows
        "favae",  # Feature Adaptive Variational Autoencoder (ICCV 2023)
        "gcad",  # Graph Convolutional Anomaly Detection (ICCV 2023)
        "glad",  # Global-Local Adaptive Diffusion (ECCV 2024)
        "imdd",  # Image-level Multi-scale Discriminative Detector
        "inctrl",  # In-context Residual Learning (CVPR 2024)
        "intra",  # Industrial Transformer (ICCV 2023)
        "memseg",  # Memory-guided Segmentation
        "oddoneout",  # Odd-One-Out neighbor comparison (CVPR 2025)
        "one_svm_cnn",  # One-Class SVM with CNN features
        "oneformore",  # One-for-More continual diffusion (CVPR 2025)
        "padim",  # Patch Distribution Modeling
        "padim_lite",  # PaDiM-like Gaussian baseline on embeddings (image-level)
        "panda",  # Prototypical Anomaly Network (ICCV 2023)
        "patchcore",  # PatchCore patch-level detection (CVPR 2022)
        "patchcore_lite",  # PatchCore-like memory bank (image-level)
        "softpatch",  # SoftPatch-style robust patch memory (industrial AD)
        "promptad",  # Prompt-based Few-Shot Anomaly Detection (CVPR 2024)
        "pni",  # Pyramidal Normality Indexing (CVPR 2022)
        "rdplusplus",  # Reverse Distillation++ (CVPR 2022)
        "realnet",  # RealNet realistic synthetic anomalies (CVPR 2024)
        "regad",  # Registration-based Anomaly Detection (ICCV 2023)
        "reverse_distillation",  # Reverse Distillation
        "riad",  # Reconstruction from Adjacent Image Decomposition
        "simplenet",  # SimpleNet ultra-fast detector (CVPR 2023)
        "spade",  # Sub-image Anomaly Detection with SPADE (ECCV 2020)
        "ssim",  # Structural Similarity-based detection
        "ssim_struct",  # SSIM with structural features
        "stfpm",  # Student-Teacher Feature Pyramid Matching (BMVC 2021)
        "student_teacher_lite",  # Lite student-teacher via embedding regression
        "vae",  # Variational Autoencoder
        "winclip",  # WinCLIP zero-shot CLIP-based (CVPR 2023)
        # Production wrappers
        "score_ensemble",  # Score-only ensemble wrapper detector
        "core_score_standardizer",  # Standardize scores for core detectors
        "vision_score_standardizer",  # Standardize scores for vision detectors
        # Optional backend wrappers (safe to import; dependencies are checked at runtime)
        "anomalib_backend",
        "patchcore_inspection_backend",
        "openclip_backend",
        # Foundation-style PoC models (safe to import; weights are loaded lazily at runtime)
        "anomalydino",
        "superad",
        # Optional foundation + sequence modeling
        "mambaad",
        # Pipelines registered as models
        "feature_pipeline",
        "vision_embedding_core",
        # Preconfigured industrial wrappers (lightweight pipelines)
        "industrial_wrappers",
    ]
)

__all__ = [
    "BaseVisionDetector",
    "BaseVisionDeepDetector",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
]

# Optional re-exports: these should not make `import pyimgano.models` fail if a
# specific model has extra third-party dependencies.
try:  # pragma: no cover - depends on optional deps
    from .loda import VisionLODA  # noqa: E402
except Exception as exc:  # noqa: BLE001 - best-effort optional exports
    warnings.warn(
        f"Optional model export VisionLODA unavailable: {exc}",
        RuntimeWarning,
    )
else:
    __all__.append("VisionLODA")

try:  # pragma: no cover - depends on optional deps
    from .vae import VAEAnomalyDetector  # noqa: E402
except Exception as exc:  # noqa: BLE001 - best-effort optional exports
    warnings.warn(
        f"Optional model export VAEAnomalyDetector unavailable: {exc}",
        RuntimeWarning,
    )
else:
    __all__.append("VAEAnomalyDetector")

try:  # pragma: no cover - depends on optional deps
    from .ae import OptimizedAEDetector  # noqa: E402
except Exception as exc:  # noqa: BLE001 - best-effort optional exports
    warnings.warn(
        f"Optional model export OptimizedAEDetector unavailable: {exc}",
        RuntimeWarning,
    )
else:
    __all__.append("OptimizedAEDetector")
