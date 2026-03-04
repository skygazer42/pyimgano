"""pyimgano model registry + factory entrypoint.

This module provides a unified model registry and factory interface.

Industrial constraint
---------------------
Importing ``pyimgano.models`` should be lightweight and auditable. In
particular, discovery operations such as:

- ``pyimgano --list-models``
- ``pyimgano --list-models --tags vision,deep``

must *not* implicitly import heavy optional roots like ``torch`` or ``cv2``.

To achieve this, ``pyimgano.models`` populates the registry by **scanning**
model source files for ``@register_model(...)`` decorators and registering
**lazy constructors**. The real model modules are imported only when the model
is instantiated via ``create_model(...)`` or when users directly import the
implementation module.
"""

from __future__ import annotations

import ast
import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .baseCv import BaseVisionDeepDetector
from .baseml import BaseVisionDetector
from .registry import MODEL_REGISTRY, create_model, list_models, register_model


_MODEL_MODULE_ALLOWLIST: tuple[str, ...] = (
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
    "core_knn_cosine",  # kNN cosine distance baseline (embeddings)
    "core_knn_cosine_calibrated",  # kNN cosine + unsupervised score standardization
    "core_oddoneout",  # Odd-One-Out neighbor comparison (core, embeddings/features)
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
    "lof_core",  # LOF core_* registry entry (indirect in legacy auto-import)
    "lof_native",  # LOF vision wrapper (indirect in legacy auto-import)
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
    "core_mahalanobis_shrinkage",  # Mahalanobis with covariance shrinkage (embeddings)
    "core_cosine_mahalanobis",  # Mahalanobis on L2-normalized embeddings (cosine-style)
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
    "torch_autoencoder",  # Torch MLP autoencoder on embeddings/features
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
    "patchcore_lite_map",  # PatchCore-lite anomaly map (pixel-map memory bank)
    "patch_embedding_core_map",  # Patch embeddings + core detector (pixel-map baseline)
    "patchcore_online",  # PatchCore-lite with incremental memory updates (study-only)
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
    "ssim_map",  # SSIM pixel-map template baselines
    "ssim_struct",  # SSIM with structural features
    "template_ncc_map",  # NCC pixel-map template baseline
    "phase_corr_map",  # Phase correlation pixel-map template baseline
    "pixel_stats_map",  # Per-pixel statistics pixel-map baselines (mean/std/MAD)
    "ref_patch_distance",  # Reference-based patch distance pixel-map baseline
    "stfpm",  # Student-Teacher Feature Pyramid Matching (BMVC 2021)
    "student_teacher_lite",  # Lite student-teacher via embedding regression
    "vae",  # Variational Autoencoder
    "vqvae",  # VQ-VAE reconstruction baseline
    "winclip",  # WinCLIP zero-shot CLIP-based (CVPR 2023)
    # Production wrappers
    "score_ensemble",  # Score-only ensemble wrapper detector
    "core_score_standardizer",  # Standardize scores for core detectors
    "vision_score_standardizer",  # Standardize scores for vision detectors
    # Optional backend wrappers (safe to import; dependencies are checked at runtime)
    "anomalib_backend",
    "patchcore_inspection_backend",
    "openclip_backend",
    "openclip_patch_map",
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
)


def _module_source_path(module_name: str) -> Optional[Path]:
    pkg_root = Path(__file__).resolve().parent
    mod_path = pkg_root / f"{module_name}.py"
    if mod_path.is_file():
        return mod_path
    pkg_init = pkg_root / module_name / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _as_register_model_call(node: ast.AST) -> Optional[ast.Call]:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == "register_model":
        return node
    if isinstance(func, ast.Attribute) and func.attr == "register_model":
        return node
    return None


def _extract_model_name(call: ast.Call) -> Optional[str]:
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return str(first.value)
    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return str(kw.value.value)
    return None


def _extract_tags(call: ast.Call) -> tuple[str, ...]:
    for kw in call.keywords:
        if kw.arg != "tags":
            continue
        value = _safe_literal_eval(kw.value)
        if isinstance(value, (list, tuple)):
            return tuple(str(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted(str(v) for v in value))
    return ()


def _extract_metadata(call: ast.Call) -> dict[str, Any]:
    for kw in call.keywords:
        if kw.arg != "metadata":
            continue
        value = _safe_literal_eval(kw.value)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _class_defines_pixel_map_method(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name in {
            "predict_anomaly_map",
            "get_anomaly_map",
        }:
            return True
    return False


def _scan_module_for_models(module_name: str) -> list[tuple[str, tuple[str, ...], dict[str, Any]]]:
    path = _module_source_path(module_name)
    if path is None:
        warnings.warn(f"Model module {module_name!r} not found on disk; skipping scan.", RuntimeWarning)
        return []

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - import-time guardrail
        warnings.warn(f"Failed reading {path}: {exc}", RuntimeWarning)
        return []

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:  # pragma: no cover - repo should be syntactically valid
        warnings.warn(f"Failed parsing {path}: {exc}", RuntimeWarning)
        return []

    out: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        has_map_method = bool(
            isinstance(node, ast.ClassDef) and _class_defines_pixel_map_method(node)
        )

        for dec in getattr(node, "decorator_list", ()):
            call = _as_register_model_call(dec)
            if call is None:
                continue

            model_name = _extract_model_name(call)
            if model_name is None:
                warnings.warn(
                    f"Skipping a register_model decorator in {path} with non-literal name.",
                    RuntimeWarning,
                )
                continue

            tags = list(_extract_tags(call))
            if has_map_method and "pixel_map" not in tags:
                tags.append("pixel_map")
            metadata = _extract_metadata(call)
            out.append((model_name, tuple(tags), dict(metadata)))
    return out


def _make_lazy_constructor(*, model_name: str, module_name: str):
    def _ctor(*args: Any, **kwargs: Any):  # noqa: ANN401 - generic factory signature
        try:
            import_module(f"{__name__}.{module_name}")
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None)
            root = (str(missing).split(".", 1)[0] if missing else "").strip()
            dep_to_extra = {
                # Deep backends
                "torch": "torch",
                "torchvision": "torch",
                # Deployment backends
                "onnx": "onnx",
                "onnxruntime": "onnx",
                "openvino": "openvino",
                # Image / speed
                "skimage": "skimage",
                "numba": "numba",
                # Visualization
                "matplotlib": "viz",
                # Existing optional backends
                "open_clip": "clip",
                "faiss": "faiss",
                "anomalib": "anomalib",
                "diffusers": "diffusion",
                "mamba_ssm": "mamba",
            }
            extra = dep_to_extra.get(root)
            if extra is not None:
                raise ImportError(
                    f"Optional dependency {root!r} is required to construct model {model_name!r} "
                    f"(implementation module {module_name!r}).\n"
                    "Install it via:\n"
                    f"  pip install 'pyimgano[{extra}]'\n"
                    f"Original error: {exc}"
                ) from exc
            raise
        real = MODEL_REGISTRY.get(model_name)
        if real is _ctor:
            raise RuntimeError(
                f"Lazy model {model_name!r} failed to resolve after importing module {module_name!r}."
            )
        return real(*args, **kwargs)

    _ctor.__name__ = f"lazy_{model_name}"
    _ctor.__qualname__ = f"lazy_{model_name}"
    _ctor.__module__ = __name__
    setattr(_ctor, "_pyimgano_lazy", {"name": model_name, "module": module_name})
    return _ctor


def _register_lazy_models(modules: Sequence[str]) -> None:
    for module_name in modules:
        for model_name, tags, metadata in _scan_module_for_models(str(module_name)):
            meta = dict(metadata)
            meta["_lazy_placeholder"] = True
            meta["_lazy_module"] = str(module_name)
            MODEL_REGISTRY.register(
                str(model_name),
                _make_lazy_constructor(model_name=str(model_name), module_name=str(module_name)),
                tags=tags,
                metadata=meta,
                overwrite=False,
            )


def load_all_models() -> None:
    """Eagerly import all known model modules (best-effort).

    Notes
    -----
    This is intended for tooling / debugging. It defeats the import-cost
    optimization and should not be called in normal discovery flows.
    """

    for module_name in _MODEL_MODULE_ALLOWLIST:
        try:
            import_module(f"{__name__}.{module_name}")
        except Exception as exc:  # noqa: BLE001 - tooling convenience
            warnings.warn(f"Failed to import model module {module_name!r}: {exc}", RuntimeWarning)


_register_lazy_models(_MODEL_MODULE_ALLOWLIST)

__all__ = [
    "BaseVisionDetector",
    "BaseVisionDeepDetector",
    "MODEL_REGISTRY",
    "create_model",
    "list_models",
    "register_model",
]

_LAZY_EXPORTS: dict[str, str] = {
    # Optional re-exports: keep `import pyimgano.models` lightweight by avoiding
    # importing these modules unless the symbols are accessed explicitly.
    "VisionLODA": "loda",
    "VAEAnomalyDetector": "vae",
    "OptimizedAEDetector": "ae",
}


def __getattr__(name: str):  # pragma: no cover - exercised indirectly
    module_name = _LAZY_EXPORTS.get(str(name))
    if module_name is None:
        raise AttributeError(name)
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
