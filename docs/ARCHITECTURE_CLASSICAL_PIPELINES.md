# Classical Pipelines Architecture (Detector + Extractor + Cache)

This note explains the recommended architecture for **classical** (non end-to-end)
anomaly detection workflows in `pyimgano`.

The goal is to make industrial workflows practical:
- large datasets (many images)
- repeated evaluation runs
- feature reuse / caching
- clear, stable contracts (`fit`, `decision_function`, `predict`)

---

## Core Contract: `BaseDetector`

All classical models converge on `pyimgano.models.base_detector.BaseDetector`:
- `decision_function(X) -> scores` (higher = more anomalous)
- `predict(X) -> {0,1}` using a fitted threshold
- `threshold_`, `labels_`, `decision_scores_` computed at fit time

This gives consistent thresholding behavior across algorithms.

---

## Vision Wrappers: `BaseVisionDetector`

Most user-facing classical models are `vision_*` registry entries (e.g. `vision_ecod`).

They inherit from `pyimgano.models.baseml.BaseVisionDetector` and accept **image inputs**:
- `paths` input mode: list of file paths
- (best-effort) `numpy` input mode: list of decoded images (H,W,C) `np.ndarray`

Internally, a vision wrapper:
1. uses a **feature extractor** to convert each image into a 1D vector
2. fits a **core detector** on the 2D feature matrix `(N, D)`
3. uses `BaseDetector` to derive `threshold_` and `labels_`

---

## Feature Extractors: `pyimgano.features`

Feature extractors are first-class:
- interface: `.extract(inputs) -> np.ndarray` shape `(N, D)`
- registry: `pyimgano.features.list_feature_extractors()` / `create_feature_extractor(...)`

Extractors can be:
- handcrafted (HOG/LBP/Gabor/FFT statistics)
- embeddings (torchvision backbones; planned)

They can accept either paths or numpy arrays (extractor-dependent).

---

## Composition: `vision_feature_pipeline`

`vision_feature_pipeline` is a registry model that composes:
- a feature extractor spec (registry name + kwargs)
- a `core_*` detector name + kwargs

This lets you build “industrial baselines” without writing new Python classes.

---

## Caching (Paths + Numpy Arrays)

When repeatedly evaluating classical detectors, feature extraction dominates runtime.

`BaseVisionDetector.set_feature_cache(cache_dir)` enables best-effort disk caching:
- `cache_dir/paths/`: cached features for path inputs (keyed by file metadata + extractor fingerprint)
- `cache_dir/arrays/`: cached features for numpy inputs (keyed by array hash + extractor fingerprint)

This is optional, local-only, and can be cleared safely.

