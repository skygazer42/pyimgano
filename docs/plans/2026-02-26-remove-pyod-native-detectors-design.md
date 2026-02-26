# Remove PyOD Dependency (Native Detectors) - Design

**Date:** 2026-02-26

## Goal

Make `pyimgano` fully independent from the `pyod` package by:

- Implementing `pyimgano`-native detector base classes (classical + deep-learning training loop).
- Replacing PyOD-backed classical/ML detectors with native implementations (using existing core deps only).
- Removing `pyod` from runtime dependencies while keeping the public API/CLI stable.

## Motivation

Today, a large part of the registry is “wrappers around PyOD”. This creates:

- **Dependency coupling:** importing/using detectors pulls in `pyod` (and sometimes transitive heavy deps).
- **Control limits:** bugfixes and API changes depend on upstream.
- **Product identity:** we want algorithms and contracts to be first-class `pyimgano` assets.

Removing PyOD also reduces package footprint and failure modes.

## Constraints

- **No new heavyweight dependencies** (keep package size stable or smaller).
- Keep core dependencies limited to what we already ship: `numpy`, `scipy`, `scikit-learn`, `torch`, `opencv-python`, etc.
- Maintain the registry-driven API: `pyimgano.models.create_model(...)`, `decision_function`, `predict`, `predict_proba`, and the thresholding semantics used across the codebase (`decision_scores_`, `threshold_`, `labels_`).

## Architecture

### Native Classical Base

Introduce `pyimgano.models.base_detector.BaseDetector` (name intentionally mirrors PyOD) providing:

- `contamination` validation (float in `(0, 0.5]`, and future extensibility)
- `_set_n_classes(y)` for compatibility (default 2)
- `_process_decision_scores()` to compute `threshold_` and `labels_`
- `predict(X)` using `threshold_`
- `predict_proba(X, method=...)` with `linear` (min-max) and optional `unify` mode

Then update `pyimgano.models.baseml.BaseVisionDetector` to inherit from our base and keep:

- feature extractor protocol (`extract(X) -> (n, d)` float array)
- image-path pipeline semantics
- shared disk feature caching behavior

### Native Deep Base

Introduce `pyimgano.models.base_deep.BaseDeepLearningDetector` which:

- inherits our `BaseDetector`
- provides a minimal PyOD-like training loop (`training_prepare`, `train`, `evaluate`, `decision_function`)
- expects subclasses to implement `build_model`, `training_forward`, `evaluating_forward`

Then update `pyimgano.models.baseCv.BaseVisionDeepDetector` to inherit from our deep base, keeping:

- numpy-first + path-first inputs
- dataset wrappers already in `pyimgano.datasets`
- consistent thresholding to enable `predict()` and `predict_proba()`

### Algorithm Strategy

For PyOD-backed classical detectors:

- Prefer direct, small native implementations (NumPy/SciPy/Sklearn primitives).
- Where sklearn provides a canonical estimator (e.g. `IsolationForest`, `OneClassSVM`, `KernelDensity`, `MinCovDet`), use sklearn as the base and apply `pyimgano`’s thresholding/proba contract on top.
- For PyOD-specific algorithms (ECOD/COPOD/ABOD/COF/LOCI/...), implement the core scoring logic natively, referencing PyOD and paper definitions for correctness.

For PyOD-backed “deep” wrappers that depend on external heavy stacks (GAN-based wrappers, XGBOD, etc.):

- Remove from default registry if we cannot provide a native implementation without adding large deps.
- Replace later with `pyimgano`-native equivalents (separate milestone), or reintroduce as optional backends via extras if needed.

## Testing & Verification

- Add a guard test that fails if `pyimgano/` imports `pyod` anywhere.
- Extend contract tests to cover `fit → decision_function → predict → predict_proba` for several classical detectors.
- Keep existing deep-learning tests passing by providing a compatible deep base.
- Update any tests that currently `importorskip("pyod")`.

## Risks

- Subtle differences in score scaling and `predict_proba` behavior vs PyOD may change downstream thresholds. We mitigate via consistent contract tests and conservative defaults.
- Some registry entries may be removed temporarily if they were purely wrappers around PyOD + extra deps.

## Rollout

Do the migration in small commits:

1. Add native bases + tests.
2. Switch base classes to use native bases.
3. Port/replace PyOD-backed algorithms used by tests/CLI.
4. Remove any remaining PyOD imports.
5. Remove `pyod` from `pyproject.toml`/`requirements.txt` and update docs.

