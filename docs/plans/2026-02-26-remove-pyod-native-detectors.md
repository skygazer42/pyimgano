# Remove PyOD Dependency (Native Detectors) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove `pyod` from `pyimgano` runtime dependencies by implementing native detector base classes and replacing PyOD-backed detectors with native implementations, while keeping the registry API stable and package size minimal.

**Architecture:** Add `pyimgano.models.base_detector.BaseDetector` and `pyimgano.models.base_deep.BaseDeepLearningDetector` to provide PyOD-like thresholding/training semantics. Update `BaseVisionDetector` and `BaseVisionDeepDetector` to inherit from these native bases. Port critical PyOD-backed classical algorithms to native cores (NumPy/SciPy/Sklearn), and remove PyOD-only heavy wrappers from the default registry where necessary. Add guard tests to prevent reintroducing `pyod`.

**Tech Stack:** Python, NumPy, SciPy, scikit-learn, PyTorch (already core), OpenCV (already core). **No PyOD**.

---

### Task 1: Add native `BaseDetector` (thresholding + predict)

**Files:**
- Create: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_thresholding.py`

**Step 1: Write failing test**
- Add tests asserting:
  - contamination validation
  - `_process_decision_scores()` sets `threshold_` and `labels_`
  - `predict()` uses `threshold_` and returns `{0,1}`

**Step 2: Run test (red)**
- Run: `pytest -q -o addopts='' tests/test_base_detector_thresholding.py`

**Step 3: Implement minimal base**
- Implement `BaseDetector` with:
  - `__init__(contamination=0.1)`
  - `_set_n_classes(y)`
  - `_process_decision_scores()`
  - `predict(X)`

**Step 4: Run test (green)**
- Same command, expected PASS

**Step 5: Commit**
- `git add pyimgano/models/base_detector.py tests/test_base_detector_thresholding.py`
- `git commit -m "feat: add native BaseDetector thresholding contract"`

---

### Task 2: Add `predict_proba` to native `BaseDetector`

**Files:**
- Modify: `pyimgano/models/base_detector.py`
- Test: `tests/test_base_detector_predict_proba.py`

**Steps:**
1. Add failing tests for `predict_proba(method="linear")` returning `(n,2)` in `[0,1]`.
2. Add optional `method="unify"` test if implemented.
3. Implement `predict_proba` (min-max scaling based on training scores; optional erf unify).
4. Run: `pytest -q -o addopts='' tests/test_base_detector_predict_proba.py`
5. Commit: `feat: add predict_proba to native BaseDetector`

---

### Task 3: Replace `pyimgano.models.baseml.BaseVisionDetector` to use native BaseDetector

**Files:**
- Modify: `pyimgano/models/baseml.py`
- Test: `tests/contracts/test_detector_contract.py`

**Steps:**
1. Write/adjust tests so `vision_ecod/vision_copod/vision_knn` contract runs without PyOD.
2. Update `BaseVisionDetector` to inherit from `pyimgano.models.base_detector.BaseDetector`.
3. Ensure feature extractor defaults still work and caching still works.
4. Run: `pytest -q -o addopts='' tests/contracts/test_detector_contract.py -k 'ecod or copod or knn'`
5. Commit: `refactor: BaseVisionDetector no longer depends on pyod`

---

### Task 4: Add native deep base (PyOD-like training loop)

**Files:**
- Create: `pyimgano/models/base_deep.py`
- Test: `tests/test_base_deep_training_loop.py`

**Steps:**
1. Add a tiny fake detector implementing `build_model/training_forward/evaluating_forward`.
2. Implement `BaseDeepLearningDetector`:
  - optimizer/criterion resolution (small mapping, no pyod utils)
  - `training_prepare/train/evaluate/decision_function`
3. Run: `pytest -q -o addopts='' tests/test_base_deep_training_loop.py`
4. Commit: `feat: add native BaseDeepLearningDetector`

---

### Task 5: Update `BaseVisionDeepDetector` to use native deep base (remove pyod.models.base_dl)

**Files:**
- Modify: `pyimgano/models/baseCv.py`
- Test: `tests/test_dl_models.py::TestPatchCore::test_initialization`

**Steps:**
1. Remove any `optional_import("pyod.models.base_dl")` and inheritance from PyOD.
2. Inherit from `pyimgano.models.base_deep.BaseDeepLearningDetector`.
3. Keep existing image-path / numpy-array dataset logic.
4. Run: `pytest -q -o addopts='' tests/test_dl_models.py -k initialization`
5. Commit: `refactor: BaseVisionDeepDetector no longer depends on pyod`

---

### Task 6: Remove `pyod`-skipping in tests (switch to native contract)

**Files:**
- Modify: `tests/test_detectors_compat.py`
- Modify: `tests/test_predict_proba_contract.py`

**Steps:**
1. Remove `pytest.importorskip("pyod")`.
2. Ensure tests use `vision_iforest` (native) on feature vectors and `predict_proba` works.
3. Run: `pytest -q -o addopts='' tests/test_detectors_compat.py tests/test_predict_proba_contract.py`
4. Commit: `test: stop skipping when pyod missing`

---

### Task 7: Add guard test to prevent reintroducing `pyod` imports

**Files:**
- Create: `tests/test_no_pyod_imports.py`

**Steps:**
1. Implement a test that scans `pyimgano/` for `import pyod` / `from pyod`.
2. Run: `pytest -q -o addopts='' tests/test_no_pyod_imports.py`
3. Commit: `test: add no-pyod-import guard`

---

### Task 8: Add `check_parameter` utility (remove pyod.utils usage)

**Files:**
- Create: `pyimgano/utils/param_check.py`
- Modify: `pyimgano/models/imdd.py`

**Steps:**
1. Replace `from pyod.utils import check_parameter`.
2. Add unit tests for parameter validation if needed.
3. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
4. Commit: `feat: add param_check utilities`

---

### Task 9: Add torch activation resolver (remove pyod.utils.torch_utility)

**Files:**
- Create: `pyimgano/utils/torch_activations.py`
- Modify: `pyimgano/models/ae1svm.py`
- Modify: `pyimgano/models/deep_svdd.py`

**Steps:**
1. Provide `get_activation_by_name("relu"/"tanh"/"sigmoid"/...)`.
2. Update call sites.
3. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
4. Commit: `feat: add torch activation helpers`

---

### Task 10: Add pairwise distance helper (remove pyod.utils.stat_models)

**Files:**
- Create: `pyimgano/utils/pairwise.py`
- Modify: `pyimgano/models/ae1svm.py`

**Steps:**
1. Implement `pairwise_distances_no_broadcast(X, Y)` in NumPy.
2. Update AE1SVM to use it.
3. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
4. Commit: `feat: add pairwise distance helper`

---

### Task 11: Port `vision_ecod` to native implementation

**Files:**
- Modify: `pyimgano/models/ecod.py`
- Test: `tests/contracts/test_detector_contract.py`

**Steps:**
1. Add focused tests for ECOD scoring monotonicity + finite outputs.
2. Implement ECOD scoring natively (paper + PyOD reference).
3. Run: `pytest -q -o addopts='' tests/contracts/test_detector_contract.py -k ecod`
4. Commit: `feat: implement ECOD natively`

---

### Task 12: Port `vision_copod` to native implementation

**Files:**
- Modify: `pyimgano/models/copod.py`

**Steps:**
1. Add/adjust tests for COPOD.
2. Implement COPOD scoring natively (copula / tail probability aggregation).
3. Run: `pytest -q -o addopts='' tests/contracts/test_detector_contract.py -k copod`
4. Commit: `feat: implement COPOD natively`

---

### Task 13: Port `vision_knn` to native implementation

**Files:**
- Modify: `pyimgano/models/knn.py`

**Steps:**
1. Preserve methods: `largest/mean/median` aggregation.
2. Implement via sklearn `NearestNeighbors` + distance aggregation.
3. Run: `pytest -q -o addopts='' tests/test_pyod_models.py::TestKNN::test_knn_methods`
4. Commit: `feat: implement KNN natively`

---

### Task 14: Port `vision_pca` to native implementation

**Files:**
- Modify: `pyimgano/models/pca.py`

**Steps:**
1. Implement PCA reconstruction error score using sklearn PCA.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py::TestPCA::test_fit_predict`
3. Commit: `feat: implement PCA detector natively`

---

### Task 15: Port `vision_kde` to native implementation

**Files:**
- Modify: `pyimgano/models/kde.py`

**Steps:**
1. Implement negative log density via sklearn `KernelDensity`.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k kde`
3. Commit: `feat: implement KDE detector natively`

---

### Task 16: Port `vision_gmm` to native implementation

**Files:**
- Modify: `pyimgano/models/gmm.py`

**Steps:**
1. Implement negative log likelihood via sklearn `GaussianMixture`.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k gmm`
3. Commit: `feat: implement GMM detector natively`

---

### Task 17: Port `vision_iforest` to non-PyOD backend (sklearn)

**Files:**
- Modify: `pyimgano/models/iforest.py`

**Steps:**
1. Use sklearn `IsolationForest` and map scores so higher => more anomalous.
2. Ensure `predict_proba` works via BaseDetector.
3. Run: `pytest -q -o addopts='' tests/test_predict_proba_contract.py`
4. Commit: `feat: implement IsolationForest without pyod`

---

### Task 18: Port `vision_sos` to native implementation

**Files:**
- Modify: `pyimgano/models/sos.py`

**Steps:**
1. Implement SOS probabilities per the algorithm (affinity + perplexity).
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k sos`
3. Commit: `feat: implement SOS natively`

---

### Task 19: Port `vision_sod` to native implementation

**Files:**
- Modify: `pyimgano/models/sod.py`

**Steps:**
1. Implement SOD via subspace projection + neighbor reference set.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k sod`
3. Commit: `feat: implement SOD natively`

---

### Task 20: Port `vision_rod` to native implementation

**Files:**
- Modify: `pyimgano/models/rod.py`

**Steps:**
1. Implement rotation-based outlier detection core scoring.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k rod`
3. Commit: `feat: implement ROD natively`

---

### Task 21: Port `vision_qmcd` to native implementation

**Files:**
- Modify: `pyimgano/models/qmcd.py`

**Steps:**
1. Implement quantile-based robust covariance distance.
2. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k qmcd`
3. Commit: `feat: implement QMCD natively`

---

### Task 22: Port `vision_lmdd` to native implementation (share with IMDD core if possible)

**Files:**
- Modify: `pyimgano/models/lmdd.py`
- Modify: `pyimgano/models/imdd.py` (optional: share core)

**Steps:**
1. Remove any PyOD imports.
2. Ensure scoring is finite and stable.
3. Run: `pytest -q -o addopts='' tests/test_pyod_models.py -k lmdd`
4. Commit: `feat: implement LMDD natively`

---

### Task 23: Port `vision_abod` to native implementation (fast mode)

**Files:**
- Modify: `pyimgano/models/abod.py`

**Steps:**
1. Implement kNN-based ABOD approximation (angle variance).
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement ABOD natively`

---

### Task 24: Port `vision_cof` to native implementation

**Files:**
- Modify: `pyimgano/models/cof.py`

**Steps:**
1. Implement COF scoring natively.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement COF natively`

---

### Task 25: Port `vision_loci` to native implementation

**Files:**
- Modify: `pyimgano/models/loci.py`

**Steps:**
1. Implement LOCI score computation.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement LOCI natively`

---

### Task 26: Port `vision_hbos` to native implementation

**Files:**
- Modify: `pyimgano/models/hbos.py`

**Steps:**
1. Implement HBOS histogram scoring natively.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement HBOS natively`

---

### Task 27: Port `vision_mcd` to non-PyOD backend (sklearn MinCovDet)

**Files:**
- Modify: `pyimgano/models/mcd.py`

**Steps:**
1. Use sklearn `MinCovDet` and Mahalanobis distance as score.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement MCD without pyod`

---

### Task 28: Port `vision_ocsvm` to non-PyOD backend (sklearn OneClassSVM)

**Files:**
- Modify: `pyimgano/models/ocsvm.py`

**Steps:**
1. Use sklearn `OneClassSVM`, convert to anomaly score (higher => more abnormal).
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement OCSVM without pyod`

---

### Task 29: Port `vision_kpca` to native implementation

**Files:**
- Modify: `pyimgano/models/kpca.py`

**Steps:**
1. Remove `from pyod.models.base import BaseDetector`.
2. Implement KPCA reconstruction or distance score using sklearn `KernelPCA`.
3. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
4. Commit: `feat: implement KPCA natively`

---

### Task 30: Port `vision_inne` to native implementation

**Files:**
- Modify: `pyimgano/models/inne.py`

**Steps:**
1. Implement INNE core algorithm natively.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement INNE natively`

---

### Task 31: Port `vision_feature_bagging` to native implementation

**Files:**
- Modify: `pyimgano/models/feature_bagging.py`

**Steps:**
1. Implement simple feature-subspace bagging with base detectors.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement FeatureBagging natively`

---

### Task 32: Port `vision_lscp` to native implementation (simplified but stable)

**Files:**
- Modify: `pyimgano/models/lscp.py`

**Steps:**
1. Implement a simplified LSCP ensemble selector.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement LSCP natively`

---

### Task 33: Port `vision_suod` to native implementation (score-ensemble backend)

**Files:**
- Modify: `pyimgano/models/suod.py`
- Modify: `pyimgano/models/score_ensemble.py` (if needed)

**Steps:**
1. Implement SUOD as an internal score ensemble wrapper (no external `suod` package).
2. Run: `pytest -q -o addopts='' tests/test_score_ensemble.py`
3. Commit: `feat: implement SUOD natively`

---

### Task 34: Port `vision_rgraph` to native implementation

**Files:**
- Modify: `pyimgano/models/rgraph.py`

**Steps:**
1. Implement robust graph-based scoring (minimal, stable).
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement RGraph natively`

---

### Task 35: Port `vision_sampling` to native implementation

**Files:**
- Modify: `pyimgano/models/sampling.py`

**Steps:**
1. Implement sampling-based detector natively.
2. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
3. Commit: `feat: implement Sampling detector natively`

---

### Task 36: Fix deep models that import PyOD base directly (`deep_svdd`, etc.)

**Files:**
- Modify: `pyimgano/models/deep_svdd.py`

**Steps:**
1. Replace `from pyod.models.base import BaseDetector` with native base usage.
2. Ensure `vision_deep_svdd` is constructible and can fit on feature vectors if intended.
3. Run: `pytest -q -o addopts='' tests/test_models_import_optional.py`
4. Commit: `refactor: remove pyod base usage from deep_svdd`

---

### Task 37: Remove (or replace) PyOD-only heavy wrappers from default registry

**Files:**
- Modify: `pyimgano/models/__init__.py`
- Modify: `tests/test_more_models_added.py`
- Modify: `docs/MODEL_INDEX.md`

**Targets (initial):**
- `vision_dif`, `vision_lunar`, `vision_anogan`, `vision_so_gaal`, `vision_so_gaal_new`, `vision_mo_gaal`, `vision_xgbod`

**Steps:**
1. Remove from `_auto_import([...])` list.
2. Update tests to reflect native-only registry.
3. Run: `pytest -q -o addopts='' tests/test_cli_smoke.py tests/test_more_models_added.py`
4. Commit: `refactor: drop pyod-only heavy wrappers from registry`

---

### Task 38: Remove PyOD version checks and any remaining references in runtime code

**Files:**
- Modify/Delete: `pyimgano/models/_version_check.py` (or make it no-op without importing pyod)
- Search/modify: remaining `pyod` mentions in `pyimgano/`

**Steps:**
1. Run: `rg -n \"\\bpyod\\b\" pyimgano | cat`
2. Remove remaining imports/usages.
3. Run: `pytest -q -o addopts='' tests/test_no_pyod_imports.py`
4. Commit: `chore: remove remaining pyod references from runtime`

---

### Task 39: Remove `pyod` from packaging metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Steps:**
1. Remove `pyod` from core dependencies.
2. Run: `python -c \"import pyimgano; import pyimgano.models; print('ok')\"`
3. Run: `pytest -q -o addopts=''`
4. Commit: `chore: remove pyod dependency from package metadata`

---

### Task 40: Documentation + migration notes for PyOD removal

**Files:**
- Modify: `docs/MIGRATION.md`
- Modify: `CHANGELOG.md`
- Optional: `docs/source/conf.py` (remove `pyod` from `autodoc_mock_imports`)

**Steps:**
1. Document removed/changed model names (if any) and the new native contract.
2. Document how to migrate code that relied on PyOD wrappers.
3. Run: `pytest -q -o addopts='' tests/test_tools_audit_public_api.py tests/test_tools_audit_registry.py`
4. Commit: `docs: add migration notes for pyod removal`

---

**Plan complete.** Recommended execution approach: implement Tasks 1-7 sequentially (contract foundations), then port detectors (Tasks 11-36), then remove packaging dependency (Tasks 38-40).

