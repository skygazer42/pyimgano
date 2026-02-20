# MVTec + VisA Hybrid AD Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano` “default-strong” for MVTec AD + VisA (image-level + pixel-level), with one-class default and optional few-shot calibration/training, while keeping the core install lightweight.

**Architecture:** Keep `pyimgano` as the stable API + dataset/eval/postprocess/benchmark layer; add optional backend adapters (`anomalib`, `faiss`) to reach high quality quickly without bloating core dependencies.

**Tech Stack:** Python, NumPy, OpenCV, scikit-learn, PyOD, PyTorch/TorchVision; optional `anomalib`, optional `faiss-cpu`.

---

## Notes / Constraints

- **Core install must remain light.** Optional backends must be gated behind `extras`.
- **Default training is one-class.** Few-shot is opt-in and must not break one-class usage.
- **Score semantics:** benchmarking/evaluation must use continuous scores (not 0/1 labels).
- **Where possible, prefer path-based datasets** to avoid loading full datasets into RAM.

---

### Task 1: Add optional dependency extras (`anomalib`, `faiss`)

**Files:**
- Modify: `pyproject.toml`
- Modify: `setup.py`

**Step 1: Write failing test**

No unit test needed (packaging metadata). Instead, add a doc check in CI later.

**Step 2: Implement minimal change**

- Add `[project.optional-dependencies]` extras:
  - `anomalib = ["anomalib>=<min>"]`
  - `faiss = ["faiss-cpu>=<min>"]`
  - `backends = ["pyimgano[anomalib,faiss]"]`

**Step 3: Manual verification**

Run (optional): `python -m pip install -e .[anomalib]`

**Step 4: Commit**

`git commit -m "build: add anomalib/faiss optional extras"`

---

### Task 2: Add optional-dep helpers (`requires_anomalib`, `requires_faiss`)

**Files:**
- Create: `pyimgano/utils/optional_deps.py`
- Test: `tests/test_optional_deps.py`

**Step 1: Write failing test**

```python
from pyimgano.utils.optional_deps import optional_import

def test_optional_import_missing():
    mod, err = optional_import("this_package_does_not_exist_123")
    assert mod is None
    assert err is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_optional_deps.py -v`
Expected: FAIL (module not found because helper doesn’t exist yet)

**Step 3: Write minimal implementation**

Implement:
- `optional_import(name) -> (module|None, exc|None)`
- `require(name, extra_hint)` which raises a clean `ImportError`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_optional_deps.py -v`
Expected: PASS

**Step 5: Commit**

`git commit -m "feat: add optional dependency helpers"`

---

### Task 3: Fix benchmark to use continuous scores (`decision_function`) not `predict`

**Files:**
- Modify: `pyimgano/benchmark.py`
- Test: `tests/test_integration.py`

**Step 1: Write failing test**

Update `tests/test_integration.py` so “scores” come from `decision_function()`:

```python
scores = detector.decision_function(synthetic_dataset["test_all"])
assert scores.shape == (len(synthetic_dataset["test_all"]),)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py::TestBasicWorkflow::test_fit_predict_workflow -v`
Expected: FAIL (classical models currently require feature_extractor)

**Step 3: Implement minimal benchmark fix**

- In `AlgorithmBenchmark._benchmark_single()` use:
  - `test_scores = detector.decision_function(test_images)`
  - Optionally compute labels: `test_pred = detector.predict(test_images)` if needed

**Step 4: Re-run test**

Expected: Still failing until Task 4–6 land; this task just fixes benchmark semantics.

**Step 5: Commit**

`git commit -m "fix: benchmark uses decision_function scores"`

---

### Task 4: Provide a default `feature_extractor` for classical vision detectors

**Files:**
- Modify: `pyimgano/models/baseml.py`
- Modify: `pyimgano/utils/image_ops.py` (if needed)
- Test: `tests/test_integration.py`

**Step 1: Write failing test**

In integration tests, keep:

```python
detector = models.create_model("vision_ecod", contamination=0.1)
detector.fit(synthetic_dataset["train"])
scores = detector.decision_function(synthetic_dataset["test_all"])
assert len(scores) == len(synthetic_dataset["test_all"])
```

**Step 2: Run test**

Expected: FAIL with missing `feature_extractor`.

**Step 3: Implement minimal change**

In `BaseVisionDetector.__init__`:
- if `feature_extractor is None`, default to `pyimgano.utils.ImagePreprocessor(resize=(224, 224), output_tensor=False)`

**Step 4: Run test**

Expected: progress; may still fail until Task 5 updates model constructors.

**Step 5: Commit**

`git commit -m "feat: default feature extractor for classical detectors"`

---

### Task 5: Make common classical models accept `feature_extractor=None`

**Files:**
- Modify: `pyimgano/models/ecod.py`
- Modify: `pyimgano/models/copod.py`
- Modify: `pyimgano/models/knn.py`
- Modify: `pyimgano/models/pca.py`
- Test: `tests/test_integration.py`
- Test: `tests/test_pyod_models.py`

**Step 1: Update tests**

- Add/adjust tests to ensure `create_model("vision_ecod")` works without passing `feature_extractor`.

**Step 2: Implement**

Change constructors from:

```python
def __init__(self, *, feature_extractor, ...):
```

to:

```python
def __init__(self, *, feature_extractor=None, ...):
```

and pass through to `super().__init__(..., feature_extractor=feature_extractor)`.

**Step 3: Run tests**

Run: `pytest tests/test_integration.py tests/test_pyod_models.py -v`

**Step 4: Commit**

`git commit -m "fix: classical models allow default feature extractor"`

---

### Task 6: Fix `quick_benchmark` default configs for classical models

**Files:**
- Modify: `pyimgano/benchmark.py`
- Test: `tests/test_integration.py`

**Step 1: Add test**

```python
results = quick_benchmark(..., algorithms=["ECOD", "COPOD"])
assert results["ECOD"]["success"]
```

**Step 2: Implement**

Ensure quick_benchmark works without forcing a user-provided feature extractor (relies on Task 4/5).

**Step 3: Commit**

`git commit -m "fix: quick_benchmark works without feature_extractor"`

---

### Task 7: Add VisA dataset loader (path-based) + factory support

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_datasets_visa.py`

**Step 1: Write failing test**

Create a tiny synthetic VisA-like directory structure in `tmp_path`:

```python
from pyimgano.utils.datasets import load_dataset

def test_load_visa_paths(tmp_path):
    # create visa_pytorch/<cat>/train/good, test/good, test/bad, ground_truth/bad
    ds = load_dataset("visa", str(tmp_path), category="dummy")
    train = ds.get_train_paths()
    test, labels, masks = ds.get_test_paths()
    assert len(train) > 0
    assert len(test) == len(labels)
    assert masks is None or len(masks) == len(test)
```

**Step 2: Implement**

- Add `VisADataset`:
  - Prefer the canonical `visa_pytorch` directory layout (as used in common pipelines).
  - Return **paths** (not loaded images) for train/test.
  - Load masks if available (`ground_truth/bad/*.png`).
- Add to `load_dataset()` mapping: `'visa': VisADataset`.

**Step 3: Commit**

`git commit -m "feat: add VisA dataset loader"`

---

### Task 8: Add path-based accessors for MVTec/BTAD/Custom datasets

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_datasets_paths.py`

**Step 1: Write failing test**

```python
from pyimgano.utils.datasets import MVTecDataset

def test_mvtec_get_paths(tmp_path):
    # create minimal mvtec-like structure
    ds = MVTecDataset(root=str(tmp_path), category="bottle", resize=None, load_masks=True)
    train = ds.get_train_paths()
    test, labels, masks = ds.get_test_paths()
    assert isinstance(train, list)
```

**Step 2: Implement**

Add to dataset classes:
- `get_train_paths() -> list[str]`
- `get_test_paths() -> tuple[list[str], np.ndarray, Optional[np.ndarray]]`

Keep existing `get_train_data()` / `get_test_data()` for backward compatibility.

**Step 3: Commit**

`git commit -m "feat: add path-based dataset accessors"`

---

### Task 9: Add pixel-level metrics (Pixel AUROC/AP + AUPRO)

**Files:**
- Modify: `pyimgano/evaluation.py`
- Test: `tests/test_evaluation_pixel.py`

**Step 1: Write failing test**

```python
import numpy as np
from pyimgano.evaluation import compute_pixel_auroc

def test_pixel_auroc_perfect():
    y = np.zeros((1, 10, 10), dtype=np.uint8)
    y[:, 2:5, 2:5] = 1
    s = y.astype(np.float32)
    assert compute_pixel_auroc(y, s) > 0.99
```

**Step 2: Implement**

- Add `compute_pixel_auroc(pixel_labels, pixel_scores)`
- Add `compute_pixel_ap(pixel_labels, pixel_scores)`
- Rename/alias `compute_pro_score` to `compute_aupro` (keep backward-compatible alias).

**Step 3: Commit**

`git commit -m "feat: add pixel-level evaluation metrics"`

---

### Task 10: Extend `evaluate_detector` to optionally compute pixel metrics

**Files:**
- Modify: `pyimgano/evaluation.py`
- Test: `tests/test_evaluation.py`

**Step 1: Write failing test**

```python
results = evaluate_detector(y_true, y_scores, pixel_labels=pix_y, pixel_scores=pix_s)
assert "pixel_metrics" in results
```

**Step 2: Implement**

Add optional kwargs to `evaluate_detector`:
- `pixel_labels: Optional[np.ndarray]`
- `pixel_scores: Optional[np.ndarray]`

Return:
- `pixel_metrics`: dict with Pixel AUROC/AP/AUPRO

**Step 3: Commit**

`git commit -m \"feat: evaluate_detector supports pixel-level metrics\"`

---

### Task 11: Add anomaly-map post-processing utilities

**Files:**
- Create: `pyimgano/postprocess/anomaly_map.py`
- Create: `pyimgano/postprocess/__init__.py`
- Test: `tests/test_postprocess_anomaly_map.py`

**Step 1: Write failing test**

```python
import numpy as np
from pyimgano.postprocess.anomaly_map import AnomalyMapPostprocess

def test_postprocess_preserves_shape():
    pp = AnomalyMapPostprocess(gaussian_sigma=1.0)
    m = np.random.rand(32, 32).astype("float32")
    out = pp(m)
    assert out.shape == m.shape
```

**Step 2: Implement**

- `AnomalyMapPostprocess` dataclass:
  - normalization
  - gaussian blur (cv2)
  - morphological open/close (optional)
  - connected-component filtering (optional)

**Step 3: Commit**

`git commit -m "feat: add anomaly-map postprocessing"`

---

### Task 12: PatchCore correctness improvements (no square assumption, offline-friendly weights)

**Files:**
- Modify: `pyimgano/models/patchcore.py`
- Modify: `tests/test_dl_models.py`

**Step 1: Write failing test**

- Ensure `get_anomaly_map()` does not assume `sqrt(n_patches)` but uses real spatial dims.

**Step 2: Implement**

- Change `_extract_patch_features()` to return `(features, (h, w))`.
- Store spatial size during scoring for consistent reshape.
- Add `pretrained: bool = True` parameter; tests set `pretrained=False` to avoid downloads.

**Step 3: Commit**

`git commit -m "fix: PatchCore anomaly map shape + pretrained flag"`

---

### Task 13: PatchCore optional FAISS kNN backend

**Files:**
- Create: `pyimgano/models/knn_index.py`
- Modify: `pyimgano/models/patchcore.py`
- Test: `tests/test_patchcore_knn_backend.py`

**Step 1: Test**

Write a unit test that uses the sklearn backend by default and skips faiss if not installed.

**Step 2: Implement**

- Add a small interface:
  - `KNNIndex.fit(X)`
  - `KNNIndex.kneighbors(X, k)`
- Provide `SklearnKNNIndex` and `FaissKNNIndex` (optional import).
- PatchCore accepts `knn_backend="sklearn"|"faiss"`.

**Step 3: Commit**

`git commit -m "feat: optional FAISS backend for PatchCore"`

---

### Task 14: Standardize score vs label usage in docs/examples

**Files:**
- Modify: `README.md`
- Modify: `docs/QUICKSTART.md`
- Modify: `pyimgano/benchmark.py` docstrings

**Step 1: Update docs**

- Ensure examples consistently use:
  - `scores = detector.decision_function(test_paths)`
  - `pred = detector.predict(test_paths)`

**Step 2: Commit**

`git commit -m "docs: align examples to score/label semantics"`

---

### Task 15: Add few-shot calibration utilities (threshold + postprocess tuning)

**Files:**
- Create: `pyimgano/calibration/fewshot.py`
- Create: `pyimgano/calibration/__init__.py`
- Test: `tests/test_fewshot_calibration.py`

**Step 1: Write failing test**

```python
import numpy as np
from pyimgano.calibration.fewshot import fit_threshold

def test_fit_threshold_separates():
    normal = np.array([0.1, 0.2, 0.3])
    anomaly = np.array([0.9, 0.8])
    thr = fit_threshold(normal, anomaly)
    assert 0.3 < thr < 0.8
```

**Step 2: Implement**

- `fit_threshold(normal_scores, anomaly_scores, objective="f1")`
- `tune_postprocess(params_grid, ...)` (minimal grid search)

**Step 3: Commit**

`git commit -m "feat: add few-shot calibration utilities"`

---

### Task 16: Add a default “MVTec+VisA” pipeline helper

**Files:**
- Create: `pyimgano/pipelines/mvtec_visa.py`
- Create: `pyimgano/pipelines/__init__.py`
- Test: `tests/test_pipeline_smoke.py`

**Step 1: Write failing smoke test**

```python
from pyimgano.pipelines.mvtec_visa import build_default_detector

def test_build_default_detector():
    det = build_default_detector(model="patchcore", device="cpu", pretrained=False)
    assert det is not None
```

**Step 2: Implement**

- `build_default_detector(model=..., ...)`
- `run_benchmark(dataset=..., categories=..., ...)` (small, composable functions)

**Step 3: Commit**

`git commit -m "feat: add mvtec+visa default pipeline helpers"`

---

### Task 17: Add optional anomalib backend wrapper detector(s)

**Files:**
- Create: `pyimgano/models/anomalib_backend.py`
- Modify: `pyimgano/models/__init__.py` (auto-import if deps present)
- Test: `tests/test_anomalib_backend_optional.py`

**Step 1: Test**

Test should skip if anomalib not installed.

**Step 2: Implement**

- Register models like:
  - `vision_patchcore_anomalib`
  - `vision_efficientad_anomalib`
- Wrapper should:
  - raise clean ImportError if anomalib missing
  - expose `fit`, `decision_function`, and `predict_anomaly_map` where possible

**Step 3: Commit**

`git commit -m "feat: optional anomalib backend models"`

---

### Task 18: Add unified result reporting (JSON + plots) for pipeline runs

**Files:**
- Modify: `pyimgano/utils/advanced_viz.py` (reuse existing)
- Create: `pyimgano/reporting/report.py`
- Test: `tests/test_reporting_json.py`

**Step 1: Write failing test**

Ensure report JSON schema contains both image and pixel metrics when provided.

**Step 2: Implement**

Add `save_run_report(path, results)` helper.

**Step 3: Commit**

`git commit -m "feat: add run reporting utilities"`

---

### Task 19: Add CLI entrypoint for quick benchmark on MVTec/VisA

**Files:**
- Create: `pyimgano/cli.py`
- Modify: `pyproject.toml` (console script)
- Docs: `README.md`

**Step 1: Implement minimal CLI**

Command:
- `pyimgano-benchmark --dataset mvtec --root ./mvtec_ad --category bottle --model patchcore`

**Step 2: Commit**

`git commit -m "feat: add pyimgano-benchmark CLI"`

---

### Task 20: Verification + merge to main

**Files:**
- N/A

**Step 1: Run fast test suite**

Run: `pytest -m "not slow" -v`

**Step 2: Optional slow tests**

Run: `pytest -m "slow" -v`

**Step 3: Merge + push**

```bash
cd /Users/luke/code/pyimgano
git checkout main
git pull --ff-only
git merge --ff-only mvtec-visa-hybrid
git push
```

