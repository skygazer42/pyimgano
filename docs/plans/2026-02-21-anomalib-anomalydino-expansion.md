# anomalib Backend Expansion + AnomalyDINO PoC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add many more algorithms to `pyimgano` quickly via optional `anomalib` backend wrappers, and ship a foundation-style PoC model `vision_anomalydino` (DINOv2 patch kNN) with image-level + pixel-level outputs.

**Architecture:** Keep `pyimgano` as the stable API + datasets/eval/postprocess layer. Add (1) a generic anomalib checkpoint wrapper registered under multiple `vision_*_anomalib` aliases, and (2) a torch-lazy AnomalyDINO implementation whose core scoring is testable without torch via an injected embedder.

**Tech Stack:** Python, NumPy, OpenCV, scikit-learn; optional anomalib (`pyimgano[anomalib]`); optional FAISS (`pyimgano[faiss]`); DINOv2 via `torch.hub` (torch/torchvision already core deps).

---

### Task 1: Create a generic anomalib checkpoint wrapper

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Test: `tests/test_anomalib_backend_optional.py`

**Step 1: Write failing test**

Add a test asserting a generic wrapper exists in the registry and raises `ImportError` if anomalib is missing.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_anomalib_backend_optional.py -v`
Expected: FAIL (model name not registered yet)

**Step 3: Implement minimal code**

Implement `VisionAnomalibCheckpoint` with:
- `__init__(checkpoint_path, device="cpu", contamination=0.1, inferencer=None)`
- if `inferencer` is None: require anomalib + create `TorchInferencer`
- `fit(train_paths)` calibrates `threshold_` from `decision_function(train_paths)`
- `decision_function(paths)` returns continuous scores
- `predict(paths)` uses `threshold_` to return 0/1 labels

**Step 4: Re-run test (green)**

Expected: PASS.

**Step 5: Commit**

`git commit -m "feat: add generic anomalib checkpoint wrapper"`

---

### Task 2: Add alias model registrations for anomalib model families

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`

**Step 1: Add alias registrations**

Register multiple names (constructor functions) that all instantiate `VisionAnomalibCheckpoint`, e.g.:
- `vision_padim_anomalib`
- `vision_stfpm_anomalib`
- `vision_draem_anomalib`
- `vision_fastflow_anomalib`
- `vision_reverse_distillation_anomalib`
- `vision_dfm_anomalib`
- `vision_cflow_anomalib`

**Step 2: Commit**

`git commit -m "feat: add anomalib backend aliases"`

---

### Task 3: Make anomalib wrapper testable without anomalib (inferencer injection)

**Files:**
- Modify: `pyimgano/models/anomalib_backend.py`
- Create: `tests/test_anomalib_backend_wrapper.py`

**Step 1: Write failing tests**

Test `VisionAnomalibCheckpoint` with a fake inferencer:
- `.predict(path)` returns dict-like object containing `pred_score` and `anomaly_map`
- `fit()` sets `threshold_`
- `predict()` returns {0,1}
- `get_anomaly_map()` and `predict_anomaly_map()` return numpy arrays

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_anomalib_backend_wrapper.py -v`

**Step 3: Implement minimal code**

Add/adjust:
- robust `_extract_score_and_map` (dict/object)
- map conversion to numpy

**Step 4: Re-run tests**

Expected: PASS.

**Step 5: Commit**

`git commit -m "test: add anomalib wrapper tests (no anomalib required)"`

---

### Task 4: Implement AnomalyDINO core helpers (pure numpy)

**Files:**
- Create: `pyimgano/models/anomalydino.py`
- Test: `tests/test_anomalydino_core.py`

**Step 1: Write failing tests**

Add tests for:
- `topk_mean` aggregation behaves as expected
- anomaly-map reshape from patch vector to (grid_h, grid_w)

**Step 2: Run red**

Run: `pytest tests/test_anomalydino_core.py -v`

**Step 3: Implement minimal helpers**

Implement:
- `_aggregate_patch_scores(patch_scores, method="topk_mean", topk=0.01)`
- `_reshape_patch_scores(patch_scores, grid_h, grid_w)`

**Step 4: Run green**

**Step 5: Commit**

`git commit -m "feat: add anomalydino core score helpers"`

---

### Task 5: Implement `VisionAnomalyDINO` with injected embedder (no torch required for tests)

**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Test: `tests/test_anomalydino_core.py`

**Step 1: Write failing tests**

Use a fake embedder returning deterministic patch embeddings so we can test:
- `fit()` builds a memory bank + kNN index
- anomaly images yield higher `decision_function` scores than normals
- `get_anomaly_map()` returns correct shape and finite values

**Step 2: Run red**

Run: `pytest tests/test_anomalydino_core.py -v`

**Step 3: Implement minimal code**

Implement:
- `fit(paths)` builds memory bank and calibrates threshold
- `decision_function(paths)` scores using kNN distances
- `predict(paths)` labels via threshold
- `get_anomaly_map(path)` builds pixel map (resize to original)

**Step 4: Run green**

**Step 5: Commit**

`git commit -m "feat: add vision_anomalydino (embedder-injectable)"`

---

### Task 6: Add default DINOv2 embedder via torch.hub (lazy import)

**Files:**
- Modify: `pyimgano/models/anomalydino.py`

**Step 1: Implement**

Add `TorchHubDinoV2Embedder`:
- lazily imports `torch`/`torchvision` at runtime
- loads `dinov2_vits14` (configurable)
- exposes `embed(path) -> (patch_embeddings, grid_shape, original_size)`

**Step 2: Add parameters to VisionAnomalyDINO**

- `embedder=None` uses default torch-hub embedder
- `device`, `image_size`, `pretrained` flags

**Step 3: Commit**

`git commit -m "feat: add torch.hub dinov2 embedder for anomalydino"`

---

### Task 7: Register `vision_anomalydino` in the model registry auto-import

**Files:**
- Modify: `pyimgano/models/__init__.py`

**Step 1: Write failing test**

Add a simple import test ensuring module can be auto-imported even if torch isn’t present (must not crash on import).

**Step 2: Implement**

Add `"anomalydino"` to the `_auto_import([...])` list.

**Step 3: Commit**

`git commit -m "feat: register vision_anomalydino"`

---

### Task 8: Documentation for new algorithms

**Files:**
- Modify: `README.md`
- Modify: `docs/DEEP_LEARNING_MODELS.md`

**Step 1: Update docs**

Add:
- how to use `vision_anomalib_checkpoint`
- how to use `vision_*_anomalib` aliases
- how to use `vision_anomalydino`
- emphasize `decision_function` vs `predict`

**Step 2: Commit**

`git commit -m "docs: add anomalib backend + anomalydino usage"`

---

### Task 9: Pipeline example using AnomalyDINO

**Files:**
- Modify: `pyimgano/pipelines/mvtec_visa.py`
- Create: `examples/anomalydino_mvtec_visa.py`

**Step 1: Implement**

Add an example that:
- loads a split
- fits `vision_anomalydino`
- evaluates image + pixel metrics

**Step 2: Commit**

`git commit -m "feat: add anomalydino pipeline example"`

---

### Task 10: Hardening + edge cases

**Files:**
- Modify: `pyimgano/models/anomalydino.py`
- Modify: `pyimgano/models/anomalib_backend.py`

**Step 1: Add tests**

Cover:
- empty training set
- mismatched anomaly-map sizes (ensure resize works)
- contamination edge values (0 < contamination < 0.5)

**Step 2: Implement checks**

Raise clear `ValueError` with actionable messages.

**Step 3: Commit**

`git commit -m "fix: harden backend wrappers and anomalydino"`

---

### Task 11: Verification

**Files:**
- N/A

**Step 1: Run fast tests**

Run: `pytest -m \"not slow\" -v`

**Step 2: Compile check**

Run: `python -m compileall pyimgano tests examples docs`

---

### Task 12: Merge and push

**Files:**
- N/A

**Step 1: Merge**

```bash
git checkout main
git pull --ff-only
git merge --ff-only anomalib-anomalydino
git push
```

