# Model Failure Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix concrete model training/inference defects surfaced by the GPU validation sweep so more registry models complete real `fit -> save weights -> infer` successfully.

**Architecture:** Start with failures that look like real library bugs rather than environment or checkpoint gating: broken input normalization, broken registry wrappers, and missing train-path implementations. Add narrow regression tests for each confirmed root cause, then implement minimal fixes in the affected model modules or shared input utilities. Re-run the targeted validation commands after each fix group.

**Tech Stack:** Python, pytest, PyTorch, NumPy, OpenCV, existing pyimgano model registry.

---

### Task 1: Reproduce broken train paths

**Files:**
- Test: `tests/test_cutpaste.py`
- Test: `tests/test_differnet.py`
- Test: `tests/test_memseg.py`
- Test: `tests/test_riad.py`
- Test: `tests/test_ae1svm.py`
- Test: `tests/test_one_svm_cnn.py`

**Step 1: Write failing tests**
- Add regression tests proving the affected models accept image batches in the same forms used elsewhere in the repo.
- Add a regression test proving the `vision_ae1svm` registry path can actually fit and score.
- Add a regression test proving `one_class_cnn` is not exposed as a trainable registry detector unless it implements the detector contract.

**Step 2: Run tests to verify they fail**
- Run: `pytest tests/test_cutpaste.py tests/test_differnet.py tests/test_memseg.py tests/test_riad.py tests/test_ae1svm.py tests/test_one_svm_cnn.py -q`
- Expected: failures matching the currently observed validation errors.

### Task 2: Fix image-batch normalization

**Files:**
- Modify: `pyimgano/models/cutpaste.py`
- Modify: `pyimgano/models/differnet.py`
- Modify: `pyimgano/models/memseg.py`
- Modify: `pyimgano/models/riad.py`
- Possibly modify: shared image loading helper under `pyimgano/models/` or `pyimgano/utils/`
- Test: `tests/test_cutpaste.py`
- Test: `tests/test_differnet.py`
- Test: `tests/test_memseg.py`
- Test: `tests/test_riad.py`

**Step 1: Implement minimal normalization fixes**
- Accept list/tuple/stacked-array image batches consistently.
- Preserve existing path-based behavior.
- Avoid changing unrelated model semantics.

**Step 2: Run focused tests**
- Run: `pytest tests/test_cutpaste.py tests/test_differnet.py tests/test_memseg.py tests/test_riad.py -q`
- Expected: pass.

### Task 3: Fix broken registry model entries

**Files:**
- Modify: `pyimgano/models/ae1svm.py`
- Modify: `pyimgano/models/one_svm_cnn.py`
- Test: `tests/test_ae1svm.py`
- Test: `tests/test_one_svm_cnn.py`

**Step 1: Implement minimal fix**
- Either make `vision_ae1svm` satisfy the detector contract, or adjust its registry exposure if the current wrapper is invalid.
- Either make `one_class_cnn` satisfy the trainable detector contract, or stop exposing it as a trainable detector entry.

**Step 2: Run focused tests**
- Run: `pytest tests/test_ae1svm.py tests/test_one_svm_cnn.py -q`
- Expected: pass.

### Task 4: Re-run targeted GPU validation

**Files:**
- Validate only; no new files required beyond existing result artifacts.

**Step 1: Re-run affected models**
- Run the real GPU validation harness only for the models fixed above.
- Confirm `fit -> save weights/detector -> infer` succeeds.

**Step 2: Record outcome**
- Update the user with exactly which models moved from failed to passed, and which remain blocked by design.
