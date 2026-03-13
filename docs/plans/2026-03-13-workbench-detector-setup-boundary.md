# Workbench Detector Setup Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` detector-assembly coupling by moving detector creation, tiling, and optional preprocessing wrapping into a dedicated setup helper.

**Architecture:** Add a `pyimgano.workbench.detector_setup` module that builds the runtime detector from `WorkbenchConfig`. Move `create_workbench_detector(...)`, `apply_tiling(...)`, and `PreprocessingDetector(...)` wiring into this helper while keeping pixel-map capability checks in `runner.py`.

**Tech Stack:** Python, pytest, existing workbench service, tiling adapter, preprocessing wrapper.

---

### Task 1: Lock The Detector Setup Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_detector_setup.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper builds detector -> tiling -> preprocessing in order when preprocessing is configured
- the helper skips preprocessing wrapping when no preprocessing config is present
- `runner.py` no longer owns `_create_detector`, `apply_tiling(...)`, or `PreprocessingDetector(...)` directly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_detector_setup.py tests/test_architecture_boundaries.py -k "workbench_detector_setup or runner_uses_detector_setup_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns detector assembly inline.

### Task 2: Add The Detector Setup Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/detector_setup.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `build_workbench_runtime_detector(config)`
- move detector creation, tiling, and optional preprocessing wrapper construction into the helper
- keep downstream training/inference flow unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_detector_setup.py tests/test_integration_workbench_train_then_infer.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/detector_setup.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_detector_setup.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_detector_setup.py tests/test_integration_workbench_train_then_infer.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-detector-setup-boundary.md pyimgano/workbench/detector_setup.py pyimgano/workbench/runner.py tests/test_workbench_detector_setup.py tests/test_architecture_boundaries.py
```

Expected: no output
