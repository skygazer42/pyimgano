# Workbench Infer Config Payload Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by moving infer-config payload construction into a dedicated helper while preserving the public compatibility function.

**Architecture:** Add a `pyimgano.workbench.infer_config_payload` helper that delegates to the existing service boundary. Keep `runner.py` exporting `build_infer_config_payload(...)` as a thin compatibility shim so existing callers continue to work unchanged, but remove its direct dependency on `pyimgano.services.workbench_service`.

**Tech Stack:** Python, pytest, existing workbench service boundary, `WorkbenchConfig`.

---

### Task 1: Lock The Infer Config Payload Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_infer_config_payload.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated helper delegates infer-config payload construction to the workbench service boundary
- `runner.py` uses the helper instead of importing `pyimgano.services.workbench_service` directly
- the public `runner.build_infer_config_payload(...)` compatibility function still delegates through the new helper

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_infer_config_payload.py tests/test_architecture_boundaries.py -k "infer_config_payload or runner_uses_infer_config_payload_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.infer_config_payload` does not exist and `runner.py` still imports `pyimgano.services.workbench_service` directly.

### Task 2: Add The Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/infer_config_payload.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `build_workbench_infer_config_payload(config, report)`
- delegate to `pyimgano.services.workbench_service.build_infer_config_payload(...)`
- remove direct service import from `runner.py`
- keep `runner.build_infer_config_payload(...)` as a compatibility shim that calls the helper

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_infer_config_payload.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/infer_config_payload.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_infer_config_payload.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_infer_config_payload.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_workbench_export_infer_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-infer-config-payload-boundary.md pyimgano/workbench/infer_config_payload.py pyimgano/workbench/runner.py tests/test_workbench_infer_config_payload.py tests/test_architecture_boundaries.py
```

Expected: no output
