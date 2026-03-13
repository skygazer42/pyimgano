# Workbench Model Compatibility Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce duplicated model capability logic in workbench modules by introducing a shared model-compatibility boundary used by both runtime guardrails and preflight validation.

**Architecture:** Add a `pyimgano.workbench.model_compatibility` helper that centralizes two concerns: loading model capability summaries from the registry and enumerating pixel-map-dependent workbench options from config. Refactor `runtime_guardrails.py` and `preflight.py` to depend on this helper instead of importing capability machinery directly.

**Tech Stack:** Python, dataclasses, pytest, existing model registry/capabilities helpers, `WorkbenchConfig`.

---

### Task 1: Lock The Shared Compatibility Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_model_compatibility.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the shared helper reports enabled pixel-map-dependent options in stable order
- the shared helper exposes model capability summary needed by workbench modules
- `runtime_guardrails.py` and `preflight.py` no longer import registry capability helpers directly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_model_compatibility.py tests/test_architecture_boundaries.py -k "model_compatibility or runtime_guardrails_uses_model_compatibility_boundary or preflight_uses_model_compatibility_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.model_compatibility` does not exist and the existing modules still own capability lookups directly.

### Task 2: Add The Shared Compatibility Helper And Refactor Callers

**Files:**
- Create: `pyimgano/workbench/model_compatibility.py`
- Modify: `pyimgano/workbench/runtime_guardrails.py`
- Modify: `pyimgano/workbench/preflight.py`

**Step 1: Write minimal implementation**

- add a small dataclass for workbench model capability summary
- add a helper that loads capability summary from the model registry
- add a helper that enumerates pixel-map-dependent workbench options from config
- refactor runtime guardrails and preflight to use these shared helpers

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_model_compatibility.py tests/test_workbench_runtime_guardrails.py tests/test_workbench_preflight_preprocessing.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/model_compatibility.py`
- Modify: `pyimgano/workbench/runtime_guardrails.py`
- Modify: `pyimgano/workbench/preflight.py`
- Modify: `tests/test_workbench_model_compatibility.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_model_compatibility.py tests/test_workbench_runtime_guardrails.py tests/test_workbench_runner_pixel_map_requirements.py tests/test_workbench_preflight_preprocessing.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-model-compatibility-boundary.md pyimgano/workbench/model_compatibility.py pyimgano/workbench/runtime_guardrails.py pyimgano/workbench/preflight.py tests/test_workbench_model_compatibility.py tests/test_architecture_boundaries.py
```

Expected: no output
