# Workbench Runtime Guardrails Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` responsibility by moving runtime guardrails and model capability validation into a dedicated workbench helper.

**Architecture:** Add a `pyimgano.workbench.runtime_guardrails` module with a single public entrypoint that validates save-run prerequisites and pixel-map-dependent feature compatibility. Keep `runner.py` as orchestration only; it should delegate guardrails before initializing run context or touching datasets.

**Tech Stack:** Python, pytest, existing `WorkbenchConfig`, model registry capabilities helpers.

---

### Task 1: Lock The Runtime Guardrails Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_runtime_guardrails.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `adaptation.save_maps` still requires `output.save_run=true`
- `training.enabled` still requires `output.save_run=true`
- pixel-map-dependent workbench features still reject non-pixel-map models with the existing message shape
- `runner.py` no longer owns registry capability lookups or inline save-run guardrails

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_guardrails.py tests/test_architecture_boundaries.py -k "runtime_guardrails or runner_uses_runtime_guardrails_boundary" -v
```

Expected: FAIL because `pyimgano.workbench.runtime_guardrails` does not exist and `runner.py` still contains inline guardrails.

### Task 2: Add The Runtime Guardrails Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/runtime_guardrails.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add `validate_workbench_runtime_guardrails(config=...)`
- move `output.save_run` prerequisites into that helper
- move pixel-map model capability validation into that helper
- preserve current `ValueError` messages and option detection

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_guardrails.py tests/test_workbench_runner_pixel_map_requirements.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/runtime_guardrails.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_runtime_guardrails.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_runtime_guardrails.py tests/test_workbench_runner_pixel_map_requirements.py tests/test_workbench_preflight_preprocessing.py tests/test_workbench_runner_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-runtime-guardrails-boundary.md pyimgano/workbench/runtime_guardrails.py pyimgano/workbench/runner.py tests/test_workbench_runtime_guardrails.py tests/test_architecture_boundaries.py
```

Expected: no output
