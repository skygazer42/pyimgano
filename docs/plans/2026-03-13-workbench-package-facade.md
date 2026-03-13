# Workbench Package Facade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano.workbench` behave like a coherent package by exposing a stable lazy-loaded public facade instead of a partial eager import surface.

**Architecture:** Replace the current `pyimgano.workbench.__init__` implementation with a grouped lazy export map, following the existing package facade patterns used elsewhere in the repo. Export the core config types, dataset seam, main workflow entrypoints, and preflight contracts without eagerly importing heavy workflow modules at package import time.

**Tech Stack:** Python, importlib-based lazy exports, pytest subprocess import checks.

---

### Task 1: Lock The Workbench Package Facade With Failing Tests

**Files:**
- Create: `tests/test_workbench_package.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- importing `pyimgano.workbench` does not eagerly import `runner` or `preflight`
- the package exposes a stable public export list and grouped export source map
- resolving one public symbol does not eagerly import unrelated workbench workflow modules
- `workbench/__init__.py` uses a lazy export facade rather than direct `from .runner import ...` / `from .preflight import ...`

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_architecture_boundaries.py -k "workbench_package" -v
```

Expected: FAIL because `pyimgano.workbench.__init__` is still a partial eager import file and does not provide the grouped export facade.

### Task 2: Add The Lazy Package Facade

**Files:**
- Modify: `pyimgano/workbench/__init__.py`

**Step 1: Write minimal implementation**

- add grouped export metadata for config, dataset, preflight, and execution symbols
- build a stable `_WORKBENCH_EXPORT_SOURCES` map and `__all__`
- add `__getattr__` and `__dir__` lazy export behavior
- keep existing exports available while adding the main workflow/preflight package-level symbols

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/__init__.py`
- Modify: `tests/test_workbench_package.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_package.py tests/test_workbench_preprocessing_config.py tests/test_workbench_feature_pipeline.py tests/test_workbench_export_infer_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-package-facade.md pyimgano/workbench/__init__.py tests/test_workbench_package.py tests/test_architecture_boundaries.py
```

Expected: no output
