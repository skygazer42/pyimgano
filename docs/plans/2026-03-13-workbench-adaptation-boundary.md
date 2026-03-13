# Workbench Adaptation Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `pyimgano.workbench.adaptation` into coherent type and runtime boundaries while preserving the existing public imports used by recipes, tests, and services.

**Architecture:** Move adaptation dataclasses into `pyimgano.workbench.adaptation_types` and runtime builder functions into `pyimgano.workbench.adaptation_runtime`. Keep `pyimgano.workbench.adaptation` as a thin compatibility facade that re-exports the public config types and compatibility helper functions, so internal modules can depend on narrower boundaries without breaking downstream imports.

**Tech Stack:** Python dataclasses, pytest, string-based architecture boundary tests.

---

### Task 1: Lock The Adaptation Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_adaptation_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated runtime module hosts tiling and postprocess builders
- `pyimgano.workbench.adaptation` behaves like a compatibility facade instead of defining dataclasses and runtime logic inline
- `config_parser`, `detector_setup`, and `category_execution` use the narrower adaptation boundaries

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_adaptation_runtime.py tests/test_architecture_boundaries.py -k "adaptation_runtime or adaptation_module_uses_boundaries or adaptation_boundary" -v
```

Expected: FAIL because the dedicated adaptation boundary modules do not exist yet and the facade/runtime call sites still point at the monolithic `adaptation.py`.

### Task 2: Extract Adaptation Types And Runtime Modules

**Files:**
- Create: `pyimgano/workbench/adaptation_types.py`
- Create: `pyimgano/workbench/adaptation_runtime.py`
- Modify: `pyimgano/workbench/adaptation.py`
- Modify: `pyimgano/workbench/config_parser.py`
- Modify: `pyimgano/workbench/detector_setup.py`
- Modify: `pyimgano/workbench/category_execution.py`

**Step 1: Write minimal implementation**

- move `TilingConfig`, `MapPostprocessConfig`, and `AdaptationConfig` into `adaptation_types.py`
- move `apply_tiling(...)` and `build_postprocess(...)` into `adaptation_runtime.py`
- keep `pyimgano.workbench.adaptation` exporting the same public names as a compatibility facade
- update internal call sites to import from `adaptation_types` or `adaptation_runtime` directly where appropriate

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_adaptation_runtime.py tests/test_workbench_tiling_integration.py tests/test_workbench_postprocess_integration.py tests/test_workbench_detector_setup.py tests/test_workbench_category_execution.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/adaptation.py`
- Modify: `pyimgano/workbench/adaptation_types.py`
- Modify: `pyimgano/workbench/adaptation_runtime.py`
- Modify: `pyimgano/workbench/config_parser.py`
- Modify: `pyimgano/workbench/detector_setup.py`
- Modify: `pyimgano/workbench/category_execution.py`
- Modify: `tests/test_workbench_adaptation_runtime.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_adaptation_runtime.py tests/test_workbench_tiling_integration.py tests/test_workbench_postprocess_integration.py tests/test_workbench_adaptation_service.py tests/test_workbench_adaptation_config.py tests/test_workbench_detector_setup.py tests/test_workbench_category_execution.py tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-adaptation-boundary.md pyimgano/workbench/adaptation.py pyimgano/workbench/adaptation_types.py pyimgano/workbench/adaptation_runtime.py pyimgano/workbench/config_parser.py pyimgano/workbench/detector_setup.py pyimgano/workbench/category_execution.py tests/test_workbench_adaptation_runtime.py tests/test_architecture_boundaries.py
```

Expected: no output
