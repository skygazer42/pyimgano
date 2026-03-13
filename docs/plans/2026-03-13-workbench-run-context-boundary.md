# Workbench Run Context Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.runner` run-directory setup responsibility by moving run initialization and metadata persistence into a dedicated workbench helper.

**Architecture:** Add a `pyimgano.workbench.run_context` helper that decides whether a run directory is needed, creates it, prepares standard subdirectories, and writes `environment.json` plus `config.json`. Keep top-level report persistence in `runner.py`; this helper only handles initial run bootstrapping.

**Tech Stack:** Python, dataclasses, pytest, existing reporting run-path helpers, environment collector, `WorkbenchConfig`.

---

### Task 1: Lock The Run Context Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_run_context.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- the helper initializes a run directory, creates standard subdirectories, and writes `config.json` plus `environment.json`
- the helper returns `None` when `output.save_run` is false
- `runner.py` no longer owns `build_workbench_run_dir_name(...)`, `ensure_run_dir(...)`, or `build_workbench_run_paths(...)` directly

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_run_context.py tests/test_architecture_boundaries.py -k "workbench_run_context or runner_uses_run_context_boundary" -v
```

Expected: FAIL because the helper module does not exist and `runner.py` still owns inline run initialization.

### Task 2: Add The Run Context Helper And Refactor Runner

**Files:**
- Create: `pyimgano/workbench/run_context.py`
- Modify: `pyimgano/workbench/runner.py`

**Step 1: Write minimal implementation**

- add a small dataclass for initialized run context
- add `initialize_workbench_run_context(config, recipe_name)`
- move run-directory creation, path preparation, environment/config persistence into the helper
- keep final `report.json` writes in `runner.py`

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_run_context.py tests/test_workbench_repro_provenance.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/run_context.py`
- Modify: `pyimgano/workbench/runner.py`
- Modify: `tests/test_workbench_run_context.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_run_context.py tests/test_workbench_repro_provenance.py tests/test_workbench_runner_smoke.py tests/test_workbench_manifest_smoke.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-run-context-boundary.md pyimgano/workbench/run_context.py pyimgano/workbench/runner.py tests/test_workbench_run_context.py tests/test_architecture_boundaries.py
```

Expected: no output
