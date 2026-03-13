# Workbench Config Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `pyimgano.workbench.config` into coherent type and parsing boundaries while preserving the existing public import path and `WorkbenchConfig.from_dict(...)` contract.

**Architecture:** Move dataclass declarations into `pyimgano.workbench.config_types` and parsing/validation logic into `pyimgano.workbench.config_parser`. Keep `pyimgano.workbench.config` as a thin compatibility facade that re-exports the public config types, so the package boundary becomes clearer without forcing downstream import changes.

**Tech Stack:** Python dataclasses, pytest, AST/string-based architecture boundary tests.

---

### Task 1: Lock The Config Boundary With Failing Tests

**Files:**
- Create: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- `WorkbenchConfig.from_dict(...)` delegates through a dedicated parser boundary
- `pyimgano.workbench.config` behaves like a compatibility facade rather than hosting inline parser helpers
- the parser helper exposes the same config object graph and preserves critical validation behavior

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -k "config_parser or config_module_uses_config_boundaries" -v
```

Expected: FAIL because `config.py` still contains inline helpers/parser logic and `pyimgano.workbench.config_parser` does not exist.

### Task 2: Extract Config Types And Parser Modules

**Files:**
- Create: `pyimgano/workbench/config_types.py`
- Create: `pyimgano/workbench/config_parser.py`
- Modify: `pyimgano/workbench/config.py`

**Step 1: Write minimal implementation**

- move workbench config dataclasses into `config_types.py`
- add parser helpers and `build_workbench_config_from_dict(...)` in `config_parser.py`
- keep `WorkbenchConfig.from_dict(...)` available and behavior-compatible
- keep `from pyimgano.workbench.config import WorkbenchConfig` and sibling type imports working

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_workbench_config.py tests/test_workbench_training_config.py tests/test_workbench_adaptation_config.py tests/test_workbench_preprocessing_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/workbench/config_types.py`
- Modify: `pyimgano/workbench/config_parser.py`
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_config.py tests/test_workbench_config_parser.py tests/test_workbench_preprocessing_config.py tests/test_workbench_training_config.py tests/test_workbench_adaptation_config.py tests/test_workbench_runtime_guardrails.py tests/test_workbench_export_infer_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-config-boundary.md pyimgano/workbench/config.py pyimgano/workbench/config_types.py pyimgano/workbench/config_parser.py tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py
```

Expected: no output
