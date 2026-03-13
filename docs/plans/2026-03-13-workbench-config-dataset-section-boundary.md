# Workbench Config Dataset Section Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.config_section_parsers` size and coupling by moving dataset and split-policy parsing into a dedicated helper boundary while preserving existing imports and validation behavior.

**Architecture:** Keep `pyimgano.workbench.config_section_parsers` as the compatibility facade that exposes section parser names used by `config_parser.py` and tests. Add `pyimgano.workbench.config_dataset_section_parser` to own `_parse_split_policy_config(...)` and `_parse_dataset_config(...)`, so dataset-specific parsing rules and manifest dataset validation live in a narrower boundary.

**Tech Stack:** Python, pytest, dataclasses, string-based architecture boundary tests.

---

### Task 1: Lock The Dataset Section Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new `config_dataset_section_parser` module exposes `_parse_split_policy_config(...)` and `_parse_dataset_config(...)`
- `config_section_parsers.py` imports the new helper module
- `config_section_parsers.py` no longer defines `_parse_split_policy_config(...)` or `_parse_dataset_config(...)` inline
- the new helper module hosts the manifest dataset validation and split-policy parsing logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -k "config_dataset_section_parser or config_section_parsers_uses_dataset_section_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `config_section_parsers.py` still defines those functions inline.

### Task 2: Add The Dataset Section Helper And Re-Export Through The Facade

**Files:**
- Create: `pyimgano/workbench/config_dataset_section_parser.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`

**Step 1: Write minimal implementation**

- move `_parse_split_policy_config(...)` and `_parse_dataset_config(...)` to the new helper module
- import and re-export them from `config_section_parsers.py`
- keep current validation messages and behavior-compatible imports

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/config_dataset_section_parser.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_workbench_package.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-config-dataset-section-boundary.md pyimgano/workbench/config_dataset_section_parser.py pyimgano/workbench/config_section_parsers.py tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py
```

Expected: no output
