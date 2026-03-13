# Workbench Config Adaptation Section Boundary Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Continue reducing `pyimgano.workbench.config_section_parsers` size by moving adaptation parsing into a dedicated helper boundary while preserving existing imports and validation behavior.

**Architecture:** Keep `pyimgano.workbench.config_section_parsers` as the compatibility facade that still exports `_parse_adaptation_config(...)` for `config_parser.py` and any existing callers. Add `pyimgano.workbench.config_adaptation_section_parser` to own tiling and postprocess config parsing so adaptation rules live next to the adaptation type boundary instead of remaining inside the broader section facade.

**Tech Stack:** Python, pytest, dataclasses, string-based architecture boundary tests.

---

### Task 1: Lock The Adaptation Section Boundary With Failing Tests

**Files:**
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a new `config_adaptation_section_parser` module exposes `_parse_adaptation_config(...)`
- `config_section_parsers.py` imports the new helper module
- `config_section_parsers.py` no longer defines `_parse_adaptation_config(...)` inline
- the new helper module hosts the tiling and postprocess parsing logic

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -k "config_adaptation_section_parser or config_section_parsers_uses_adaptation_section_boundary" -v
```

Expected: FAIL because the helper module does not exist yet and `config_section_parsers.py` still defines `_parse_adaptation_config(...)` inline.

### Task 2: Add The Adaptation Section Helper And Re-Export Through The Facade

**Files:**
- Create: `pyimgano/workbench/config_adaptation_section_parser.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`

**Step 1: Write minimal implementation**

- move `_parse_adaptation_config(...)` to the new helper module
- import and re-export it from `config_section_parsers.py`
- keep current validation messages and defaults unchanged

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py tests/test_workbench_adaptation_config.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Create: `pyimgano/workbench/config_adaptation_section_parser.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_workbench_adaptation_config.py tests/test_workbench_preprocessing_config.py tests/test_workbench_package.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-config-adaptation-section-boundary.md pyimgano/workbench/config_adaptation_section_parser.py pyimgano/workbench/config_section_parsers.py tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py
```

Expected: no output
