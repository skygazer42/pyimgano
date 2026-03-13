# Workbench Config Parser Layering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pyimgano.workbench.config_parser` complexity by separating scalar parse helpers and section-specific config parsing from the top-level config assembly flow.

**Architecture:** Move low-level coercion/validation helpers into `pyimgano.workbench.config_parse_primitives` and section-specific parsing logic into `pyimgano.workbench.config_section_parsers`. Keep `pyimgano.workbench.config_parser.build_workbench_config_from_dict(...)` as the public top-level assembler, so callers keep the same import while the internal parser layers become narrower and more coherent.

**Tech Stack:** Python, pytest, dataclass config types, string-based architecture boundary tests.

---

### Task 1: Lock The Config Parser Layers With Failing Tests

**Files:**
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

Add tests that prove:
- a dedicated primitives module hosts low-level parse helpers like `_require_mapping(...)` and `_parse_resize(...)`
- a dedicated section parser module hosts dataset/model/output/adaptation/preprocessing/training/defects parsing
- `config_parser.py` acts as the top-level assembler instead of defining those helper implementations inline

**Step 2: Run test to verify it fails**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py -k "config_parser or config_parse_primitives or config_section_parsers" -v
```

Expected: FAIL because the dedicated parser-layer modules do not exist yet and `config_parser.py` still hosts the helper implementations inline.

### Task 2: Extract Parser Primitive And Section Modules

**Files:**
- Create: `pyimgano/workbench/config_parse_primitives.py`
- Create: `pyimgano/workbench/config_section_parsers.py`
- Modify: `pyimgano/workbench/config_parser.py`

**Step 1: Write minimal implementation**

- move scalar parsing helpers into `config_parse_primitives.py`
- move section-specific parsers into `config_section_parsers.py`
- keep `build_workbench_config_from_dict(...)` in `config_parser.py` as the top-level assembler
- preserve current validation behavior and error messages

**Step 2: Run focused tests to verify they pass**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_workbench_config.py tests/test_workbench_preprocessing_config.py tests/test_workbench_training_config.py tests/test_workbench_adaptation_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

### Task 3: Regression Sweep

**Files:**
- Modify: `pyimgano/workbench/config_parser.py`
- Modify: `pyimgano/workbench/config_parse_primitives.py`
- Modify: `pyimgano/workbench/config_section_parsers.py`
- Modify: `tests/test_workbench_config_parser.py`
- Modify: `tests/test_architecture_boundaries.py`

**Step 1: Run regression suite**

Run:

```bash
pytest --no-cov tests/test_workbench_config_parser.py tests/test_workbench_config.py tests/test_workbench_preprocessing_config.py tests/test_workbench_training_config.py tests/test_workbench_adaptation_config.py tests/test_workbench_runtime_guardrails.py tests/test_workbench_export_infer_config.py tests/test_architecture_boundaries.py -v
```

Expected: PASS

**Step 2: Check patch hygiene**

Run:

```bash
git diff --check -- docs/plans/2026-03-13-workbench-config-parser-layering.md pyimgano/workbench/config_parser.py pyimgano/workbench/config_parse_primitives.py pyimgano/workbench/config_section_parsers.py tests/test_workbench_config_parser.py tests/test_architecture_boundaries.py
```

Expected: no output
