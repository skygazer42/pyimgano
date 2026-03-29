# Workbench Preflight Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the orchestration layer of workbench preflight so manifest and non-manifest flows remain easier to reason about without changing preflight result shapes.

**Architecture:** Keep `pyimgano.workbench.preflight` as the top-level shell and `pyimgano.workbench.preflight_summary` as the dataset dispatch layer, but extract small orchestration helpers where manifest/non-manifest preflight modules still mix early-return handling, source resolution, and final branch report assembly. Prefer tightening tests and explicit helper seams over broad module churn.

**Tech Stack:** Python 3.9+, existing `pyimgano.workbench.preflight*`, `pyimgano.workbench.manifest_*`, `pyimgano.workbench.non_manifest_*`, pytest.

---

### Task 1: Add failing tests for preflight summary dispatch helpers

**Files:**
- Create: `tests/test_workbench_preflight_dispatch.py`
- Test: `pyimgano/workbench/preflight_summary.py`

**Step 1: Write the failing test**

```python
def test_resolve_preflight_dataset_dispatch_uses_manifest_branch():
    from pyimgano.workbench.preflight_dispatch import resolve_preflight_dataset_dispatch
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_dispatch.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.workbench.preflight_dispatch'`

**Step 3: Write minimal implementation**

```python
def resolve_preflight_dataset_dispatch(*, config):
    return "manifest"
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_dispatch.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_workbench_preflight_dispatch.py pyimgano/workbench/preflight_dispatch.py
git commit -m "test: add preflight dispatch scaffolding"
```

### Task 2: Extract dataset dispatch helper from `preflight_summary`

**Files:**
- Create: `pyimgano/workbench/preflight_dispatch.py`
- Modify: `pyimgano/workbench/preflight_summary.py`
- Test: `tests/test_workbench_preflight_dispatch.py`
- Test: `tests/test_workbench_preflight_manifest.py`
- Test: `tests/test_workbench_preflight_non_manifest.py`

**Step 1: Write the failing test**

```python
def test_resolve_workbench_preflight_summary_delegates_dataset_choice(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_dispatch.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py -q`
Expected: FAIL because `preflight_summary.py` still decides the dataset branch inline

**Step 3: Write minimal implementation**

```python
dataset_branch = preflight_dispatch.resolve_preflight_dataset_dispatch(config=config)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_dispatch.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/workbench/preflight_dispatch.py pyimgano/workbench/preflight_summary.py tests/test_workbench_preflight_dispatch.py tests/test_workbench_preflight_manifest.py tests/test_workbench_preflight_non_manifest.py
git commit -m "refactor: extract workbench preflight dispatch helper"
```

### Task 3: Add failing tests for manifest preflight early-return helpers

**Files:**
- Create: `tests/test_workbench_manifest_preflight_flow.py`
- Test: `pyimgano/workbench/manifest_preflight.py`

**Step 1: Write the failing test**

```python
def test_resolve_manifest_preflight_source_or_summary_returns_summary_early():
    from pyimgano.workbench.manifest_preflight_flow import resolve_manifest_preflight_source_or_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_manifest_preflight_flow.py -q`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
def resolve_manifest_preflight_source_or_summary(...):
    return {"summary": None}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_manifest_preflight_flow.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_workbench_manifest_preflight_flow.py pyimgano/workbench/manifest_preflight_flow.py
git commit -m "test: add manifest preflight flow scaffolding"
```

### Task 4: Extract manifest source/record early-return helpers

**Files:**
- Create: `pyimgano/workbench/manifest_preflight_flow.py`
- Modify: `pyimgano/workbench/manifest_preflight.py`
- Test: `tests/test_workbench_manifest_preflight_flow.py`
- Test: `tests/test_workbench_manifest_preflight.py`

**Step 1: Write the failing test**

```python
def test_run_manifest_preflight_uses_flow_helpers_for_source_and_records(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_manifest_preflight_flow.py tests/test_workbench_manifest_preflight.py -q`
Expected: FAIL because `manifest_preflight.py` still handles early-return flow inline

**Step 3: Write minimal implementation**

```python
source = manifest_preflight_flow.resolve_manifest_preflight_source_or_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_manifest_preflight_flow.py tests/test_workbench_manifest_preflight.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/workbench/manifest_preflight.py pyimgano/workbench/manifest_preflight_flow.py tests/test_workbench_manifest_preflight_flow.py tests/test_workbench_manifest_preflight.py
git commit -m "refactor: extract manifest preflight flow helpers"
```

### Task 5: Add failing tests for non-manifest preflight early-return helpers

**Files:**
- Create: `tests/test_workbench_non_manifest_preflight_flow.py`
- Test: `pyimgano/workbench/non_manifest_preflight.py`

**Step 1: Write the failing test**

```python
def test_resolve_non_manifest_preflight_source_or_summary_returns_summary_early():
    from pyimgano.workbench.non_manifest_preflight_flow import resolve_non_manifest_preflight_source_or_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_non_manifest_preflight_flow.py -q`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
def resolve_non_manifest_preflight_source_or_summary(...):
    return {"summary": None}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_non_manifest_preflight_flow.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_workbench_non_manifest_preflight_flow.py pyimgano/workbench/non_manifest_preflight_flow.py
git commit -m "test: add non-manifest preflight flow scaffolding"
```

### Task 6: Extract non-manifest source/listing early-return helpers

**Files:**
- Create: `pyimgano/workbench/non_manifest_preflight_flow.py`
- Modify: `pyimgano/workbench/non_manifest_preflight.py`
- Test: `tests/test_workbench_non_manifest_preflight_flow.py`
- Test: `tests/test_workbench_preflight_non_manifest.py`

**Step 1: Write the failing test**

```python
def test_run_non_manifest_preflight_uses_flow_helpers_for_source_and_listing(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_non_manifest_preflight_flow.py tests/test_workbench_preflight_non_manifest.py -q`
Expected: FAIL because `non_manifest_preflight.py` still handles early-return flow inline

**Step 3: Write minimal implementation**

```python
source = non_manifest_preflight_flow.resolve_non_manifest_preflight_source_or_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_non_manifest_preflight_flow.py tests/test_workbench_preflight_non_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/workbench/non_manifest_preflight.py pyimgano/workbench/non_manifest_preflight_flow.py tests/test_workbench_non_manifest_preflight_flow.py tests/test_workbench_preflight_non_manifest.py
git commit -m "refactor: extract non-manifest preflight flow helpers"
```

### Task 7: Add failing tests for top-level preflight orchestration delegation

**Files:**
- Modify: `tests/test_workbench_preflight_preprocessing.py`
- Modify: `tests/test_workbench_manifest_preflight.py`
- Test: `pyimgano/workbench/preflight.py`

**Step 1: Write the failing test**

```python
def test_run_preflight_uses_preflight_summary_dispatch_boundary(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_preprocessing.py tests/test_workbench_manifest_preflight.py -q`
Expected: FAIL because `preflight.py` still carries inline dispatch or summary assumptions

**Step 3: Write minimal implementation**

```python
summary = resolve_workbench_preflight_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_preprocessing.py tests/test_workbench_manifest_preflight.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/workbench/preflight.py tests/test_workbench_preflight_preprocessing.py tests/test_workbench_manifest_preflight.py
git commit -m "test: lock top-level preflight delegation"
```

### Task 8: Add failing architecture tests for new preflight helper modules

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/workbench/preflight_dispatch.py`
- Test: `pyimgano/workbench/manifest_preflight_flow.py`
- Test: `pyimgano/workbench/non_manifest_preflight_flow.py`

**Step 1: Write the failing test**

```python
def test_preflight_helper_modules_define_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because new helper modules are not covered

**Step 3: Write minimal implementation**

```python
__all__ = ["resolve_preflight_dataset_dispatch"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/workbench/preflight_dispatch.py pyimgano/workbench/manifest_preflight_flow.py pyimgano/workbench/non_manifest_preflight_flow.py
git commit -m "test: add preflight helper boundary expectations"
```

### Task 9: Add failing docs contract tests for workbench preflight

**Files:**
- Create: `tests/test_workbench_preflight_docs_contract.py`
- Test: `docs/CLI_REFERENCE.md`

**Step 1: Write the failing test**

```python
def test_cli_reference_documents_train_preflight_output_contract():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_docs_contract.py -q`
Expected: FAIL because docs contract checks do not exist yet

**Step 3: Write minimal implementation**

```python
assert "--preflight" in text
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_workbench_preflight_docs_contract.py
git commit -m "test: add preflight docs contract checks"
```

### Task 10: Update docs for preflight orchestration contract wording

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Test: `tests/test_workbench_preflight_docs_contract.py`

**Step 1: Write the failing test**

```python
def test_cli_reference_mentions_preflight_json_shape_and_issue_reporting():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_docs_contract.py -q`
Expected: FAIL because docs wording is not locked yet

**Step 3: Write minimal implementation**

```markdown
- `--preflight` prints `{"preflight": ...}` and preserves issue severity/details.
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_preflight_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/CLI_REFERENCE.md tests/test_workbench_preflight_docs_contract.py
git commit -m "docs: clarify workbench preflight contract"
```

### Task 11: Run the targeted preflight verification suite

**Files:**
- Verify only:
  - `tests/test_workbench_manifest_preflight.py`
  - `tests/test_workbench_preflight_manifest.py`
  - `tests/test_workbench_preflight_non_manifest.py`
  - `tests/test_workbench_manifest_preflight_components.py`
  - `tests/test_workbench_preflight_preprocessing.py`
  - `tests/test_workbench_preflight_docs_contract.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_workbench_manifest_preflight.py \
  tests/test_workbench_preflight_manifest.py \
  tests/test_workbench_preflight_non_manifest.py \
  tests/test_workbench_manifest_preflight_components.py \
  tests/test_workbench_preflight_preprocessing.py \
  tests/test_workbench_preflight_docs_contract.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify workbench preflight hardening suite"
```

### Task 12: Run static verification on touched preflight files

**Files:**
- Verify only:
  - `pyimgano/workbench/preflight.py`
  - `pyimgano/workbench/preflight_summary.py`
  - `pyimgano/workbench/manifest_preflight.py`
  - `pyimgano/workbench/non_manifest_preflight.py`
  - `pyimgano/workbench/preflight_dispatch.py`
  - `pyimgano/workbench/manifest_preflight_flow.py`
  - `pyimgano/workbench/non_manifest_preflight_flow.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/workbench/preflight.py \
  pyimgano/workbench/preflight_summary.py \
  pyimgano/workbench/manifest_preflight.py \
  pyimgano/workbench/non_manifest_preflight.py \
  pyimgano/workbench/preflight_dispatch.py \
  pyimgano/workbench/manifest_preflight_flow.py \
  pyimgano/workbench/non_manifest_preflight_flow.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/workbench/preflight.py \
  pyimgano/workbench/preflight_summary.py \
  pyimgano/workbench/manifest_preflight.py \
  pyimgano/workbench/non_manifest_preflight.py \
  pyimgano/workbench/preflight_dispatch.py \
  pyimgano/workbench/manifest_preflight_flow.py \
  pyimgano/workbench/non_manifest_preflight_flow.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize workbench preflight hardening checks"
```
