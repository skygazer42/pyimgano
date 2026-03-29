# Bundle CLI Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden `pyimgano-bundle validate` and `pyimgano-bundle run` by separating payload assembly, rendering, and exit-code handling without changing existing public behavior.

**Architecture:** Keep `pyimgano.bundle_cli` as the public CLI shell, but extract smaller helpers for validate and run summary assembly plus text rendering. Reuse existing schema/validation modules such as `pyimgano.reporting.deploy_bundle`, `pyimgano.inference.validate_infer_config`, and `pyimgano.weights.bundle_audit` rather than duplicating manifest or infer-config logic.

**Tech Stack:** Python 3.9+, argparse, JSON, existing `pyimgano.bundle_cli`, `pyimgano.reporting.deploy_bundle`, `pyimgano.inference.validate_infer_config`, `pyimgano.weights.bundle_audit`, pytest.

---

### Task 1: Add failing tests for bundle rendering helper scaffolding

**Files:**
- Create: `tests/test_bundle_rendering.py`
- Test: `pyimgano/bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_format_bundle_validate_summary_line():
    from pyimgano.bundle_rendering import format_bundle_validate_summary_line
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_rendering.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.bundle_rendering'`

**Step 3: Write minimal implementation**

```python
def format_bundle_validate_summary_line(...):
    return ""
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_bundle_rendering.py pyimgano/bundle_rendering.py
git commit -m "test: add bundle rendering scaffolding"
```

### Task 2: Extract validate/run text summary helpers

**Files:**
- Create: `pyimgano/bundle_rendering.py`
- Modify: `pyimgano/bundle_cli.py`
- Test: `tests/test_bundle_rendering.py`
- Test: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_bundle_cli_text_uses_validate_summary_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_rendering.py tests/test_bundle_cli.py -q`
Expected: FAIL because `bundle_cli.py` still formats summaries inline

**Step 3: Write minimal implementation**

```python
import pyimgano.bundle_rendering as bundle_rendering
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_rendering.py tests/test_bundle_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_rendering.py pyimgano/bundle_cli.py tests/test_bundle_rendering.py tests/test_bundle_cli.py
git commit -m "refactor: extract bundle rendering helpers"
```

### Task 3: Add failing tests for validate helper extraction

**Files:**
- Create: `tests/test_bundle_cli_helpers.py`
- Test: `pyimgano/bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_build_validate_reason_codes_uses_existing_reason_map():
    from pyimgano.bundle_cli_helpers import build_validate_reason_codes
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
def build_validate_reason_codes(...):
    return []
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_bundle_cli_helpers.py pyimgano/bundle_cli_helpers.py
git commit -m "test: add bundle cli helper scaffolding"
```

### Task 4: Move validate status/reason helper logic behind `bundle_cli_helpers`

**Files:**
- Create: `pyimgano/bundle_cli_helpers.py`
- Modify: `pyimgano/bundle_cli.py`
- Test: `tests/test_bundle_cli_helpers.py`
- Test: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_build_validate_status_handles_missing_manifest_and_invalid_weights():
    from pyimgano.bundle_cli_helpers import build_validate_status
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: FAIL because helper is missing or not wired

**Step 3: Write minimal implementation**

```python
def build_validate_status(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_cli_helpers.py pyimgano/bundle_cli.py tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py
git commit -m "refactor: extract bundle validate helpers"
```

### Task 5: Add failing tests for run batch-gate helper extraction

**Files:**
- Modify: `tests/test_bundle_cli_helpers.py`
- Modify: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_build_batch_gate_summary_reports_triggered_reasons():
    from pyimgano.bundle_cli_helpers import build_batch_gate_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: FAIL because the helper is missing

**Step 3: Write minimal implementation**

```python
def build_batch_gate_summary(...):
    return {"status": "pass"}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_cli_helpers.py tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py
git commit -m "test: cover bundle batch gate helpers"
```

### Task 6: Move run batch-gate summary logic behind helpers

**Files:**
- Modify: `pyimgano/bundle_cli_helpers.py`
- Modify: `pyimgano/bundle_cli.py`
- Test: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_bundle_cli_run_preserves_batch_gate_payload_shape_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli.py tests/test_bundle_cli_helpers.py -q`
Expected: FAIL because helper extraction changes run report assembly

**Step 3: Write minimal implementation**

```python
batch_gate = build_batch_gate_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli.py tests/test_bundle_cli_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_cli.py pyimgano/bundle_cli_helpers.py tests/test_bundle_cli.py
git commit -m "refactor: extract bundle batch gate helpers"
```

### Task 7: Add failing tests for input-source summary helper extraction

**Files:**
- Modify: `tests/test_bundle_cli_helpers.py`
- Modify: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_build_input_source_summary_reports_manifest_vs_image_dir():
    from pyimgano.bundle_cli_helpers import build_input_source_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def build_input_source_summary(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_cli_helpers.py tests/test_bundle_cli_helpers.py tests/test_bundle_cli.py
git commit -m "test: cover bundle input source helpers"
```

### Task 8: Extract run input-source summary assembly

**Files:**
- Modify: `pyimgano/bundle_cli_helpers.py`
- Modify: `pyimgano/bundle_cli.py`
- Test: `tests/test_bundle_cli.py`

**Step 1: Write the failing test**

```python
def test_bundle_cli_run_preserves_input_source_contract_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli.py tests/test_bundle_cli_helpers.py -q`
Expected: FAIL because helper extraction changes input-source payload shape

**Step 3: Write minimal implementation**

```python
input_summary = build_input_source_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_cli.py tests/test_bundle_cli_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/bundle_cli.py pyimgano/bundle_cli_helpers.py tests/test_bundle_cli.py
git commit -m "refactor: extract bundle input summary helpers"
```

### Task 9: Add failing tests for bundle docs contract

**Files:**
- Create: `tests/test_bundle_docs_contract.py`
- Test: `docs/CLI_REFERENCE.md`
- Test: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_reference_documents_bundle_validate_and_run():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_docs_contract.py -q`
Expected: FAIL because docs contract checks do not exist yet

**Step 3: Write minimal implementation**

```python
assert "pyimgano-bundle" in text
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_bundle_docs_contract.py
git commit -m "test: add bundle docs contract checks"
```

### Task 10: Update docs for bundle validate/run contract wording

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`
- Test: `tests/test_bundle_docs_contract.py`

**Step 1: Write the failing test**

```python
def test_readme_mentions_bundle_validate_and_weights_audit():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_bundle_docs_contract.py -q`
Expected: FAIL because docs wording is not locked yet

**Step 3: Write minimal implementation**

```markdown
pyimgano-bundle validate ...
pyimgano-bundle run ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_bundle_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/CLI_REFERENCE.md README.md tests/test_bundle_docs_contract.py
git commit -m "docs: clarify bundle validate and run contracts"
```

### Task 11: Add failing architecture tests for bundle helper modules

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/bundle_rendering.py`
- Test: `pyimgano/bundle_cli_helpers.py`

**Step 1: Write the failing test**

```python
def test_bundle_helper_modules_define_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because new helper modules are not covered

**Step 3: Write minimal implementation**

```python
__all__ = ["format_bundle_validate_summary_line"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/bundle_rendering.py pyimgano/bundle_cli_helpers.py
git commit -m "test: add bundle helper boundary expectations"
```

### Task 12: Run the targeted bundle verification suite

**Files:**
- Verify only:
  - `tests/test_bundle_cli.py`
  - `tests/test_bundle_rendering.py`
  - `tests/test_bundle_cli_helpers.py`
  - `tests/test_bundle_docs_contract.py`
  - `tests/test_deploy_bundle_manifest.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_bundle_cli.py \
  tests/test_bundle_rendering.py \
  tests/test_bundle_cli_helpers.py \
  tests/test_bundle_docs_contract.py \
  tests/test_deploy_bundle_manifest.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify bundle hardening suite"
```

### Task 13: Run static verification on touched bundle files

**Files:**
- Verify only:
  - `pyimgano/bundle_cli.py`
  - `pyimgano/bundle_rendering.py`
  - `pyimgano/bundle_cli_helpers.py`
  - `pyimgano/reporting/deploy_bundle.py`
  - `tests/test_bundle_rendering.py`
  - `tests/test_bundle_cli_helpers.py`
  - `tests/test_bundle_docs_contract.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/bundle_cli.py \
  pyimgano/bundle_rendering.py \
  pyimgano/bundle_cli_helpers.py \
  pyimgano/reporting/deploy_bundle.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/bundle_cli.py \
  pyimgano/bundle_rendering.py \
  pyimgano/bundle_cli_helpers.py \
  pyimgano/reporting/deploy_bundle.py \
  tests/test_bundle_rendering.py \
  tests/test_bundle_cli_helpers.py \
  tests/test_bundle_docs_contract.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize bundle hardening checks"
```
