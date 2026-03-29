# Deploy Bundle Validation Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden `pyimgano.reporting.deploy_bundle` by making deploy-bundle validation rules easier to audit and less drift-prone while preserving the current schema and validation outcomes.

**Architecture:** Keep `validate_deploy_bundle_manifest(...)` as the top-level validator shell, but extract smaller helpers for entry/ref/role validation, required flag validation, operator-contract validation, and weight-audit validation. Reuse existing computed helpers from the build side where they improve clarity without changing semantics.

**Tech Stack:** Python 3.9+, JSON, existing `pyimgano.reporting.deploy_bundle`, existing deploy-bundle tests, pytest.

---

### Task 1: Add failing tests for deploy-bundle validation helper scaffolding

**Files:**
- Create: `tests/test_deploy_bundle_validation_helpers.py`
- Test: `pyimgano/reporting/deploy_bundle.py`

**Step 1: Write the failing test**

```python
def test_validate_required_presence_flag_matches_boolean_contract():
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_required_presence_flag
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.reporting.deploy_bundle_validation_helpers'`

**Step 3: Write minimal implementation**

```python
def validate_required_presence_flag(...):
    return []
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_deploy_bundle_validation_helpers.py pyimgano/reporting/deploy_bundle_validation_helpers.py
git commit -m "test: add deploy bundle validation helper scaffolding"
```

### Task 2: Extract required-flag and exact-mapping validation helpers

**Files:**
- Create: `pyimgano/reporting/deploy_bundle_validation_helpers.py`
- Modify: `pyimgano/reporting/deploy_bundle.py`
- Test: `tests/test_deploy_bundle_validation_helpers.py`
- Test: `tests/test_deploy_bundle_contract_v1.py`

**Step 1: Write the failing test**

```python
def test_validate_exact_mapping_rejects_non_mapping_and_mismatch():
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_exact_mapping
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_contract_v1.py -q`
Expected: FAIL because helper is missing or not wired

**Step 3: Write minimal implementation**

```python
def validate_exact_mapping(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_contract_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle_validation_helpers.py pyimgano/reporting/deploy_bundle.py tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_contract_v1.py
git commit -m "refactor: extract deploy bundle basic validation helpers"
```

### Task 3: Add failing tests for artifact ref/role validation helpers

**Files:**
- Modify: `tests/test_deploy_bundle_validation_helpers.py`
- Test: `pyimgano/reporting/deploy_bundle_validation_helpers.py`

**Step 1: Write the failing test**

```python
def test_validate_artifact_refs_requires_existing_manifest_entries(tmp_path):
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_artifact_refs
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def validate_artifact_refs(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_validation_helpers.py
git commit -m "test: cover deploy bundle artifact ref helpers"
```

### Task 4: Extract artifact ref/role validation from the top-level validator

**Files:**
- Modify: `pyimgano/reporting/deploy_bundle_validation_helpers.py`
- Modify: `pyimgano/reporting/deploy_bundle.py`
- Test: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_deploy_bundle_validation_helpers.py`

**Step 1: Write the failing test**

```python
def test_validate_deploy_bundle_manifest_keeps_artifact_ref_errors_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_validation_helpers.py -q`
Expected: FAIL because helper extraction changes artifact-ref error behavior

**Step 3: Write minimal implementation**

```python
errors.extend(validate_artifact_refs(...))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_validation_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle.py pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_validation_helpers.py
git commit -m "refactor: extract deploy bundle artifact validation helpers"
```

### Task 5: Add failing tests for operator-contract digest helper extraction

**Files:**
- Modify: `tests/test_deploy_bundle_validation_helpers.py`
- Modify: `tests/test_deploy_bundle_manifest.py`

**Step 1: Write the failing test**

```python
def test_validate_operator_contract_digests_preserves_source_unavailable_behavior(tmp_path):
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_operator_contract_digests
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def validate_operator_contract_digests(...):
    return []
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py
git commit -m "test: cover deploy bundle operator digest helpers"
```

### Task 6: Extract operator-contract validation helpers from the top-level validator

**Files:**
- Modify: `pyimgano/reporting/deploy_bundle_validation_helpers.py`
- Modify: `pyimgano/reporting/deploy_bundle.py`
- Test: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_run_quality.py`

**Step 1: Write the failing test**

```python
def test_run_quality_bundle_digest_signals_stay_stable_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_run_quality.py -q`
Expected: FAIL because helper extraction changes digest consistency behavior

**Step 3: Write minimal implementation**

```python
errors.extend(validate_operator_contract_digests(...))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_run_quality.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle.py pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py tests/test_run_quality.py
git commit -m "refactor: extract deploy bundle operator validation helpers"
```

### Task 7: Add failing tests for weight-audit validation helper extraction

**Files:**
- Modify: `tests/test_deploy_bundle_validation_helpers.py`
- Modify: `tests/test_deploy_bundle_manifest.py`

**Step 1: Write the failing test**

```python
def test_validate_weight_audit_files_reports_model_card_and_manifest_errors(tmp_path):
    from pyimgano.reporting.deploy_bundle_validation_helpers import validate_weight_audit_files
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def validate_weight_audit_files(...):
    return []
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py
git commit -m "test: cover deploy bundle weight audit helpers"
```

### Task 8: Move weight-audit validation behind helper functions

**Files:**
- Modify: `pyimgano/reporting/deploy_bundle_validation_helpers.py`
- Modify: `pyimgano/reporting/deploy_bundle.py`
- Test: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_run_acceptance_states_v1.py`

**Step 1: Write the failing test**

```python
def test_run_acceptance_bundle_ready_state_stays_stable_after_weight_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_run_acceptance_states_v1.py -q`
Expected: FAIL because helper extraction changes bundle readiness interpretation

**Step 3: Write minimal implementation**

```python
errors.extend(validate_weight_audit_files(...))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_run_acceptance_states_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/deploy_bundle.py pyimgano/reporting/deploy_bundle_validation_helpers.py tests/test_deploy_bundle_manifest.py tests/test_run_acceptance_states_v1.py
git commit -m "refactor: extract deploy bundle weight audit helpers"
```

### Task 9: Add failing architecture tests for deploy-bundle validation helpers

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/reporting/deploy_bundle_validation_helpers.py`

**Step 1: Write the failing test**

```python
def test_deploy_bundle_validation_helper_module_defines_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because the helper module is not covered

**Step 3: Write minimal implementation**

```python
__all__ = ["validate_required_presence_flag"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/reporting/deploy_bundle_validation_helpers.py
git commit -m "test: add deploy bundle validation helper boundary expectations"
```

### Task 10: Run the targeted deploy-bundle verification suite

**Files:**
- Verify only:
  - `tests/test_deploy_bundle_manifest.py`
  - `tests/test_deploy_bundle_contract_v1.py`
  - `tests/test_deploy_bundle_validation_helpers.py`
  - `tests/test_run_quality.py`
  - `tests/test_run_acceptance_states_v1.py`
  - `tests/test_runs_cli.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_deploy_bundle_manifest.py \
  tests/test_deploy_bundle_contract_v1.py \
  tests/test_deploy_bundle_validation_helpers.py \
  tests/test_run_quality.py \
  tests/test_run_acceptance_states_v1.py \
  tests/test_runs_cli.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify deploy bundle validation hardening suite"
```

### Task 11: Run static verification on touched deploy-bundle files

**Files:**
- Verify only:
  - `pyimgano/reporting/deploy_bundle.py`
  - `pyimgano/reporting/deploy_bundle_validation_helpers.py`
  - `tests/test_deploy_bundle_validation_helpers.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/reporting/deploy_bundle.py \
  pyimgano/reporting/deploy_bundle_validation_helpers.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/reporting/deploy_bundle.py \
  pyimgano/reporting/deploy_bundle_validation_helpers.py \
  tests/test_deploy_bundle_validation_helpers.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize deploy bundle validation hardening checks"
```
