# Runs / Reporting Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the engineering quality of `pyimgano-runs` by making run comparison, trust/comparability normalization, and CLI rendering easier to reason about without changing any existing public behavior.

**Architecture:** Keep `pyimgano/reporting/run_index.py` as the source of structured run/comparison facts, but extract smaller helpers for trust, operator-contract, and comparability normalization. Keep `pyimgano/runs_cli.py` as the public entrypoint, but thin it into argument parsing plus rendering of already-structured payloads. Prefer additive helper modules and focused regression tests over a broad reporting rewrite.

**Tech Stack:** Python 3.9+, argparse, JSON, existing `pyimgano.reporting`, `pyimgano.runs_cli`, `pyimgano.reporting.run_quality`, `pyimgano.reporting.run_acceptance`, pytest.

---

### Task 1: Add failing tests for run-index trust/operator helper extraction

**Files:**
- Create: `tests/test_run_index_helpers.py`
- Test: `pyimgano/reporting/run_index.py`

**Step 1: Write the failing test**

```python
def test_build_trust_comparison_exposes_operator_contract_status():
    from pyimgano.reporting.run_index_helpers import build_trust_comparison
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.reporting.run_index_helpers'`

**Step 3: Write minimal implementation**

```python
def build_trust_comparison(...):
    return {"checked": False}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_run_index_helpers.py pyimgano/reporting/run_index_helpers.py
git commit -m "test: add run index helper scaffolding"
```

### Task 2: Extract trust/operator-contract normalization helpers from `run_index`

**Files:**
- Create: `pyimgano/reporting/run_index_helpers.py`
- Modify: `pyimgano/reporting/run_index.py`
- Test: `tests/test_run_index_helpers.py`

**Step 1: Write the failing test**

```python
def test_operator_contract_status_from_trust_summary_handles_missing_and_consistent():
    from pyimgano.reporting.run_index_helpers import operator_contract_status_from_trust_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: FAIL because the helper is not implemented

**Step 3: Write minimal implementation**

```python
def operator_contract_status_from_trust_summary(trust_summary):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py tests/test_run_index.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py pyimgano/reporting/run_index.py tests/test_run_index_helpers.py
git commit -m "refactor: extract run index trust helpers"
```

### Task 3: Add failing tests for comparability-gate status helpers

**Files:**
- Modify: `tests/test_run_index_helpers.py`
- Test: `pyimgano/reporting/run_index_helpers.py`

**Step 1: Write the failing test**

```python
def test_comparability_gate_status_reports_unchecked_and_incompatible():
    from pyimgano.reporting.run_index_helpers import comparability_gate_status
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: FAIL because `comparability_gate_status` is missing

**Step 3: Write minimal implementation**

```python
def comparability_gate_status(summary):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py tests/test_run_index_helpers.py
git commit -m "test: cover run comparability gate helpers"
```

### Task 4: Move comparability gate and blocking-flag logic behind helpers

**Files:**
- Modify: `pyimgano/reporting/run_index_helpers.py`
- Modify: `pyimgano/reporting/run_index.py`
- Test: `tests/test_run_index_helpers.py`
- Test: `tests/test_run_index.py`

**Step 1: Write the failing test**

```python
def test_compare_blocking_flags_uses_incompatible_gate_states():
    from pyimgano.reporting.run_index_helpers import compare_blocking_flags
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py tests/test_run_index.py -q`
Expected: FAIL because `compare_blocking_flags` is missing or not wired

**Step 3: Write minimal implementation**

```python
def compare_blocking_flags(...):
    return flags
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py tests/test_run_index.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py pyimgano/reporting/run_index.py tests/test_run_index_helpers.py tests/test_run_index.py
git commit -m "refactor: extract run comparability gate helpers"
```

### Task 5: Add failing tests for candidate digest/brief rendering inputs

**Files:**
- Modify: `tests/test_run_index_helpers.py`
- Test: `pyimgano/reporting/run_index_helpers.py`

**Step 1: Write the failing test**

```python
def test_format_candidate_incompatibility_digest_is_stable():
    from pyimgano.reporting.run_index_helpers import format_candidate_incompatibility_digest
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: FAIL because the helper is missing

**Step 3: Write minimal implementation**

```python
def format_candidate_incompatibility_digest(entry):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py tests/test_run_index_helpers.py
git commit -m "test: cover candidate digest formatting"
```

### Task 6: Extract candidate digest/brief helper usage from `run_index`

**Files:**
- Modify: `pyimgano/reporting/run_index_helpers.py`
- Modify: `pyimgano/reporting/run_index.py`
- Test: `tests/test_run_index.py`

**Step 1: Write the failing test**

```python
def test_compare_run_summaries_preserves_candidate_digest_fields():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index.py -q`
Expected: FAIL because extracted helper wiring is incomplete

**Step 3: Write minimal implementation**

```python
digest = format_candidate_incompatibility_digest(entry)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index.py tests/test_run_index_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index.py pyimgano/reporting/run_index_helpers.py tests/test_run_index.py
git commit -m "refactor: extract run candidate formatting helpers"
```

### Task 7: Add failing tests for `runs_cli` rendering helper extraction

**Files:**
- Create: `tests/test_runs_cli_rendering.py`
- Test: `pyimgano/runs_cli.py`

**Step 1: Write the failing test**

```python
def test_format_run_brief_line_renders_quality_and_trust():
    from pyimgano.runs_cli_rendering import format_run_brief_line
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py -q`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
def format_run_brief_line(run):
    return ""
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_runs_cli_rendering.py pyimgano/runs_cli_rendering.py
git commit -m "test: add runs cli rendering scaffolding"
```

### Task 8: Extract run/list brief rendering into `runs_cli_rendering`

**Files:**
- Create: `pyimgano/runs_cli_rendering.py`
- Modify: `pyimgano/runs_cli.py`
- Test: `tests/test_runs_cli_rendering.py`
- Test: `tests/test_runs_cli.py`

**Step 1: Write the failing test**

```python
def test_runs_cli_list_uses_rendering_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py tests/test_runs_cli.py -q`
Expected: FAIL because `runs_cli.py` is not delegating yet

**Step 3: Write minimal implementation**

```python
import pyimgano.runs_cli_rendering as runs_cli_rendering
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py tests/test_runs_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/runs_cli.py pyimgano/runs_cli_rendering.py tests/test_runs_cli_rendering.py tests/test_runs_cli.py
git commit -m "refactor: extract runs cli brief rendering"
```

### Task 9: Add failing tests for compare text rendering helpers

**Files:**
- Modify: `tests/test_runs_cli_rendering.py`
- Test: `pyimgano/runs_cli_rendering.py`

**Step 1: Write the failing test**

```python
def test_format_compare_run_brief_line_includes_primary_metric_status():
    from pyimgano.runs_cli_rendering import format_compare_run_brief_line
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py -q`
Expected: FAIL because the helper is missing

**Step 3: Write minimal implementation**

```python
def format_compare_run_brief_line(run, *, primary_metric_name=None, primary_metric_row=None):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/runs_cli_rendering.py tests/test_runs_cli_rendering.py
git commit -m "test: cover compare run brief rendering"
```

### Task 10: Delegate compare text rendering out of `runs_cli`

**Files:**
- Modify: `pyimgano/runs_cli.py`
- Modify: `pyimgano/runs_cli_rendering.py`
- Test: `tests/test_runs_cli.py`

**Step 1: Write the failing test**

```python
def test_runs_cli_compare_text_uses_rendering_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: FAIL because compare text still renders inline

**Step 3: Write minimal implementation**

```python
line = runs_cli_rendering.format_compare_run_brief_line(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/runs_cli.py pyimgano/runs_cli_rendering.py tests/test_runs_cli.py
git commit -m "refactor: delegate compare rendering from runs cli"
```

### Task 11: Add failing tests for compare JSON trust/comparability contract stability

**Files:**
- Modify: `tests/test_runs_cli.py`
- Modify: `tests/test_run_index.py`

**Step 1: Write the failing test**

```python
def test_runs_compare_json_exposes_trust_and_gate_fields():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: FAIL because fields are missing or not normalized consistently

**Step 3: Write minimal implementation**

```python
payload["comparability"] = {...}
payload["trust"] = {...}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index.py pyimgano/runs_cli.py tests/test_runs_cli.py tests/test_run_index.py
git commit -m "test: lock runs compare json contract"
```

### Task 12: Normalize compare JSON assembly around structured payloads

**Files:**
- Modify: `pyimgano/reporting/run_index.py`
- Modify: `pyimgano/runs_cli.py`
- Test: `tests/test_runs_cli.py`
- Test: `tests/test_run_index.py`

**Step 1: Write the failing test**

```python
def test_compare_run_summaries_keeps_same_payload_shape_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: FAIL because helper extraction changes payload assembly

**Step 3: Write minimal implementation**

```python
comparison_payload = {...}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index.py pyimgano/runs_cli.py tests/test_runs_cli.py tests/test_run_index.py
git commit -m "refactor: normalize runs compare payload assembly"
```

### Task 13: Add failing tests for quality/acceptance rendering delegation

**Files:**
- Modify: `tests/test_runs_cli.py`
- Test: `pyimgano/runs_cli.py`

**Step 1: Write the failing test**

```python
def test_runs_quality_text_uses_rendering_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py -q`
Expected: FAIL because quality/acceptance text still builds inline

**Step 3: Write minimal implementation**

```python
def format_quality_summary(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/runs_cli.py pyimgano/runs_cli_rendering.py tests/test_runs_cli.py
git commit -m "test: cover runs quality rendering delegation"
```

### Task 14: Extract quality/acceptance text rendering helpers

**Files:**
- Modify: `pyimgano/runs_cli_rendering.py`
- Modify: `pyimgano/runs_cli.py`
- Test: `tests/test_runs_cli.py`

**Step 1: Write the failing test**

```python
def test_runs_acceptance_text_uses_rendering_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: FAIL because acceptance text still renders inline

**Step 3: Write minimal implementation**

```python
summary = runs_cli_rendering.format_acceptance_summary(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/runs_cli.py pyimgano/runs_cli_rendering.py tests/test_runs_cli.py
git commit -m "refactor: extract runs quality and acceptance rendering"
```

### Task 15: Add failing tests for `run_index` metric display hint helpers

**Files:**
- Modify: `tests/test_run_index_helpers.py`
- Test: `pyimgano/reporting/run_index_helpers.py`

**Step 1: Write the failing test**

```python
def test_format_metric_value_returns_none_for_non_numeric_values():
    from pyimgano.reporting.run_index_helpers import format_metric_value
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: FAIL because helper is missing or not exported

**Step 3: Write minimal implementation**

```python
def format_metric_value(value):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py tests/test_run_index_helpers.py
git commit -m "test: cover run metric display helpers"
```

### Task 16: Move metric formatting and gate-summary helpers behind shared functions

**Files:**
- Modify: `pyimgano/reporting/run_index_helpers.py`
- Modify: `pyimgano/reporting/run_index.py`
- Modify: `pyimgano/runs_cli_rendering.py`
- Test: `tests/test_run_index_helpers.py`
- Test: `tests/test_runs_cli_rendering.py`

**Step 1: Write the failing test**

```python
def test_format_candidate_comparability_gates_is_stable():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py tests/test_runs_cli_rendering.py -q`
Expected: FAIL because helpers are not yet shared

**Step 3: Write minimal implementation**

```python
def format_candidate_comparability_gates(gates):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_run_index_helpers.py tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py pyimgano/reporting/run_index.py pyimgano/runs_cli_rendering.py tests/test_run_index_helpers.py tests/test_runs_cli_rendering.py
git commit -m "refactor: share run metric and gate formatting helpers"
```

### Task 17: Add failing architecture tests for new helper modules

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/reporting/run_index_helpers.py`
- Test: `pyimgano/runs_cli_rendering.py`

**Step 1: Write the failing test**

```python
def test_runs_helper_modules_define_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because helper modules are missing from export expectations

**Step 3: Write minimal implementation**

```python
__all__ = ["format_run_brief_line"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/reporting/run_index_helpers.py pyimgano/runs_cli_rendering.py
git commit -m "test: add runs helper boundary expectations"
```

### Task 18: Add explicit `__all__` and import-boundary discipline to helper modules

**Files:**
- Modify: `pyimgano/reporting/run_index_helpers.py`
- Modify: `pyimgano/runs_cli_rendering.py`
- Test: `tests/test_architecture_boundaries.py`

**Step 1: Write the failing test**

```python
def test_runs_helper_modules_only_import_allowed_internal_modules():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because import boundaries are not locked yet

**Step 3: Write minimal implementation**

```python
__all__ = [...]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/reporting/run_index_helpers.py pyimgano/runs_cli_rendering.py tests/test_architecture_boundaries.py
git commit -m "refactor: harden runs helper module boundaries"
```

### Task 19: Add failing docs contract tests for `pyimgano-runs`

**Files:**
- Create: `tests/test_runs_docs_contract.py`
- Test: `docs/CLI_REFERENCE.md`
- Test: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_reference_documents_runs_compare_quality_acceptance():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_docs_contract.py -q`
Expected: FAIL because the new contract assertions are not encoded yet

**Step 3: Write minimal implementation**

```python
assert "pyimgano-runs" in text
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_runs_docs_contract.py
git commit -m "test: add runs docs contract checks"
```

### Task 20: Update docs for compare/quality/trust fields

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`
- Test: `tests/test_runs_docs_contract.py`

**Step 1: Write the failing test**

```python
def test_readme_mentions_runs_quality_and_compare_contracts():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_docs_contract.py -q`
Expected: FAIL because docs omit expected wording

**Step 3: Write minimal implementation**

```markdown
- `pyimgano-runs quality`
- `pyimgano-runs compare`
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/CLI_REFERENCE.md README.md tests/test_runs_docs_contract.py
git commit -m "docs: clarify runs compare and quality contracts"
```

### Task 21: Add focused regression tests for list/compare text summaries

**Files:**
- Modify: `tests/test_runs_cli.py`
- Modify: `tests/test_runs_cli_rendering.py`

**Step 1: Write the failing test**

```python
def test_runs_compare_text_preserves_reason_and_contract_markers():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: FAIL because the rendered text does not yet match the locked summary contract

**Step 3: Write minimal implementation**

```python
parts.append("operator_contract=...")
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_runs_cli_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_runs_cli.py tests/test_runs_cli_rendering.py pyimgano/runs_cli_rendering.py
git commit -m "test: lock runs text summaries"
```

### Task 22: Add focused regression tests for compare/list JSON summaries

**Files:**
- Modify: `tests/test_runs_cli.py`
- Modify: `tests/test_run_index.py`

**Step 1: Write the failing test**

```python
def test_runs_list_json_exposes_quality_and_trust_fields():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: FAIL because the JSON contract is not fully locked

**Step 3: Write minimal implementation**

```python
payload["artifact_quality"] = ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_runs_cli.py tests/test_run_index.py pyimgano/reporting/run_index.py pyimgano/runs_cli.py
git commit -m "test: lock runs json summaries"
```

### Task 23: Run the targeted runs/reporting verification suite

**Files:**
- Verify only:
  - `tests/test_run_index.py`
  - `tests/test_run_index_helpers.py`
  - `tests/test_runs_cli.py`
  - `tests/test_runs_cli_rendering.py`
  - `tests/test_run_quality.py`
  - `tests/test_run_acceptance_states_v1.py`
  - `tests/test_runs_docs_contract.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_run_index.py \
  tests/test_run_index_helpers.py \
  tests/test_runs_cli.py \
  tests/test_runs_cli_rendering.py \
  tests/test_run_quality.py \
  tests/test_run_acceptance_states_v1.py \
  tests/test_runs_docs_contract.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify runs reporting hardening suite"
```

### Task 24: Run static verification on touched runs/reporting files

**Files:**
- Verify only:
  - `pyimgano/runs_cli.py`
  - `pyimgano/runs_cli_rendering.py`
  - `pyimgano/reporting/run_index.py`
  - `pyimgano/reporting/run_index_helpers.py`
  - `tests/test_run_index_helpers.py`
  - `tests/test_runs_cli_rendering.py`
  - `tests/test_runs_docs_contract.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/runs_cli.py \
  pyimgano/runs_cli_rendering.py \
  pyimgano/reporting/run_index.py \
  pyimgano/reporting/run_index_helpers.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/runs_cli.py \
  pyimgano/runs_cli_rendering.py \
  pyimgano/reporting/run_index.py \
  pyimgano/reporting/run_index_helpers.py \
  tests/test_run_index_helpers.py \
  tests/test_runs_cli_rendering.py \
  tests/test_runs_docs_contract.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize runs reporting hardening checks"
```
