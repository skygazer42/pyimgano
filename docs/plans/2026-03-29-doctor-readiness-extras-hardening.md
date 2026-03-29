# Doctor Readiness / Extras Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden `pyimgano-doctor` around extras checks and readiness diagnostics without changing existing public CLI behavior, JSON fields, or exit-code semantics.

**Architecture:** Keep `pyimgano.services.doctor_service` as the source of JSON-ready extras/readiness payloads, but extract smaller helpers for extras, accelerator, and readiness assembly. Keep `pyimgano.doctor_cli` as the public entrypoint, but move text summary rendering into helper functions so the CLI mostly parses args, delegates, and maps stable payload states onto exit codes.

**Tech Stack:** Python 3.9+, argparse, JSON, existing `pyimgano.services.doctor_service`, `pyimgano.doctor_cli`, `pyimgano.reporting.run_acceptance`, pytest.

---

### Task 1: Add failing tests for doctor rendering helper scaffolding

**Files:**
- Create: `tests/test_doctor_rendering.py`
- Test: `pyimgano/doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_format_doctor_quality_summary_line():
    from pyimgano.doctor_rendering import format_doctor_quality_summary_line
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.doctor_rendering'`

**Step 3: Write minimal implementation**

```python
def format_doctor_quality_summary_line(...):
    return ""
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_doctor_rendering.py pyimgano/doctor_rendering.py
git commit -m "test: add doctor rendering scaffolding"
```

### Task 2: Extract doctor text summary helpers for quality/readiness/extras

**Files:**
- Create: `pyimgano/doctor_rendering.py`
- Modify: `pyimgano/doctor_cli.py`
- Test: `tests/test_doctor_rendering.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_doctor_cli_text_uses_rendering_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py tests/test_doctor_cli.py -q`
Expected: FAIL because `doctor_cli.py` still renders summaries inline

**Step 3: Write minimal implementation**

```python
import pyimgano.doctor_rendering as doctor_rendering
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py tests/test_doctor_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/doctor_rendering.py pyimgano/doctor_cli.py tests/test_doctor_rendering.py tests/test_doctor_cli.py
git commit -m "refactor: extract doctor rendering helpers"
```

### Task 3: Add failing tests for service extras helper extraction

**Files:**
- Create: `tests/test_doctor_service_helpers.py`
- Test: `pyimgano/services/doctor_service.py`

**Step 1: Write the failing test**

```python
def test_build_require_extras_check_reports_missing_and_install_hint():
    from pyimgano.services.doctor_service_helpers import build_require_extras_check
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
def build_require_extras_check(required_extras):
    return {"required": [], "missing": [], "ok": True, "install_hint": None}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_doctor_service_helpers.py pyimgano/services/doctor_service_helpers.py
git commit -m "test: add doctor service helper scaffolding"
```

### Task 4: Move extras requirement helpers behind `doctor_service_helpers`

**Files:**
- Create: `pyimgano/services/doctor_service_helpers.py`
- Modify: `pyimgano/services/doctor_service.py`
- Test: `tests/test_doctor_service_helpers.py`
- Test: `tests/test_doctor_service.py`

**Step 1: Write the failing test**

```python
def test_split_csv_args_preserves_repeatable_comma_syntax():
    from pyimgano.services.doctor_service_helpers import split_csv_args
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_service.py -q`
Expected: FAIL because the helper is missing or not wired

**Step 3: Write minimal implementation**

```python
def split_csv_args(values):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/doctor_service_helpers.py pyimgano/services/doctor_service.py tests/test_doctor_service_helpers.py tests/test_doctor_service.py
git commit -m "refactor: extract doctor extras helpers"
```

### Task 5: Add failing tests for accelerator payload helper extraction

**Files:**
- Modify: `tests/test_doctor_service_helpers.py`
- Modify: `tests/test_doctor_accelerators_v1.py`

**Step 1: Write the failing test**

```python
def test_build_accelerator_checks_returns_json_ready_shape():
    from pyimgano.services.doctor_service_helpers import build_accelerator_checks
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_accelerators_v1.py -q`
Expected: FAIL because helper is not exported from the helper module

**Step 3: Write minimal implementation**

```python
def build_accelerator_checks():
    return {}
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_accelerators_v1.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/doctor_service_helpers.py tests/test_doctor_service_helpers.py tests/test_doctor_accelerators_v1.py
git commit -m "test: cover doctor accelerator helpers"
```

### Task 6: Move accelerator helper usage out of the doctor service body

**Files:**
- Modify: `pyimgano/services/doctor_service_helpers.py`
- Modify: `pyimgano/services/doctor_service.py`
- Test: `tests/test_doctor_accelerators_v1.py`
- Test: `tests/test_doctor_service.py`

**Step 1: Write the failing test**

```python
def test_collect_doctor_payload_delegates_accelerator_payload_building(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_accelerators_v1.py tests/test_doctor_service.py -q`
Expected: FAIL because `collect_doctor_payload(...)` still uses inline accelerator assembly

**Step 3: Write minimal implementation**

```python
accelerators_payload = doctor_service_helpers.build_accelerator_checks()
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_accelerators_v1.py tests/test_doctor_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/doctor_service.py pyimgano/services/doctor_service_helpers.py tests/test_doctor_accelerators_v1.py tests/test_doctor_service.py
git commit -m "refactor: extract doctor accelerator helpers"
```

### Task 7: Add failing tests for readiness summary helper extraction

**Files:**
- Modify: `tests/test_doctor_service_helpers.py`
- Modify: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_build_readiness_summary_handles_error_status():
    from pyimgano.services.doctor_service_helpers import build_readiness_summary
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_cli.py -q`
Expected: FAIL because the helper is missing

**Step 3: Write minimal implementation**

```python
def build_readiness_summary(readiness):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_service_helpers.py tests/test_doctor_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/doctor_service_helpers.py tests/test_doctor_service_helpers.py tests/test_doctor_cli.py
git commit -m "test: cover doctor readiness helpers"
```

### Task 8: Extract readiness summary assembly from `doctor_service`

**Files:**
- Modify: `pyimgano/services/doctor_service_helpers.py`
- Modify: `pyimgano/services/doctor_service.py`
- Test: `tests/test_doctor_service.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_collect_doctor_payload_keeps_readiness_shape_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_service.py tests/test_doctor_cli.py -q`
Expected: FAIL because payload assembly changed shape

**Step 3: Write minimal implementation**

```python
readiness = doctor_service_helpers.build_readiness_summary(raw_readiness)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_service.py tests/test_doctor_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/doctor_service.py pyimgano/services/doctor_service_helpers.py tests/test_doctor_service.py tests/test_doctor_cli.py
git commit -m "refactor: extract doctor readiness helpers"
```

### Task 9: Add failing tests for doctor CLI text summary helpers

**Files:**
- Modify: `tests/test_doctor_rendering.py`
- Modify: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_format_doctor_acceptance_summary_line_handles_run_readiness():
    from pyimgano.doctor_rendering import format_doctor_acceptance_summary_line
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py tests/test_doctor_cli.py -q`
Expected: FAIL because the helper is missing

**Step 3: Write minimal implementation**

```python
def format_doctor_acceptance_summary_line(...):
    return ""
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_rendering.py tests/test_doctor_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/doctor_rendering.py tests/test_doctor_rendering.py tests/test_doctor_cli.py
git commit -m "test: cover doctor readiness rendering"
```

### Task 10: Delegate more doctor CLI text rendering to helper functions

**Files:**
- Modify: `pyimgano/doctor_cli.py`
- Modify: `pyimgano/doctor_rendering.py`
- Test: `tests/test_doctor_cli.py`
- Test: `tests/test_doctor_rendering.py`

**Step 1: Write the failing test**

```python
def test_doctor_cli_text_uses_acceptance_summary_helper(monkeypatch):
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py tests/test_doctor_rendering.py -q`
Expected: FAIL because summary rendering still happens inline

**Step 3: Write minimal implementation**

```python
line = doctor_rendering.format_doctor_acceptance_summary_line(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py tests/test_doctor_rendering.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/doctor_cli.py pyimgano/doctor_rendering.py tests/test_doctor_cli.py tests/test_doctor_rendering.py
git commit -m "refactor: delegate doctor text summaries"
```

### Task 11: Add failing tests for doctor JSON/exit-code parity

**Files:**
- Modify: `tests/test_doctor_cli.py`
- Modify: `tests/test_doctor_service.py`

**Step 1: Write the failing test**

```python
def test_doctor_json_missing_extras_and_error_readiness_drive_exit_code():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py tests/test_doctor_service.py -q`
Expected: FAIL because CLI and payload status are not fully locked together

**Step 3: Write minimal implementation**

```python
if req.get("ok") is False: return 1
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py tests/test_doctor_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/doctor_cli.py tests/test_doctor_cli.py tests/test_doctor_service.py
git commit -m "test: lock doctor exit code parity"
```

### Task 12: Normalize doctor CLI exit-code helpers

**Files:**
- Modify: `pyimgano/doctor_cli.py`
- Modify: `pyimgano/doctor_rendering.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
def test_doctor_cli_text_exit_code_matches_json_exit_code():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py -q`
Expected: FAIL because text and JSON branches do not share enough exit-code logic

**Step 3: Write minimal implementation**

```python
def _doctor_exit_code_from_payload(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/doctor_cli.py tests/test_doctor_cli.py
git commit -m "refactor: normalize doctor exit code handling"
```

### Task 13: Add failing architecture tests for doctor helper modules

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/doctor_rendering.py`
- Test: `pyimgano/services/doctor_service_helpers.py`

**Step 1: Write the failing test**

```python
def test_doctor_helper_modules_define_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because new helper modules are not covered

**Step 3: Write minimal implementation**

```python
__all__ = ["format_doctor_quality_summary_line"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/doctor_rendering.py pyimgano/services/doctor_service_helpers.py
git commit -m "test: add doctor helper boundary expectations"
```

### Task 14: Add failing docs contract tests for `pyimgano-doctor`

**Files:**
- Create: `tests/test_doctor_docs_contract.py`
- Test: `docs/CLI_REFERENCE.md`
- Test: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_reference_documents_doctor_extras_and_readiness():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_docs_contract.py -q`
Expected: FAIL because docs contract checks do not exist yet

**Step 3: Write minimal implementation**

```python
assert "pyimgano-doctor" in text
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_doctor_docs_contract.py
git commit -m "test: add doctor docs contract checks"
```

### Task 15: Update docs for extras/readiness contract wording

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`
- Test: `tests/test_doctor_docs_contract.py`

**Step 1: Write the failing test**

```python
def test_readme_mentions_doctor_require_extras_and_readiness():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_doctor_docs_contract.py -q`
Expected: FAIL because docs wording is not locked yet

**Step 3: Write minimal implementation**

```markdown
pyimgano-doctor --require-extras ...
pyimgano-doctor --run-dir ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_doctor_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/CLI_REFERENCE.md README.md tests/test_doctor_docs_contract.py
git commit -m "docs: clarify doctor extras and readiness contracts"
```

### Task 16: Run the targeted doctor verification suite

**Files:**
- Verify only:
  - `tests/test_doctor_service.py`
  - `tests/test_doctor_service_helpers.py`
  - `tests/test_doctor_cli.py`
  - `tests/test_doctor_rendering.py`
  - `tests/test_doctor_accelerators_v1.py`
  - `tests/test_doctor_docs_contract.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_doctor_service.py \
  tests/test_doctor_service_helpers.py \
  tests/test_doctor_cli.py \
  tests/test_doctor_rendering.py \
  tests/test_doctor_accelerators_v1.py \
  tests/test_doctor_docs_contract.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify doctor hardening suite"
```

### Task 17: Run static verification on touched doctor files

**Files:**
- Verify only:
  - `pyimgano/doctor_cli.py`
  - `pyimgano/doctor_rendering.py`
  - `pyimgano/services/doctor_service.py`
  - `pyimgano/services/doctor_service_helpers.py`
  - `tests/test_doctor_service_helpers.py`
  - `tests/test_doctor_rendering.py`
  - `tests/test_doctor_docs_contract.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/doctor_cli.py \
  pyimgano/doctor_rendering.py \
  pyimgano/services/doctor_service.py \
  pyimgano/services/doctor_service_helpers.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/doctor_cli.py \
  pyimgano/doctor_rendering.py \
  pyimgano/services/doctor_service.py \
  pyimgano/services/doctor_service_helpers.py \
  tests/test_doctor_service_helpers.py \
  tests/test_doctor_rendering.py \
  tests/test_doctor_docs_contract.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize doctor hardening checks"
```
