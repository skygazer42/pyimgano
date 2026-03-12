# CLI Output Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify common CLI adapter behavior by introducing shared helpers for JSON emission and error reporting, then migrate the main entrypoints to use them consistently.

**Architecture:** Add a small adapter-layer helper module that owns pretty JSON output, JSONable conversion, and stderr error formatting. Update the thin CLI entrypoints to route top-level JSON responses and terminal errors through this helper so formatting and failure behavior live in one place instead of being reimplemented per command.

**Tech Stack:** Python 3.10, pytest, `json`, `sys`, existing `to_jsonable` helper.

---

### Task 1: Add Shared CLI Output Helpers

**Files:**
- Create: `pyimgano/cli_output.py`
- Test: `tests/test_cli_output.py`

**Step 1: Write the failing test**

Add `tests/test_cli_output.py` with tests that:
- `emit_json(...)` prints sorted, indented JSON and returns the requested status code
- `emit_jsonable(...)` converts non-JSON-native values before printing
- `print_cli_error(...)` writes the standard `error:` prefix and optional context lines to stderr

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_output.py -v`
Expected: FAIL because `pyimgano.cli_output` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/cli_output.py` with:
  - `emit_json(payload, *, status_code=0, indent=2, sort_keys=True) -> int`
  - `emit_jsonable(payload, *, status_code=0, indent=2, sort_keys=True) -> int`
  - `print_cli_error(exc, *, context_lines=None) -> None`

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_output.py -v`
Expected: PASS

### Task 2: Migrate CLI Entry Points to the Shared Helpers

**Files:**
- Modify: `pyimgano/pyim_cli.py`
- Modify: `pyimgano/doctor_cli.py`
- Modify: `pyimgano/train_cli.py`
- Modify: `pyimgano/robust_cli.py`
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_doctor_cli.py`
- Test: `tests/test_train_cli_dry_run.py`
- Test: `tests/test_robust_cli_smoke.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Write the failing test**

Extend tests so they prove:
- one JSON-heavy CLI path delegates through `emit_json(...)`
- one JSONable CLI path delegates through `emit_jsonable(...)`
- one error path delegates through `print_cli_error(...)`

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_output.py tests/test_doctor_cli.py tests/test_train_cli_dry_run.py tests/test_robust_cli_smoke.py tests/test_pyim_cli_model_presets.py -v`
Expected: FAIL because the CLIs still print JSON and errors directly.

**Step 3: Write minimal implementation**

- Replace local JSON emission helpers with `pyimgano.cli_output.emit_json(...)`
- Replace direct `print(f"error: {exc}", file=sys.stderr)` calls with `print_cli_error(...)`
- Use `emit_jsonable(...)` where payloads currently go through `to_jsonable(...)`
- Keep existing exit codes and CLI semantics unchanged

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_output.py tests/test_doctor_cli.py tests/test_train_cli_dry_run.py tests/test_robust_cli_smoke.py tests/test_pyim_cli_model_presets.py tests/test_cli_smoke.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_infer_cli_smoke.py -v`
Expected: PASS

### Task 3: Run Focused Regression Coverage

**Files:**
- Test: `tests/test_cli_output.py`
- Test: `tests/test_doctor_cli.py`
- Test: `tests/test_train_cli_dry_run.py`
- Test: `tests/test_robust_cli_smoke.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_infer_cli_smoke.py`
- Test: `tests/test_service_import_style.py`
- Test: `tests/test_services_package.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_cli_output.py tests/test_doctor_cli.py tests/test_train_cli_dry_run.py tests/test_robust_cli_smoke.py tests/test_pyim_cli_model_presets.py tests/test_cli_smoke.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_infer_cli_smoke.py tests/test_service_import_style.py tests/test_services_package.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/cli_output.py pyimgano/pyim_cli.py pyimgano/doctor_cli.py pyimgano/train_cli.py pyimgano/robust_cli.py pyimgano/cli.py pyimgano/infer_cli.py tests/test_cli_output.py tests/test_doctor_cli.py tests/test_train_cli_dry_run.py tests/test_robust_cli_smoke.py tests/test_pyim_cli_model_presets.py docs/plans/2026-03-12-cli-output-unification.md`
Expected: no output
