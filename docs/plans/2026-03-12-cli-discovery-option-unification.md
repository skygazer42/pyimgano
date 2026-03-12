# CLI Discovery Option Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify repeated CLI discovery option validation so the benchmark and inference entrypoints share one adapter-layer source of truth for mutually exclusive flags and `--list-models` filter validation.

**Architecture:** Add a small helper module that owns two responsibilities only: validating mutually exclusive discovery flags and normalizing/validating `list_models` filter options. Refactor `pyimgano.cli` and `pyimgano.infer_cli` to call this helper before dispatching discovery branches, while keeping their distinct preset and benchmark/inference behavior local.

**Tech Stack:** Python 3.10, pytest, argparse namespaces, existing `pyimgano.discovery` validation helpers.

---

### Task 1: Add Shared Discovery Option Helper

**Files:**
- Create: `pyimgano/cli_discovery_options.py`
- Test: `tests/test_cli_discovery_options.py`

**Step 1: Write the failing test**

Add `tests/test_cli_discovery_options.py` with tests that prove:
- `validate_mutually_exclusive_flags(...)` raises a `ValueError` with the formatted discovery flag list when more than one flag is active
- `resolve_model_list_discovery_options(...)` rejects `--type` and `--year` when `--list-models` is not active
- `resolve_model_list_discovery_options(...)` validates configured filters and returns normalized values for downstream service calls
- `resolve_model_list_discovery_options(...)` can allow `family` outside `--list-models` when explicitly requested so `infer_cli` can preserve its preset-listing behavior

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_discovery_options.py -v`
Expected: FAIL because `pyimgano.cli_discovery_options` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/cli_discovery_options.py`
- Add a small immutable data carrier for normalized `list_models` filters
- Implement:
  - `validate_mutually_exclusive_flags(flags) -> None`
  - `resolve_model_list_discovery_options(...) -> ModelListDiscoveryOptions`
- Keep the helper adapter-only and avoid CLI-parser-specific dependencies

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_discovery_options.py -v`
Expected: PASS

### Task 2: Migrate Benchmark And Infer CLIs

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`

**Step 1: Write the failing test**

Extend CLI tests so they prove:
- `pyimgano.cli.main(... --list-models ...)` delegates `list_models` filter normalization through the shared discovery option helper before calling `discovery_service`
- `pyimgano.infer_cli.main(... --list-models ...)` delegates the same way and still preserves its existing discovery behavior

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v`
Expected: FAIL because the CLI modules still inline discovery validation and filter handling.

**Step 3: Write minimal implementation**

- Import the shared helper in both CLI modules
- Replace inline discovery flag exclusivity checks with `validate_mutually_exclusive_flags(...)`
- Replace inline `list_models` filter validation with `resolve_model_list_discovery_options(...)`
- Preserve existing CLI behavior, including `infer_cli` family handling for model preset discovery

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_cli_feature_discovery.py tests/test_cli_smoke.py tests/test_infer_cli_smoke.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Test: `tests/test_cli_discovery_options.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_infer_cli_smoke.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_cli_feature_discovery.py tests/test_cli_smoke.py tests/test_infer_cli_smoke.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-cli-discovery-option-unification.md pyimgano/cli_discovery_options.py pyimgano/cli.py pyimgano/infer_cli.py tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py`
Expected: no output
