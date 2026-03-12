# CLI Listing Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify repeated CLI discovery/listing branches behind a shared adapter so list-style commands follow one consistent shape across entrypoints.

**Architecture:** Add a small adapter-layer module that owns the common "print one item per line or emit JSON" behavior for list-style CLI responses. Refactor the large CLI entrypoints first, then migrate smaller CLIs that have the same branching pattern so JSON/text listing behavior is consistent without duplicating loops and conditionals.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano.cli_output` helper, argparse-based CLI modules.

---

### Task 1: Add Shared CLI Listing Adapter

**Files:**
- Create: `pyimgano/cli_listing.py`
- Test: `tests/test_cli_listing.py`

**Step 1: Write the failing test**

Add `tests/test_cli_listing.py` with tests that prove:
- `emit_listing(...)` prints one item per line in text mode and returns status code `0`
- `emit_listing(...)` routes JSON mode through `pyimgano.cli_output.emit_json(...)`
- `emit_listing(...)` can emit a different JSON payload than the text listing names when `json_payload=` is provided

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_listing.py -v`
Expected: FAIL because `pyimgano.cli_listing` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/cli_listing.py`
- Implement `emit_listing(items, *, json_output, json_payload=None, sort_keys=True, status_code=0) -> int`
- Keep the helper adapter-only: it should not know about discovery services or argparse objects

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_listing.py -v`
Expected: PASS

### Task 2: Migrate Discovery/Listing Call Sites

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/train_cli.py`
- Modify: `pyimgano/robust_cli.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_robust_cli_smoke.py`

**Step 1: Write the failing test**

Extend tests so they prove:
- `pyimgano.cli.main(... --list-models ...)` delegates list emission to the shared listing helper
- `pyimgano.cli.main(... --list-feature-extractors ...)` delegates list emission to the shared listing helper
- `pyimgano.infer_cli.main(... --list-model-presets --json ...)` delegates through the shared listing helper while preserving the richer JSON payload
- `pyimgano.robust_cli.main(... --list-models --json ...)` delegates through the shared listing helper

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_cli_listing.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_robust_cli_smoke.py -v`
Expected: FAIL because the CLI modules still inline their list-branch output logic.

**Step 3: Write minimal implementation**

- Import the new helper in each CLI module
- Replace repeated `if json: emit_json(...); for name in names: print(name)` branches with `emit_listing(...)`
- Preserve existing JSON payload shapes for every command

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_listing.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_robust_cli_smoke.py tests/test_cli_smoke.py tests/test_train_cli_dry_run.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Test: `tests/test_cli_listing.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_robust_cli_smoke.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_train_cli_dry_run.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_cli_listing.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_robust_cli_smoke.py tests/test_cli_smoke.py tests/test_train_cli_dry_run.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-cli-listing-unification.md pyimgano/cli_listing.py pyimgano/cli.py pyimgano/infer_cli.py pyimgano/train_cli.py pyimgano/robust_cli.py tests/test_cli_listing.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_robust_cli_smoke.py`
Expected: no output
