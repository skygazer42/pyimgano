# CLI Discovery Rendering Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify repeated discovery-mode rendering in `cli.py` and `infer_cli.py` by extracting shared adapter-layer renderers for metadata/signature payloads and model preset payloads.

**Architecture:** Add a small `cli_discovery_rendering` helper module that owns JSON-or-text emission for common discovery payload shapes. Keep discovery data assembly in `pyimgano.services.discovery_service`, but stop duplicating text rendering branches across the CLI entrypoints.

**Tech Stack:** Python 3.10, pytest, existing `cli_output` helper, existing discovery service payload shapes.

---

### Task 1: Add Shared Discovery Renderers

**Files:**
- Create: `pyimgano/cli_discovery_rendering.py`
- Test: `tests/test_cli_discovery_rendering.py`

**Step 1: Write the failing test**

Add `tests/test_cli_discovery_rendering.py` with tests that:
- `emit_signature_payload(...)` emits JSON through `cli_output.emit_jsonable(...)`
- `emit_signature_payload(...)` emits the expected text structure for metadata/signature payloads
- `emit_model_preset_payload(...)` emits the expected text structure for preset payloads

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_discovery_rendering.py -v`
Expected: FAIL because `pyimgano.cli_discovery_rendering` does not exist yet.

**Step 3: Write minimal implementation**

- Create `pyimgano/cli_discovery_rendering.py` with:
  - `emit_signature_payload(payload, *, json_output) -> int`
  - `emit_model_preset_payload(payload, *, json_output) -> int`
- Route JSON responses through `pyimgano.cli_output`
- Keep text formatting identical to current CLI output

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_discovery_rendering.py -v`
Expected: PASS

### Task 2: Migrate `cli.py` and `infer_cli.py` Discovery Branches

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `tests/test_cli_discovery.py`
- Modify: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Modify: `tests/test_cli_feature_discovery.py`

**Step 1: Write the failing test**

Extend CLI tests so they prove:
- `cli.py --model-info` uses the shared renderer
- `cli.py --feature-info` uses the shared renderer
- `infer_cli.py --model-info` uses the shared renderer
- `infer_cli.py --model-preset-info` uses the shared renderer

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_cli_discovery_rendering.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py -v`
Expected: FAIL because the CLIs still render those payloads inline.

**Step 3: Write minimal implementation**

- Import the shared renderer module in `pyimgano/cli.py` and `pyimgano/infer_cli.py`
- Replace the inline text/JSON branches for:
  - `model_info`
  - `feature_info` in `cli.py`
  - `model_preset_info`
- Leave suite/sweep rendering alone for now

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_cli_discovery_rendering.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_cli_model_info_materializes_signature_v8.py -v`
Expected: PASS

### Task 3: Run Focused Regression Coverage

**Files:**
- Test: `tests/test_cli_discovery_rendering.py`
- Test: `tests/test_cli_discovery.py`
- Test: `tests/test_cli_feature_discovery.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_cli_model_info_materializes_signature_v8.py`
- Test: `tests/test_cli_smoke.py`
- Test: `tests/test_infer_cli_smoke.py`
- Test: `tests/test_cli_output.py`

**Step 1: Run regression suite**

Run: `pytest --no-cov tests/test_cli_discovery_rendering.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_cli_model_info_materializes_signature_v8.py tests/test_cli_smoke.py tests/test_infer_cli_smoke.py tests/test_cli_output.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- pyimgano/cli_discovery_rendering.py pyimgano/cli.py pyimgano/infer_cli.py tests/test_cli_discovery_rendering.py tests/test_cli_discovery.py tests/test_cli_feature_discovery.py tests/test_infer_cli_discovery_and_model_presets_v16.py docs/plans/2026-03-12-cli-discovery-rendering-unification.md`
Expected: no output
