# Preset Discovery Filter Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move model preset family/tag filter assembly out of CLI entrypoints so preset discovery filtering is owned by neutral preset/discovery layers.

**Architecture:** Add a neutral helper for resolving model preset discovery tags from raw `tags` and optional `family`, then update `discovery_service` to accept `family` directly for preset listing APIs. Refactor `infer_cli` and `pyim_cli` to pass raw filter inputs through the service boundary instead of building preset tag lists themselves.

**Tech Stack:** Python 3.10, pytest, existing `pyimgano.presets.catalog`, existing `pyimgano.discovery.resolve_family_tags`.

---

### Task 1: Add Neutral Preset Filter Helper

**Files:**
- Modify: `pyimgano/presets/catalog.py`
- Test: `tests/test_preset_catalog.py`

**Step 1: Write the failing test**

Extend `tests/test_preset_catalog.py` with tests that prove:
- a new helper normalizes repeatable/comma-separated preset tags
- the helper expands `family` into preset tags via `resolve_family_tags(...)`
- the helper returns a plain list suitable for downstream service calls

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_preset_catalog.py -v`
Expected: FAIL because the new helper does not exist yet.

**Step 3: Write minimal implementation**

- Add `resolve_model_preset_filter_tags(tags=None, family=None) -> list[str]` to `pyimgano/presets/catalog.py`
- Reuse existing tag normalization logic instead of duplicating comma-splitting

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_preset_catalog.py -v`
Expected: PASS

### Task 2: Move Preset Filter Assembly Behind The Service Boundary

**Files:**
- Modify: `pyimgano/services/discovery_service.py`
- Modify: `pyimgano/infer_cli.py`
- Modify: `pyimgano/pyim_cli.py`
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Write the failing test**

Add or extend tests so they prove:
- `discovery_service.list_model_preset_names(...)` accepts `family=` and delegates with resolved preset tags
- `infer_cli` delegates raw `tags`/`family` to `discovery_service` for preset discovery instead of building preset tags inline
- `pyim_cli` does the same

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_preset_catalog.py tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_pyim_cli_model_presets.py -v`
Expected: FAIL because the service APIs do not accept `family` yet and the CLIs still inline preset tag assembly.

**Step 3: Write minimal implementation**

- Update `discovery_service.list_model_preset_names(...)` and `list_model_preset_infos_payload(...)` to accept `family=None`
- Resolve preset tags in the service layer via the neutral preset helper
- Remove inline preset tag assembly from `infer_cli` and `pyim_cli`
- Preserve current CLI output and filtering behavior

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_preset_catalog.py tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Test: `tests/test_preset_catalog.py`
- Test: `tests/test_discovery_service.py`
- Test: `tests/test_infer_cli_discovery_and_model_presets_v16.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_preset_catalog.py tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_pyim_cli_model_presets.py tests/test_cli_discovery_options.py tests/test_cli_discovery.py tests/test_cli_smoke.py tests/test_infer_cli_smoke.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-preset-discovery-filter-unification.md pyimgano/presets/catalog.py pyimgano/services/discovery_service.py pyimgano/infer_cli.py pyimgano/pyim_cli.py tests/test_preset_catalog.py tests/test_discovery_service.py tests/test_infer_cli_discovery_and_model_presets_v16.py tests/test_pyim_cli_model_presets.py`
Expected: no output
