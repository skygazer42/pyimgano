# Pyim Payload Typing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the remaining stringly-typed coupling between `pyimgano.services.pyim_service` and `pyimgano.pyim_cli_rendering` by introducing an explicit `PyimListPayload` type.

**Architecture:** Keep `PyimListRequest` as the input contract to `pyimgano.services.pyim_service`, but replace the `dict[str, Any]` return value with a frozen `PyimListPayload` dataclass that names each section explicitly. Update `pyimgano.pyim_cli_rendering` to render from the typed payload instead of repeatedly indexing raw dictionaries, while preserving the existing JSON output shapes and text sections.

**Tech Stack:** Python 3.10, pytest, dataclasses, existing `pyimgano.services.pyim_service`, existing `pyimgano.pyim_cli_rendering`.

---

### Task 1: Add A Typed Pyim List Payload

**Files:**
- Modify: `pyimgano/services/pyim_service.py`
- Test: `tests/test_pyim_service.py`

**Step 1: Write the failing test**

Extend service tests so they prove:
- `collect_pyim_listing_payload(...)` returns a `PyimListPayload`
- sections are accessed via attributes (`models`, `families`, `model_preset_infos`, etc.)
- the payload can build the existing `--list` JSON object for the `all` view without exposing `model_preset_infos`

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: FAIL because the service still returns a raw dict and no payload type exists yet.

**Step 3: Write minimal implementation**

- Add a frozen `PyimListPayload` dataclass
- Have `collect_pyim_listing_payload(...)` return that type
- Add the minimal helper needed for the `all` JSON payload shape

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_service.py -v`
Expected: PASS

### Task 2: Refactor Pyim Rendering To Consume The Typed Payload

**Files:**
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli_model_presets.py`

**Step 1: Write the failing test**

Extend rendering-oriented tests so they prove:
- `emit_pyim_list_payload(...)` accepts `PyimListPayload`
- `model-presets` JSON still emits `model_preset_infos`
- CLI delegation tests observe a typed payload object reaching the rendering helper

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py -v`
Expected: FAIL because rendering still expects mapping-style payload access.

**Step 3: Write minimal implementation**

- Update `pyimgano.pyim_cli_rendering` to use payload attributes
- Reuse the typed payload helper for the `all` JSON payload
- Preserve text output formatting and exit codes

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py -v`
Expected: PASS

### Task 3: Focused Regression And Hygiene

**Files:**
- Modify: `pyimgano/services/pyim_service.py`
- Modify: `pyimgano/pyim_cli_rendering.py`
- Test: `tests/test_pyim_service.py`
- Test: `tests/test_pyim_cli_rendering.py`
- Test: `tests/test_pyim_cli.py`
- Test: `tests/test_pyim_cli_model_presets.py`
- Test: `tests/test_pyim_cli_recipes_and_datasets.py`
- Test: `tests/test_pyim_cli_options.py`

**Step 1: Run focused regression suite**

Run: `pytest --no-cov tests/test_pyim_service.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli.py tests/test_pyim_cli_model_presets.py tests/test_pyim_cli_recipes_and_datasets.py tests/test_pyim_cli_options.py -v`
Expected: PASS

**Step 2: Check patch hygiene**

Run: `git diff --check -- docs/plans/2026-03-12-pyim-payload-typing.md pyimgano/services/pyim_service.py pyimgano/pyim_cli_rendering.py tests/test_pyim_service.py tests/test_pyim_cli_rendering.py tests/test_pyim_cli_model_presets.py`
Expected: no output
