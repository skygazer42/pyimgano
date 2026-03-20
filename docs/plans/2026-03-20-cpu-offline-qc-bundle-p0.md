# CPU Offline QC Bundle P0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden `pyimgano` into a CPU-first offline QC bundle contract by formalizing deploy bundle metadata and exposing stable acceptance semantics.

**Architecture:** Reuse the existing deploy-bundle, calibration-card, and run-acceptance modules rather than introducing a parallel packaging path. Add contract fields to `bundle_manifest.json`, derive threshold/evaluation summaries from existing artifacts, and extend acceptance payloads with stable bundle states and machine-readable reason codes while preserving current compatibility fields.

**Tech Stack:** Python 3.10+, pytest, pathlib, json, existing `pyimgano.reporting`, existing train/export services.

---

## Stream A: Deploy Bundle Contract v1

### Task 1: Add failing tests for contract fields in generated bundle manifests

**Files:**
- Modify: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_deploy_bundle_contract_v1.py`

**Step 1: Write the failing tests**

Add focused tests that assert `build_deploy_bundle_manifest(...)` now emits:

- `bundle_type == "cpu-offline-qc"`
- `status == "draft"`
- `compatibility` object
- `input_contract` object
- `output_contract` object
- `threshold_summary` object with provenance and split-fingerprint booleans
- `evaluation_summary` object with threshold scope / split-fingerprint summary

Also add one validator test that tampers with `threshold_summary` and expects a contract error.

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_contract_v1.py -q`

Expected: FAIL because the current manifest builder does not emit those fields and validator does not check them.

**Step 3: Write minimal implementation**

Modify `pyimgano/reporting/deploy_bundle.py` to:

- load `infer_config.json` and `calibration_card.json` when present
- derive bundle-level contract summaries
- stamp the new fields into the generated manifest
- validate those fields when present

Keep existing manifest fields unchanged for compatibility.

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_contract_v1.py -q`

Expected: PASS

### Task 2: Surface contract metadata during deploy-bundle export

**Files:**
- Modify: `pyimgano/services/train_service.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Step 1: Write the failing test**

Add one assertion to the deploy-bundle export flow test that the generated `bundle_manifest.json` contains the new contract fields and that the copied `infer_config.json` still carries bundle artifact metadata.

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_workbench_export_infer_config.py -k "bundle_manifest" -q`

Expected: FAIL because the generated bundle manifest does not yet expose the new fields.

**Step 3: Write minimal implementation**

Only adjust export code if required by the new manifest builder. Avoid expanding runtime behavior beyond writing the richer manifest.

**Step 4: Run test to verify it passes**

Run: `pytest --no-cov tests/test_workbench_export_infer_config.py -k "bundle_manifest" -q`

Expected: PASS

## Stream B: Acceptance State and Reason Codes

### Task 3: Add failing acceptance-state tests

**Files:**
- Create: `tests/test_run_acceptance_states_v1.py`
- Modify: `tests/test_runs_cli.py`

**Step 1: Write the failing tests**

Add focused acceptance tests for:

- audited run -> `acceptance_state == "audited"` and `reason_codes == []`
- deployable run -> `acceptance_state == "deployable"`
- incomplete run -> `acceptance_state == "blocked"` with `BUNDLE_REQUIRED_QUALITY_NOT_MET`
- missing infer-config run -> includes `BUNDLE_MISSING_INFER_CONFIG`

Add one CLI JSON assertion that the new fields are visible under `acceptance`.

**Step 2: Run tests to verify they fail**

Run: `pytest --no-cov tests/test_run_acceptance_states_v1.py tests/test_runs_cli.py -k "acceptance_state or reason_codes" -q`

Expected: FAIL because `evaluate_run_acceptance(...)` does not emit those fields yet.

**Step 3: Write minimal implementation**

Modify `pyimgano/reporting/run_acceptance.py` to:

- map existing run quality states onto bundle acceptance states
- derive stable machine-readable `reason_codes`
- keep `status`, `ready`, and `blocking_reasons` for compatibility

**Step 4: Run tests to verify they pass**

Run: `pytest --no-cov tests/test_run_acceptance_states_v1.py tests/test_runs_cli.py -k "acceptance_state or reason_codes" -q`

Expected: PASS

### Task 4: Surface the same acceptance semantics in doctor readiness payloads

**Files:**
- Modify: `pyimgano/services/doctor_service.py`
- Test: `tests/test_doctor_cli.py`

**Step 1: Write the failing test**

Add a test that `pyimgano-doctor --run-dir ... --json` exposes acceptance payload fields:

- `acceptance.acceptance_state`
- `acceptance.reason_codes`

Do not change deploy-bundle diagnostics beyond what the shared acceptance payload already exposes.

**Step 2: Run test to verify it fails**

Run: `pytest --no-cov tests/test_doctor_cli.py -k "acceptance_state or reason_codes" -q`

Expected: FAIL because the nested acceptance payload currently lacks those fields.

**Step 3: Write minimal implementation**

If needed, only adapt formatting or passthrough logic in `doctor_service.py`. Prefer not to duplicate acceptance logic there.

**Step 4: Run test to verify it passes**

Run: `pytest --no-cov tests/test_doctor_cli.py -k "acceptance_state or reason_codes" -q`

Expected: PASS

## Stream C: Final Verification

### Task 5: Run focused verification bundle

**Verify:**
- `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_contract_v1.py tests/test_workbench_export_infer_config.py tests/test_run_acceptance_states_v1.py tests/test_runs_cli.py tests/test_doctor_cli.py -q`
- `python -m py_compile pyimgano/reporting/deploy_bundle.py pyimgano/reporting/run_acceptance.py pyimgano/services/train_service.py pyimgano/services/doctor_service.py`
- `ruff check pyimgano/reporting/deploy_bundle.py pyimgano/reporting/run_acceptance.py pyimgano/services/train_service.py pyimgano/services/doctor_service.py tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_contract_v1.py tests/test_run_acceptance_states_v1.py`
- `black --check pyimgano/reporting/deploy_bundle.py pyimgano/reporting/run_acceptance.py pyimgano/services/train_service.py pyimgano/services/doctor_service.py tests/test_deploy_bundle_manifest.py tests/test_deploy_bundle_contract_v1.py tests/test_run_acceptance_states_v1.py`
- `git diff --check`
