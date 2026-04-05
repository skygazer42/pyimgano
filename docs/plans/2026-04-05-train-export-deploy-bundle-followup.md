# Train Export Deploy-Bundle Follow-up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Thin deploy-bundle assembly in `pyimgano-train` by extracting the remaining supporting-file copy and initial bundle-payload patch logic out of `train_service.py` without changing the current bundle contract.

**Architecture:** Keep `pyimgano.services.train_service` as the orchestration shell for train exports, but move deploy-bundle mechanics behind `pyimgano.services.train_export_helpers`. Preserve the current write order for `infer_config.json`, `handoff_report.json`, and `bundle_manifest.json`, and preserve the existing deploy-bundle JSON fields, filenames, and path-rewrite behavior.

**Tech Stack:** Python 3.10, argparse-facing service layer, JSON, pathlib, shutil, existing `pyimgano.services.train_service`, existing `pyimgano.services.train_export_helpers`, pytest.

---

### Task 1: Add failing tests for supporting-file copy helper scaffolding

**Files:**
- Modify: `tests/test_train_export_helpers.py`
- Test: `pyimgano/services/train_export_helpers.py`

**Step 1: Write the failing test**

```python
def test_copy_deploy_bundle_supporting_files_copies_run_metadata_and_optional_audit_artifacts(
    tmp_path: Path,
) -> None:
    from pyimgano.services.train_export_helpers import copy_deploy_bundle_supporting_files
```

Inside the test, create:

- `run/report.json`
- `run/config.json`
- `run/environment.json`
- `run/artifacts/calibration_card.json`
- `run/artifacts/operator_contract.json`

Then call:

```python
copy_deploy_bundle_supporting_files(
    run_dir=run_dir,
    bundle_dir=bundle_dir,
    calibration_card_filename="calibration_card.json",
    operator_contract_filename="operator_contract.json",
)
```

Assert those files exist in `bundle_dir`.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: FAIL with `ImportError` because `copy_deploy_bundle_supporting_files` does not exist.

**Step 3: Write minimal implementation**

Add to `pyimgano/services/train_export_helpers.py`:

```python
def copy_deploy_bundle_supporting_files(
    *,
    run_dir: Path,
    bundle_dir: Path,
    calibration_card_filename: str,
    operator_contract_filename: str,
) -> None:
    ...
```

The helper should copy:

- `report.json`
- `config.json`
- `environment.json`
- optional `artifacts/calibration_card.json`
- optional `artifacts/operator_contract.json`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_export_helpers.py pyimgano/services/train_export_helpers.py
git commit -m "test: cover deploy bundle supporting file copy helper"
```

### Task 2: Wire supporting-file copy through `train_service`

**Files:**
- Modify: `pyimgano/services/train_service.py`
- Modify: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_train_export_helpers.py`

**Step 1: Write the failing regression test**

In `tests/test_deploy_bundle_manifest.py`, add:

```python
def test_run_train_request_keeps_supporting_bundle_files_after_helper_extraction(tmp_path):
    ...
```

Reuse the existing `run_train_request(...)` pattern from
`test_run_train_request_writes_deploy_bundle_manifest`, then assert the resulting bundle
contains:

- `report.json`
- `config.json`
- `environment.json`
- `calibration_card.json`
- `operator_contract.json`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL after you update the test because `train_service.py` is not yet delegating to the helper.

**Step 3: Write minimal implementation**

Import and use the helper inside
`pyimgano/services/train_service.py`:

```python
from pyimgano.services.train_export_helpers import (
    copy_deploy_bundle_supporting_files as _copy_deploy_bundle_supporting_files_helper,
)
```

Replace the inline copy block in `_export_deploy_bundle(...)` with:

```python
_copy_deploy_bundle_supporting_files_helper(
    run_dir=run_dir,
    bundle_dir=bundle_dir,
    calibration_card_filename=_CALIBRATION_CARD_FILENAME,
    operator_contract_filename=_OPERATOR_CONTRACT_FILENAME,
)
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py tests/test_deploy_bundle_manifest.py
git commit -m "refactor: extract deploy bundle supporting file copy"
```

### Task 3: Add failing tests for initial bundle payload patch helper scaffolding

**Files:**
- Modify: `tests/test_train_export_helpers.py`
- Test: `pyimgano/services/train_export_helpers.py`

**Step 1: Write the failing test**

Add:

```python
def test_prepare_bundle_infer_config_payload_rewrites_audit_refs_and_deploy_flags(
    tmp_path: Path,
) -> None:
    from pyimgano.services.train_export_helpers import prepare_bundle_infer_config_payload
```

Build an `infer_config_payload` with:

- `artifact_quality.audit_refs.calibration_card = "artifacts/calibration_card.json"`
- `artifact_quality.audit_refs.operator_contract = "artifacts/operator_contract.json"`
- `artifact_quality.deploy_refs = {}`
- `artifact_quality.has_deploy_bundle = False`
- `artifact_quality.has_bundle_manifest = False`

Create `bundle_dir/calibration_card.json` and `bundle_dir/operator_contract.json`, call the
helper, and assert:

- `audit_refs.calibration_card == "calibration_card.json"`
- `audit_refs.operator_contract == "operator_contract.json"`
- `deploy_refs.bundle_manifest == "bundle_manifest.json"`
- `has_deploy_bundle is True`
- `has_bundle_manifest is True`
- `required_bundle_artifacts_present is False`
- `bundle_artifact_roles == {}`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: FAIL with `ImportError` because `prepare_bundle_infer_config_payload` does not exist.

**Step 3: Write minimal implementation**

Add to `pyimgano/services/train_export_helpers.py`:

```python
def prepare_bundle_infer_config_payload(
    infer_config_payload: dict[str, Any],
    *,
    bundle_dir: Path,
    calibration_card_filename: str,
    operator_contract_filename: str,
) -> dict[str, Any]:
    ...
```

This helper should:

- deep-copy the input payload
- patch `artifact_quality.audit_refs` when copied bundle-local files exist
- patch `artifact_quality.deploy_refs["bundle_manifest"]`
- set the bundle-presence flags and empty metadata placeholders

Do not perform checkpoint path rewriting in this helper.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_export_helpers.py pyimgano/services/train_export_helpers.py
git commit -m "test: cover initial bundle infer-config payload patch helper"
```

### Task 4: Replace inline payload patching in `_export_deploy_bundle(...)`

**Files:**
- Modify: `pyimgano/services/train_service.py`
- Modify: `tests/test_workbench_export_infer_config.py`
- Test: `tests/test_train_export_helpers.py`

**Step 1: Write the failing regression test**

In `tests/test_workbench_export_infer_config.py`, add:

```python
def test_train_cli_export_deploy_bundle_keeps_artifact_quality_patch_contract_after_helper_extraction(
    tmp_path,
):
    ...
```

Reuse the existing deploy-bundle export pattern and assert the resulting bundle
`infer_config.json` still contains:

- `artifact_quality.audit_refs.calibration_card == "calibration_card.json"`
- `artifact_quality.audit_refs.operator_contract == "operator_contract.json"`
- `artifact_quality.deploy_refs.bundle_manifest == "bundle_manifest.json"`
- `artifact_quality.has_deploy_bundle is True`
- `artifact_quality.has_bundle_manifest is True`

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py -q`
Expected: FAIL after adding the regression because `train_service.py` is still patching the payload inline.

**Step 3: Write minimal implementation**

Import and call the helper in `pyimgano/services/train_service.py`:

```python
from pyimgano.services.train_export_helpers import (
    prepare_bundle_infer_config_payload as _prepare_bundle_infer_config_payload_helper,
)
```

Replace the inline `artifact_quality` mutation block with:

```python
bundle_payload = _prepare_bundle_infer_config_payload_helper(
    infer_config_payload,
    bundle_dir=bundle_dir,
    calibration_card_filename=_CALIBRATION_CARD_FILENAME,
    operator_contract_filename=_OPERATOR_CONTRACT_FILENAME,
)
```

Then keep the existing call to `_rewrite_bundle_paths_helper(...)`.

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py tests/test_workbench_export_infer_config.py
git commit -m "refactor: extract initial deploy bundle payload patching"
```

### Task 5: Add a service-level regression for orchestration order

**Files:**
- Modify: `tests/test_train_service.py`
- Modify: `pyimgano/services/train_service.py`

**Step 1: Write the failing test**

Add:

```python
def test_export_deploy_bundle_orchestrates_helpers_before_manifest_write(tmp_path, monkeypatch):
    ...
```

Monkeypatch the helper seams and `save_run_report` so the test records call order for:

- supporting-file copy helper
- initial payload patch helper
- path rewrite helper
- handoff report write
- manifest build
- manifest metadata patch helper
- final manifest write

Assert the helper order matches the current contract.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_service.py -q`
Expected: FAIL because the orchestration order is not yet locked by a dedicated service-level test.

**Step 3: Write minimal implementation**

Keep production behavior unchanged. Adjust `_export_deploy_bundle(...)` only as needed so the
service reads clearly as:

1. create bundle
2. copy supporting files
3. prepare bundle payload
4. rewrite paths
5. write infer-config
6. write handoff report
7. compute manifest
8. patch infer-config metadata from manifest
9. rewrite infer-config
10. recompute and persist manifest

Prefer naming intermediate variables clearly over adding new behavior.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_service.py pyimgano/services/train_service.py
git commit -m "test: lock deploy bundle export orchestration order"
```

### Task 6: Re-run targeted export regressions and fit-only checkpoint flow

**Files:**
- Test: `tests/test_train_export_helpers.py`
- Test: `tests/test_train_service.py`
- Test: `tests/test_deploy_bundle_manifest.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Step 1: Run the targeted regression suite**

Run:

```bash
python3 -m pytest --no-cov \
  tests/test_train_export_helpers.py \
  tests/test_train_service.py \
  tests/test_deploy_bundle_manifest.py \
  tests/test_workbench_export_infer_config.py -q
```

Expected: PASS

**Step 2: Re-run the fit-only checkpoint regression specifically**

Run:

```bash
python3 -m pytest --no-cov \
  tests/test_workbench_export_infer_config.py::test_train_cli_export_deploy_bundle_runs_fit_only_detector_via_bundle_cli -q
```

Expected: PASS

**Step 3: Run a real CLI smoke for the audited train -> bundle path**

Run:

```bash
python3 -m pyimgano.train_cli \
  --config examples/configs/industrial_adapt_audited.json \
  --dataset custom \
  --root ./_demo_custom_dataset \
  --category custom \
  --model vision_template_ncc_map \
  --device cpu \
  --export-infer-config \
  --export-deploy-bundle \
  --json
```

Expected: exit code `0` and JSON containing `run_dir` and `deploy_bundle_dir`.

**Step 4: Run bundle replay smoke**

Run:

```bash
python3 -m pyimgano.bundle_cli run \
  runs/<new_run_dir>/deploy_bundle \
  --image-dir ./_demo_custom_dataset/test \
  --output-dir /tmp/pyimgano_bundle_followup_smoke \
  --json
```

Expected: `status=completed`, `processed=2`, `error=0`.

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py pyimgano/services/train_export_helpers.py \
  tests/test_train_export_helpers.py tests/test_train_service.py \
  tests/test_deploy_bundle_manifest.py tests/test_workbench_export_infer_config.py
git commit -m "refactor: thin deploy bundle assembly in train service"
```
