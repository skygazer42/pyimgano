# Train Export Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Harden the `pyimgano-train` export path by separating infer-config export, optional artifact emission, and deploy-bundle assembly into clearer internal stages without changing existing public behavior.

**Architecture:** Keep `pyimgano.train_cli` as the public argparse and output shell, but make `pyimgano.services.train_service` read as a staged export pipeline: load config, validate export request, run recipe, export infer-config artifacts, optionally assemble deploy bundle. Prefer extracting narrow helper modules or helper functions over broad workbench/reporting rewrites.

**Tech Stack:** Python 3.9+, argparse, JSON, existing `pyimgano.train_cli`, `pyimgano.services.train_service`, `pyimgano.reporting.deploy_bundle`, `pyimgano.reporting.calibration_card`, pytest.

---

### Task 1: Add failing tests for train export helper scaffolding

**Files:**
- Create: `tests/test_train_export_helpers.py`
- Test: `pyimgano/services/train_service.py`

**Step 1: Write the failing test**

```python
def test_require_run_dir_returns_path_for_valid_report():
    from pyimgano.services.train_export_helpers import require_run_dir
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyimgano.services.train_export_helpers'`

**Step 3: Write minimal implementation**

```python
def require_run_dir(report, *, deploy_bundle=False):
    return Path(str(report["run_dir"]))
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_export_helpers.py pyimgano/services/train_export_helpers.py
git commit -m "test: add train export helper scaffolding"
```

### Task 2: Extract run-dir and export-request validation helpers

**Files:**
- Create: `pyimgano/services/train_export_helpers.py`
- Modify: `pyimgano/services/train_service.py`
- Test: `tests/test_train_export_helpers.py`
- Test: `tests/test_train_service.py`

**Step 1: Write the failing test**

```python
def test_validate_export_request_requires_save_run_and_pixel_threshold_when_needed():
    from pyimgano.services.train_export_helpers import validate_export_request
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_train_service.py -q`
Expected: FAIL because helper is missing or not wired

**Step 3: Write minimal implementation**

```python
def validate_export_request(cfg, request):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_train_service.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_export_helpers.py pyimgano/services/train_service.py tests/test_train_export_helpers.py tests/test_train_service.py
git commit -m "refactor: extract train export validation helpers"
```

### Task 3: Add failing tests for infer-config optional artifact helper extraction

**Files:**
- Modify: `tests/test_train_export_helpers.py`
- Modify: `tests/test_workbench_export_infer_config.py`

**Step 1: Write the failing test**

```python
def test_build_optional_calibration_card_payload_preserves_prediction_defaults():
    from pyimgano.services.train_export_helpers import build_optional_calibration_card_payload
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def build_optional_calibration_card_payload(report, infer_config_payload):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_export_helpers.py tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py
git commit -m "test: cover optional train export artifacts"
```

### Task 4: Move optional infer-config artifact helpers behind `train_export_helpers`

**Files:**
- Modify: `pyimgano/services/train_export_helpers.py`
- Modify: `pyimgano/services/train_service.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Step 1: Write the failing test**

```python
def test_export_infer_artifacts_preserves_calibration_and_operator_contract_outputs():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_train_export_helpers.py -q`
Expected: FAIL because helper extraction changes export behavior

**Step 3: Write minimal implementation**

```python
calibration_card_payload = train_export_helpers.build_optional_calibration_card_payload(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_train_export_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py pyimgano/services/train_export_helpers.py tests/test_workbench_export_infer_config.py
git commit -m "refactor: extract train optional export helpers"
```

### Task 5: Add failing tests for deploy-bundle path rewrite helpers

**Files:**
- Modify: `tests/test_train_export_helpers.py`
- Modify: `tests/test_workbench_export_infer_config.py`

**Step 1: Write the failing test**

```python
def test_rewrite_bundle_paths_handles_relative_and_absolute_checkpoints():
    from pyimgano.services.train_export_helpers import rewrite_bundle_paths
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def rewrite_bundle_paths(...):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_export_helpers.py tests/test_train_export_helpers.py tests/test_workbench_export_infer_config.py
git commit -m "test: cover bundle path rewrite helpers"
```

### Task 6: Extract bundle path resolution and rewrite helpers from `_export_deploy_bundle`

**Files:**
- Modify: `pyimgano/services/train_export_helpers.py`
- Modify: `pyimgano/services/train_service.py`
- Test: `tests/test_workbench_export_infer_config.py`
- Test: `tests/test_deploy_bundle_manifest.py`

**Step 1: Write the failing test**

```python
def test_export_deploy_bundle_keeps_checkpoint_rewrite_contract_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL because helper extraction changes bundle checkpoint handling

**Step 3: Write minimal implementation**

```python
rewritten_payload = train_export_helpers.rewrite_bundle_paths(...)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py pyimgano/services/train_export_helpers.py tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py
git commit -m "refactor: extract deploy bundle path helpers"
```

### Task 7: Add failing tests for bundle metadata patch helpers

**Files:**
- Modify: `tests/test_train_export_helpers.py`
- Modify: `tests/test_deploy_bundle_manifest.py`

**Step 1: Write the failing test**

```python
def test_apply_bundle_manifest_metadata_updates_artifact_quality_fields():
    from pyimgano.services.train_export_helpers import apply_bundle_manifest_metadata
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL because helper is missing

**Step 3: Write minimal implementation**

```python
def apply_bundle_manifest_metadata(payload, manifest):
    ...
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_helpers.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_export_helpers.py tests/test_train_export_helpers.py tests/test_deploy_bundle_manifest.py
git commit -m "test: cover bundle metadata patch helpers"
```

### Task 8: Move bundle metadata patching behind helper functions

**Files:**
- Modify: `pyimgano/services/train_export_helpers.py`
- Modify: `pyimgano/services/train_service.py`
- Test: `tests/test_workbench_export_infer_config.py`
- Test: `tests/test_deploy_bundle_manifest.py`

**Step 1: Write the failing test**

```python
def test_export_deploy_bundle_preserves_artifact_quality_flags_after_helper_extraction():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py -q`
Expected: FAIL because helper extraction changes artifact_quality patching

**Step 3: Write minimal implementation**

```python
train_export_helpers.apply_bundle_manifest_metadata(bundle_payload, bundle_manifest)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add pyimgano/services/train_service.py pyimgano/services/train_export_helpers.py tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py
git commit -m "refactor: extract bundle metadata patch helpers"
```

### Task 9: Add failing tests for train export docs contract

**Files:**
- Create: `tests/test_train_export_docs_contract.py`
- Test: `docs/CLI_REFERENCE.md`
- Test: `README.md`

**Step 1: Write the failing test**

```python
def test_cli_reference_documents_train_export_infer_config_and_deploy_bundle():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_docs_contract.py -q`
Expected: FAIL because docs contract checks do not exist yet

**Step 3: Write minimal implementation**

```python
assert "--export-infer-config" in text
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_export_docs_contract.py
git commit -m "test: add train export docs contract checks"
```

### Task 10: Update docs for train export contract wording

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `README.md`
- Test: `tests/test_train_export_docs_contract.py`

**Step 1: Write the failing test**

```python
def test_readme_mentions_export_deploy_bundle_validation_flow():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_train_export_docs_contract.py -q`
Expected: FAIL because docs wording is not locked yet

**Step 3: Write minimal implementation**

```markdown
--export-infer-config
--export-deploy-bundle
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_train_export_docs_contract.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/CLI_REFERENCE.md README.md tests/test_train_export_docs_contract.py
git commit -m "docs: clarify train export contracts"
```

### Task 11: Add failing architecture tests for train export helper modules

**Files:**
- Modify: `tests/test_architecture_boundaries.py`
- Test: `pyimgano/services/train_export_helpers.py`

**Step 1: Write the failing test**

```python
def test_train_export_helper_modules_define_expected_public_exports():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: FAIL because the helper module is not covered

**Step 3: Write minimal implementation**

```python
__all__ = ["require_run_dir"]
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest --no-cov tests/test_architecture_boundaries.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_architecture_boundaries.py pyimgano/services/train_export_helpers.py
git commit -m "test: add train export helper boundary expectations"
```

### Task 12: Run the targeted train export verification suite

**Files:**
- Verify only:
  - `tests/test_train_service.py`
  - `tests/test_train_export_helpers.py`
  - `tests/test_workbench_export_infer_config.py`
  - `tests/test_deploy_bundle_manifest.py`
  - `tests/test_train_cli_dry_run.py`
  - `tests/test_train_export_docs_contract.py`
  - `tests/test_architecture_boundaries.py`

**Step 1: Run the focused suite**

```bash
python3 -m pytest --no-cov \
  tests/test_train_service.py \
  tests/test_train_export_helpers.py \
  tests/test_workbench_export_infer_config.py \
  tests/test_deploy_bundle_manifest.py \
  tests/test_train_cli_dry_run.py \
  tests/test_train_export_docs_contract.py \
  tests/test_architecture_boundaries.py -q
```

**Step 2: Verify it passes**

Expected: PASS with 0 failures

**Step 3: Commit**

```bash
git add .
git commit -m "test: verify train export hardening suite"
```

### Task 13: Run static verification on touched train/export files

**Files:**
- Verify only:
  - `pyimgano/train_cli.py`
  - `pyimgano/services/train_service.py`
  - `pyimgano/services/train_export_helpers.py`
  - `tests/test_train_export_helpers.py`
  - `tests/test_train_export_docs_contract.py`

**Step 1: Run compile verification**

```bash
python3 -m py_compile \
  pyimgano/train_cli.py \
  pyimgano/services/train_service.py \
  pyimgano/services/train_export_helpers.py
```

**Step 2: Run linter verification**

```bash
ruff check \
  pyimgano/train_cli.py \
  pyimgano/services/train_service.py \
  pyimgano/services/train_export_helpers.py \
  tests/test_train_export_helpers.py \
  tests/test_train_export_docs_contract.py
```

**Step 3: Verify both commands pass**

Expected: PASS with exit code 0

**Step 4: Commit**

```bash
git add .
git commit -m "chore: finalize train export hardening checks"
```
