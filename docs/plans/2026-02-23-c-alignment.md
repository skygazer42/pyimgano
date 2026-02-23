# PyImgAno “C-mode” Alignment (PyOD + sklearn + torch-like) — Implementation Plan (40 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano` simultaneously (1) sklearn/PyOD-like in core estimator behavior and (2) industrial-usable via consistent pipelines/CLIs/artifacts.

**Architecture:** Contract-first core + registry introspection + shared CLI tooling + reproducible run artifacts. Keep optional heavy backends gated behind extras.

**Tech Stack:** Python, NumPy, OpenCV, scikit-learn, PyOD, PyTorch/TorchVision (optional), JSON/JSONL.

---

## Commit + Tag Rules (User Requirement)

- **One commit per task** (40 commits total).
- After commits **#10 / #20 / #30 / #40**:
  - bump version
  - update `CHANGELOG.md`
  - create tag and push (`v0.5.5`, `v0.5.6`, `v0.5.7`, `v0.5.8`)

---

## Phase 1 (Commits 1–10): CLI unification + basic introspection

### Task 1: Add design doc (done)

### Task 2: Add this implementation plan

**Files:**
- Create: `docs/plans/2026-02-23-c-alignment.md`

**Step 1:** Add plan file.  
**Step 2:** `pytest -q tests/test_cli_discovery.py` (sanity)  
**Step 3:** Commit `docs:` message.

### Task 3: Introduce `pyimgano.cli_common` shared helpers

**Files:**
- Create: `pyimgano/cli_common.py`
- Test: `tests/test_cli_common.py`

**Test first:** verify:
- `_parse_model_kwargs` JSON parsing errors are stable
- `_merge_checkpoint_path` conflict detection works
- `build_model_kwargs` filters unsupported kwargs

Run: `pytest -q tests/test_cli_common.py`

### Task 4: Migrate `pyimgano-infer` to use `cli_common`

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_smoke.py`

Run: `pytest -q tests/test_infer_cli_smoke.py`

### Task 5: Migrate `pyimgano-robust-benchmark` to use `cli_common`

**Files:**
- Modify: `pyimgano/robust_cli.py`
- Test: `tests/test_robust_cli_smoke.py`

Run: `pytest -q tests/test_robust_cli_smoke.py`

### Task 6: Add registry model introspection helper (`model_info`)

**Files:**
- Create: `pyimgano/models/introspection.py`
- Modify: `pyimgano/models/registry.py`
- Test: `tests/test_registry_introspection.py`

Run: `pytest -q tests/test_registry_introspection.py`

### Task 7: Add computed capabilities (input modes, pixel maps, checkpoint, save/load)

**Files:**
- Create: `pyimgano/models/capabilities.py`
- Test: `tests/test_model_capabilities.py`

Run: `pytest -q tests/test_model_capabilities.py`

### Task 8: Add `--list-categories` to `pyimgano-benchmark`

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_list_categories.py`

Run: `pytest -q tests/test_cli_list_categories.py`

### Task 9: Centralize JSON-serializable conversion (`to_jsonable`)

**Files:**
- Create: `pyimgano/utils/jsonable.py`
- Modify: `pyimgano/reporting/report.py`
- Modify: `pyimgano/cli.py` (use shared helper)
- Test: `tests/test_jsonable.py`

Run: `pytest -q tests/test_jsonable.py tests/test_reporting_json.py`

### Task 10: Release `v0.5.5`

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`

**Steps:**
- Bump version → `0.5.5`
- Changelog entry for Phase 1 changes
- Tag: `git tag v0.5.5`

---

## Phase 2 (Commits 11–20): Dataset/input clarity + run artifacts schema

### Task 11: Introduce `pyimgano.io.image` helpers (read/resize/colorspace)

**Files:**
- Create: `pyimgano/io/image.py`
- Create: `pyimgano/io/__init__.py`
- Test: `tests/test_io_image.py`

### Task 12: Route benchmark dataset loaders through `pyimgano.io`

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_datasets_paths.py` (and any dataset tests that exist)

### Task 13: Add `CustomDataset.validate_structure()`

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_custom_dataset_validation.py`

### Task 14: `pyimgano-benchmark --dataset custom` runs validation early

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_oneclick_pipeline_cli.py`

### Task 15: Add a dataset catalog (`pyimgano.datasets.catalog`)

**Files:**
- Create: `pyimgano/datasets/catalog.py`
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_dataset_catalog.py`

### Task 16: Pipeline supports `input_mode=paths|numpy`

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_oneclick_pipeline_cli.py`

### Task 17: CLI adds `--input-mode` for benchmark runner

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke.py`

### Task 18: Add `environment.json` artifact writing

**Files:**
- Create: `pyimgano/reporting/environment.py`
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_reporting_environment.py`

### Task 19: Report schema versioning (`schema_version`, timestamp, pyimgano_version)

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Modify: `pyimgano/reporting/report.py`
- Test: `tests/test_oneclick_pipeline_cli.py`

### Task 20: Release `v0.5.6`

Version bump + changelog + tag.

---

## Phase 3 (Commits 21–30): Reproducibility + serialization + sklearn adapter

### Task 21: Add run-level seed support (`--seed`) and persist in config.json

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_repro_seed.py`

### Task 22: Add `save_detector/load_detector` (pickle-based, classical safe set)

**Files:**
- Create: `pyimgano/serialization/__init__.py`
- Create: `pyimgano/serialization/pickle.py`
- Test: `tests/test_serialization_pickle.py`

### Task 23: Pipeline option `--save-detector`

**Files:**
- Modify: `pyimgano/cli.py`
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_save_load_detector_cli.py`

### Task 24: Pipeline option `--load-detector` (skip fit)

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_save_load_detector_cli.py`

### Task 25: Add sklearn-compatible adapter

**Files:**
- Create: `pyimgano/sklearn_adapter.py`
- Test: `tests/test_sklearn_adapter.py`

### Task 26: Ensure adapter supports `sklearn.base.clone`

**Files:**
- Modify: `pyimgano/sklearn_adapter.py`
- Test: `tests/test_sklearn_adapter.py`

### Task 27: Add “contract tests” for representative detectors

**Files:**
- Create: `tests/contracts/test_detector_contract.py`

### Task 28: Add deterministic naming / run dir collision policy docs

**Files:**
- Modify: `docs/EVALUATION_AND_BENCHMARK.md`

### Task 29: Add docs for sklearn integration

**Files:**
- Create: `docs/SKLEARN_INTEGRATION.md`
- Modify: `README.md`

### Task 30: Release `v0.5.7`

Version bump + changelog + tag.

---

## Phase 4 (Commits 31–40): Performance/UX polish + audits + CI gates

### Task 31: Add feature cache (`--cache-dir`) for classical detectors

**Files:**
- Create: `pyimgano/cache/features.py`
- Modify: `pyimgano/models/baseml.py`
- Test: `tests/test_feature_cache.py`

### Task 32: Report timing breakdown in reports

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_report_timing.py`

### Task 33: Unify user-facing errors (actionable install hints)

**Files:**
- Modify: `pyimgano/utils/optional_deps.py`
- Modify: CLI surfaces to re-raise with context
- Test: `tests/test_backend_hardening.py`

### Task 34: Add `tools/audit_public_api.py`

**Files:**
- Create: `tools/audit_public_api.py`
- Test: `tests/test_tools_audit_public_api.py`

### Task 35: Add `tools/audit_registry.py`

**Files:**
- Create: `tools/audit_registry.py`
- Test: `tests/test_tools_audit_registry.py`

### Task 36: Wire audits into CI

**Files:**
- Modify: `.github/workflows/ci.yml`

### Task 37: Add unified CLI reference doc

**Files:**
- Create: `docs/CLI_REFERENCE.md`

### Task 38: README cleanup: industrial workflows section consolidation

**Files:**
- Modify: `README.md`

### Task 39: Migration notes / compatibility callouts

**Files:**
- Modify: `docs/COMPARISON.md` or add `docs/MIGRATION.md`

### Task 40: Release `v0.5.8` + merge to `main`

**Steps:**
- Bump version → `0.5.8`
- Tag and push
- Merge branch to `main`
- Push `main`

