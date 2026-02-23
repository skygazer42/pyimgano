# Manifest Dataset (JSONL) — Implementation Plan (40 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an industrial-grade JSONL manifest dataset source (multi-category, group-aware deterministic split) and wire it into the workbench so algorithm engineers can run `pyimgano-train` with `dataset.name="manifest"`.

**Architecture:** A dedicated `pyimgano.datasets.manifest` module handles JSONL parsing, path resolution, validation, and split policy. Workbench calls a single split loader that emits `train/calibration/test` paths + labels + optional masks, producing standard run artifacts and metrics.

**Tech Stack:** Python, JSONL, NumPy, OpenCV, pytest.

---

## User Decisions (Locked In)

- Single JSONL supports **multiple categories** (`category` required per record).
- `split` is optional; if missing → `split_policy` auto-split (deterministic via seed).
- `group_id` is optional; if present → group-aware split (no leakage).
- `label`:
  - `split=="train"` missing → defaults to `0`
  - `split=="test"` missing → invalid (required for meaningful metrics)
- `mask_path` optional; pixel metrics computed only when safe.
- Relative path resolution: manifest-dir first, then fallback to `dataset.root`.
- Auto-split scope: **per-category**.

---

## Commit Strategy

- Prefer small, reviewable commits (1–5 tasks per commit).
- Push to `origin/main` after all tasks complete.
- Optional: bump patch version at the end if feature is user-facing.

---

## Phase 1 (Tasks 1–10): Config schema + manifest parsing scaffolding

### Task 1: Add design doc (done)

### Task 2: Add this implementation plan

**Files:**
- Create: `docs/plans/2026-02-23-manifest-dataset.md`

**Steps:**
- Add plan file.
- Commit: `docs: add manifest dataset implementation plan`

### Task 3: Mark design as approved + resolve scope decision

**Files:**
- Modify: `docs/plans/2026-02-23-manifest-dataset-design.md`

**Steps:**
- Update status to Approved.
- Replace “Open decision” with resolved `scope="category"`.
- Commit: `docs: approve manifest dataset design`

### Task 4: Extend workbench dataset config with `manifest_path`

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_workbench_config.py`

**Steps:**
- Add `manifest_path: str | None` to `DatasetConfig`.
- Parse `dataset.manifest_path` when present.
- Validate: if `dataset.name=="manifest"` then `manifest_path` is required.
- Run: `pytest -q tests/test_workbench_config.py`
- Commit.

### Task 5: Add `SplitPolicyConfig` to workbench config

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_workbench_config.py`

**Steps:**
- Add `dataset.split_policy` object (optional) with:
  - `mode` (default `"benchmark"`)
  - `scope` (default `"category"`)
  - `seed` (default: top-level `seed`)
  - `test_normal_fraction` (default `0.2`)
- Validate ranges and allowed strings.
- Run targeted tests.
- Commit.

### Task 6: Introduce `pyimgano.datasets.manifest` module (records + validation)

**Files:**
- Create: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_records.py`

**Steps:**
- Implement JSONL reader (one JSON object per line).
- Validate required fields (`image_path`, `category`).
- Validate `split` ∈ {train,val,test} when present.
- Validate `label` ∈ {0,1} when present.
- Run tests and commit.

### Task 7: Add manifest path resolution helper

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_paths.py`

**Steps:**
- Implement:
  - absolute path pass-through
  - relative path: manifest-dir first, then dataset.root fallback
- Add tests for both resolution orders.
- Commit.

### Task 8: Add `list_manifest_categories(manifest_path)`

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_categories.py`

**Steps:**
- Return stable sorted unique categories.
- Commit.

### Task 9: Implement deterministic auto-split policy (per-category)

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_split_policy.py`

**Steps:**
- Implement `split_policy.mode=="benchmark"`:
  - label==1 → test
  - else → train vs test(normal) by `test_normal_fraction`
- Deterministic with `seed`.
- Preserve stable ordering of outputs (paths remain stable).
- Commit.

### Task 10: Implement group-aware split (`group_id` optional)

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_group_split.py`

**Steps:**
- Group semantics:
  - any anomaly in group → entire group in test
  - remaining all-normal groups split by fraction deterministically
- Validate: explicit conflicting splits inside a group → error.
- Commit.

---

## Phase 2 (Tasks 11–20): Workbench integration + category=all correctness

### Task 11: Fix `run_workbench` category="all" behavior

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_runner_all_categories.py`

**Steps:**
- Implement the all-category branch:
  - compute `categories`
  - run per-category
  - compute mean/std of core metrics
  - write run-level `report.json`
- Remove any dead/unreachable code paths.
- Commit.

### Task 12: Remove unreachable code under `build_infer_config_payload`

**Files:**
- Modify: `pyimgano/workbench/runner.py`

**Steps:**
- Delete dead code after `return` in `build_infer_config_payload`.
- Ensure mypy/linters won’t flag undefined vars.
- Commit.

### Task 13: Add manifest split loader for workbench (`load_manifest_benchmark_split`)

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_manifest_split.py`

**Steps:**
- Add helper that returns:
  - `train_paths`, `calibration_paths`, `test_paths`
  - `test_labels`, `test_masks` (optional)
  - `pixel_skip_reason` (optional)
- Workbench runner uses calibration_paths for threshold calibration when present.
- Commit.

### Task 14: Wire `dataset.name=="manifest"` into workbench `_load_split_paths`

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_manifest_e2e_smoke.py`

**Steps:**
- If dataset is manifest:
  - load split from manifest loader
  - ignore `root` for category discovery except as fallback path resolver
- Commit.

### Task 15: Add pixel-metrics gating + explicit skip reason in report

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_manifest_pixel_gating.py`

**Steps:**
- If any anomaly test record missing mask:
  - do not pass `pixel_labels/pixel_scores` to `evaluate_detector`
  - attach `pixel_metrics_status: {"enabled": false, "reason": "..."}`
- Commit.

### Task 16: Add manifest category discovery for workbench

**Files:**
- Modify: `pyimgano/workbench/runner.py`

**Steps:**
- For `dataset.name=="manifest"` and `dataset.category=="all"`:
  - call `list_manifest_categories(manifest_path)`
- Commit.

### Task 17: Update builtin recipes to accept manifest datasets unchanged

**Files:**
- Modify: `pyimgano/recipes/builtin/industrial_adapt.py`
- Modify: `pyimgano/recipes/builtin/micro_finetune_autoencoder.py`
- Test: `tests/test_recipe_manifest_compat.py`

**Steps:**
- Ensure recipes don’t assume benchmark dataset semantics beyond split loader.
- Commit.

### Task 18: Add example manifest config

**Files:**
- Create: `examples/configs/manifest_industrial_adapt_fast.json`

**Steps:**
- Include `dataset.name="manifest"`, `dataset.manifest_path`, `split_policy`.
- Commit.

### Task 19: Document manifest dataset usage in workbench docs

**Files:**
- Modify: `docs/WORKBENCH.md`
- Create: `docs/MANIFEST_DATASET.md`

**Steps:**
- Add schema, split policy, group_id guidance.
- Commit.

### Task 20: Add README mention + quick snippet

**Files:**
- Modify: `README.md`

**Steps:**
- Add short section: “Manifest dataset (JSONL)”.
- Link to `docs/MANIFEST_DATASET.md`.
- Commit.

---

## Phase 3 (Tasks 21–30): Dataset factory + CLI ergonomics

### Task 21: Add a `ManifestDataset` wrapper compatible with `load_dataset`

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Test: `tests/test_manifest_dataset_factory.py`

**Steps:**
- Add `ManifestDataset` implementing:
  - `get_train_paths/get_test_paths`
  - `get_train_data/get_test_data` (best-effort, read images)
  - `list_categories` (from manifest)
- Commit.

### Task 22: Extend `load_dataset(...)` factory to include `"manifest"`

**Files:**
- Modify: `pyimgano/utils/datasets.py`
- Modify: `pyimgano/datasets/benchmarks.py` (re-export if needed)
- Test: `tests/test_manifest_dataset_factory.py`

**Steps:**
- Support `load_dataset("manifest", root, category=..., manifest_path=..., ...)`.
- Commit.

### Task 23: Add manifest support to dataset catalog listing (optional)

**Files:**
- Modify: `pyimgano/datasets/catalog.py`
- Test: `tests/test_dataset_catalog_manifest.py`

**Steps:**
- Add dataset `"manifest"` category listing via `manifest_path` when provided.
- Keep backwards compatibility for existing datasets.
- Commit.

### Task 24: Extend CLI list-categories to support manifest

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_list_categories_manifest.py`

**Steps:**
- Add `--manifest-path` flag used when `--dataset manifest`.
- Commit.

### Task 25: Add `pyimgano-manifest` generator CLI (optional)

**Files:**
- Create: `pyimgano/manifest_cli.py`
- Modify: `pyproject.toml` (script entrypoint)
- Docs: `docs/MANIFEST_DATASET.md`

**Steps:**
- Generate a JSONL manifest from a directory tree (custom layout).
- Keep output stable + sorted.
- Commit.

### Task 26: Add tests for manifest generator

**Files:**
- Create: `tests/test_manifest_cli_generate.py`

**Steps:**
- Generate from temp dirs; verify JSONL records and relative paths.
- Commit.

### Task 27: Add config validation in `pyimgano-train --dry-run` for manifest

**Files:**
- Modify: `pyimgano/train_cli.py`
- Test: `tests/test_train_cli_manifest_dry_run.py`

**Steps:**
- Ensure `manifest_path` exists and is readable.
- Optionally preview categories count.
- Commit.

### Task 28: Add helpful error messages for common manifest mistakes

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_manifest_errors.py`

**Steps:**
- Missing required fields
- Invalid split/label
- Conflicting group splits
- Missing test labels
- Commit.

### Task 29: Add manifest-specific unit tests for determinism + ordering

**Files:**
- Modify: `tests/test_manifest_split_policy.py`

**Steps:**
- Same manifest + seed → same split.
- Different seed → different split (when choices exist).
- Output order stable.
- Commit.

### Task 30: Add integration smoke: manifest run_dir → infer-from-run

**Files:**
- Modify: `tests/test_train_infer_from_run_smoke.py`

**Steps:**
- Create tiny manifest + dummy images.
- Train with a lightweight model preset.
- Infer from run and ensure JSONL output writes.
- Commit.

---

## Phase 4 (Tasks 31–40): Polish + release hygiene

### Task 31: Ensure per-image JSONL includes manifest metadata (optional)

**Files:**
- Modify: `pyimgano/workbench/runner.py`

**Steps:**
- When input is manifest paths, attach `meta` fields when present.
- Keep schema backwards compatible.
- Commit.

### Task 32: Ensure pixel mask resizing aligns with workbench resize

**Files:**
- Modify: `pyimgano/datasets/manifest.py`
- Test: `tests/test_manifest_masks_resize.py`

**Steps:**
- Load mask as gray, resize with mask interpolation, binarize.
- Commit.

### Task 33: Add docs example: converting “custom layout” to manifest

**Files:**
- Modify: `docs/MANIFEST_DATASET.md`

**Steps:**
- Show directory → manifest generator usage (if Task 25 implemented).
- Commit.

### Task 34: Update CLI reference docs

**Files:**
- Modify: `docs/CLI_REFERENCE.md`

**Steps:**
- Document new flags (`--manifest-path`, generator CLI if added).
- Commit.

### Task 35: Add changelog entry

**Files:**
- Modify: `CHANGELOG.md`

**Steps:**
- Note: manifest dataset + workbench all-category fix.
- Commit.

### Task 36: Run formatter/lints (best-effort)

**Steps:**
- Run: `python -m black pyimgano tests`
- Run: `python -m ruff pyimgano tests` (if configured)
- Commit formatting-only changes if any.

### Task 37: Run tests (best-effort)

**Steps:**
- Create venv if needed.
- Run: `pytest -q`
- Fix only failures caused by manifest work.

### Task 38: Optional version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py` (if version exported)

**Steps:**
- Bump patch version (e.g., `0.6.4`).
- Commit.

### Task 39: Push to origin

**Steps:**
- `git push origin main`

### Task 40: Optional tag push

**Steps:**
- `git tag v0.6.4` (if version bump done)
- `git push origin v0.6.4`

