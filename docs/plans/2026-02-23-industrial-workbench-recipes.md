# PyImgAno Industrial Workbench (CLI-first + Recipes) — Implementation Plan (40 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a recipe-driven “workbench layer” (CLI-first) so algorithm engineers can run **adaptation-first** workflows (and optionally **micro-finetune** a small supported subset) while producing **reproducible, comparable run artifacts**.

**Architecture:** `pyimgano-train` loads a JSON-first config, resolves a recipe from a recipe registry, and executes a standardized pipeline that writes artifacts under `runs/.../` (compatible with existing reporting). Recipes reuse existing components (`run_benchmark`, tiling, postprocess, evaluation) and optionally call a minimal micro-finetune runner.

**Tech Stack:** Python, NumPy, OpenCV, scikit-learn/PyOD, PyTorch (for finetune, optional), JSON/JSONL.

---

## Commit + Tag Rules (User Requirement)

Work happens on `feat/workbench-recipes` with small commits. Mainline releases happen every 10 tasks:

- **Feature branch:** 1 commit per task (easy review + bisect).
- **Milestones (tasks 1–10 / 11–20 / 21–30 / 31–40):**
  - open PR
  - **squash merge to `main`** (1 commit per milestone on `main`)
  - bump version + update `CHANGELOG.md`
  - tag + push

**Suggested milestone tags (current `main` is `v0.5.8`):**
- after task 10 → `v0.6.0`
- after task 20 → `v0.6.1`
- after task 30 → `v0.6.2`
- after task 40 → `v0.6.3`

---

## Config Rules

- **Config format:** JSON by default (`.json`).
- **YAML support:** optional (`.yaml/.yml`) only when `PyYAML` is installed.
- **Weights:** never included in wheels; any downloaded weights/checkpoints live on disk (cache/artifacts).

---

## Phase 1 (Tasks 1–10): Recipe registry + config IO + `pyimgano-train` scaffold

### Task 1: Add design doc (done)

**Files:**
- Added: `docs/plans/2026-02-23-industrial-workbench-recipes-design.md`

### Task 2: Add this implementation plan

**Files:**
- Create: `docs/plans/2026-02-23-industrial-workbench-recipes.md`

**Steps:**
1) Add plan file (this file).
2) Run: `python -m pytest -q tests/test_cli_smoke.py` (sanity)
3) Commit: `git commit -m "docs: add industrial workbench recipes plan"`

### Task 3: Add JSON-first config loader with optional YAML

**Files:**
- Create: `pyimgano/config/__init__.py`
- Create: `pyimgano/config/io.py`
- Test: `tests/test_config_io.py`

**Behavior:**
- `load_config(path)` supports `.json`, and supports `.yml/.yaml` only if `PyYAML` is installed.
- Unknown extension → `ValueError` with a clear message.

**Test first:** ensure:
- JSON loads round-trip.
- YAML load raises a clean `ImportError` install hint when `PyYAML` missing.

Run: `python -m pytest -q tests/test_config_io.py`

### Task 4: Add workbench config normalization (`WorkbenchConfig.from_dict`)

**Files:**
- Create: `pyimgano/workbench/__init__.py`
- Create: `pyimgano/workbench/config.py`
- Test: `tests/test_workbench_config.py`

**Scope (minimal, Phase-1):**
- Normalize the subset needed by the first recipe:
  - dataset: `name`, `root`, `category`, `resize`, `input_mode`, `limit_train`, `limit_test`
  - model: `name`, `device`, `preset`, `pretrained`, `contamination`, `model_kwargs`, `checkpoint_path`
  - output: `output_dir`, `save_run`, `per_image_jsonl`
  - global: `seed`, `recipe`

Run: `python -m pytest -q tests/test_workbench_config.py`

### Task 5: Add recipe protocol + registry

**Files:**
- Create: `pyimgano/recipes/__init__.py`
- Create: `pyimgano/recipes/protocol.py`
- Create: `pyimgano/recipes/registry.py`
- Test: `tests/test_recipe_registry.py`

**Behavior:**
- Similar ergonomics to `pyimgano.models.registry`:
  - `register_recipe(name, tags=..., metadata=..., overwrite=...)`
  - `list_recipes(tags=...)`
  - `recipe_info(name)` returns JSON-friendly payload

Run: `python -m pytest -q tests/test_recipe_registry.py`

### Task 6: Add workbench run directory helpers + paths

**Files:**
- Modify: `pyimgano/reporting/runs.py`
- Test: `tests/test_reporting_workbench_runs.py`

**Behavior:**
- Add `build_workbench_run_dir_name(dataset, recipe, model, category=None)`:
  - Format: `YYYYMMDD_HHMMSS_<dataset>_<recipe>_<model>[_<category>]`
- Add `WorkbenchRunPaths` with:
  - `run_dir`, `report_json`, `config_json`, `environment_json`
  - `categories_dir`, `checkpoints_dir`, `artifacts_dir`
- Add `build_workbench_run_paths(run_dir)`

Run: `python -m pytest -q tests/test_reporting_workbench_runs.py`

### Task 7: Implement builtin recipe `industrial-adapt` (wrap existing benchmark pipeline)

**Files:**
- Create: `pyimgano/recipes/builtin/__init__.py`
- Create: `pyimgano/recipes/builtin/industrial_adapt.py`
- Modify: `pyimgano/recipes/__init__.py` (import builtin recipes for side effects)
- Test: `tests/test_recipe_industrial_adapt_smoke.py`

**Behavior:**
- Uses `pyimgano.pipelines.run_benchmark.run_benchmark(...)`.
- Uses `WorkbenchConfig` values for dataset/model/output settings.
- Writes `config.json` + `environment.json` to the workbench run dir.
- Returns a JSON-friendly `report` payload that includes `run_dir`.

Run: `python -m pytest -q tests/test_recipe_industrial_adapt_smoke.py`

### Task 8: Add `pyimgano-train` CLI (recipe-driven)

**Files:**
- Create: `pyimgano/train_cli.py`
- Modify: `pyproject.toml` (`[project.scripts]`)
- Test: `tests/test_train_cli_list_recipes.py`

**CLI requirements (Phase-1):**
- `pyimgano-train --list-recipes` (text + `--json`)
- `pyimgano-train --recipe-info NAME` (text + `--json`)
- `pyimgano-train --config cfg.json` runs recipe (default recipe: `industrial-adapt`)
- Optional overrides: `--dataset/--root/--category/--model/--device`

Run: `python -m pytest -q tests/test_train_cli_list_recipes.py`

### Task 9: Add end-to-end smoke test for `pyimgano-train` on a tiny custom dataset

**Files:**
- Test: `tests/test_train_cli_smoke.py`

**Test behavior:**
- Create a minimal `custom` dataset on disk (same layout used by existing tests).
- Run `pyimgano.train_cli.main([...])` with a small config and `--output-dir`.
- Assert artifacts exist:
  - `report.json`
  - `config.json`
  - `environment.json`
  - `categories/<cat>/per_image.jsonl`

Run: `python -m pytest -q tests/test_train_cli_smoke.py`

### Task 10: Release `v0.6.0` (milestone tag)

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`
- Modify: `CHANGELOG.md`

**Steps:**
1) Bump version → `0.6.0`
2) Add changelog entry (“workbench recipes + pyimgano-train (scaffold)”)
3) Squash-merge Phase 1 to `main`
4) Tag: `git tag v0.6.0 && git push --tags`

---

## Phase 2 (Tasks 11–20): Adaptation-first standardization (tiling + postprocess + maps)

### Task 11: Add adaptation config (`TilingConfig`, `MapPostprocessConfig`)

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Create: `pyimgano/workbench/adaptation.py`
- Test: `tests/test_workbench_adaptation_config.py`

**Behavior:**
- Add optional `adaptation` section to workbench config:
  - `tiling`: `tile_size`, `stride`, `score_reduce`, `score_topk`, `map_reduce`
  - `postprocess`: mirrors `AnomalyMapPostprocess` fields
  - `save_maps`: bool (default false)

Run: `python -m pytest -q tests/test_workbench_adaptation_config.py`

### Task 12: Add anomaly-map saving utility (npy) + JSONL record extension

**Files:**
- Create: `pyimgano/workbench/maps.py`
- Test: `tests/test_workbench_maps_io.py`

**Behavior:**
- `save_anomaly_map_npy(out_dir, index, input_path, anomaly_map) -> path`
- Ensure stable naming (e.g. `maps/000012_<stem>.npy`)
- JSONL records can include:
  - `anomaly_map: {path, shape, dtype}`

Run: `python -m pytest -q tests/test_workbench_maps_io.py`

### Task 13: Implement config-driven tiling wrapper for inference

**Files:**
- Modify: `pyimgano/workbench/adaptation.py`
- Test: `tests/test_workbench_tiling_integration.py`

**Behavior:**
- `apply_tiling(detector, tiling_config)` returns `TiledDetector` when enabled.

Run: `python -m pytest -q tests/test_workbench_tiling_integration.py`

### Task 14: Implement config-driven anomaly-map postprocess

**Files:**
- Modify: `pyimgano/workbench/adaptation.py`
- Test: `tests/test_workbench_postprocess_integration.py`

**Behavior:**
- `build_postprocess(postprocess_config)` returns `AnomalyMapPostprocess` or `None`.

Run: `python -m pytest -q tests/test_workbench_postprocess_integration.py`

### Task 15: Standardize threshold calibration policy in workbench

**Files:**
- Create: `pyimgano/workbench/calibration.py`
- Test: `tests/test_workbench_calibration.py`

**Behavior:**
- Reuse `pyimgano.inference.api.calibrate_threshold` for adaptation recipes.
- Persist threshold used into `report.json` + `config.json`.

Run: `python -m pytest -q tests/test_workbench_calibration.py`

### Task 16: Add shared “fit + infer + evaluate + artifacts” helper for recipes

**Files:**
- Create: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_runner_smoke.py`

**Behavior:**
- A single helper that:
  - loads dataset split (train/test)
  - fits detector
  - calibrates threshold (if configured)
  - runs inference (`infer(...)`) with optional maps
  - writes JSONL + map artifacts if enabled
  - writes `report.json` and category reports

Run: `python -m pytest -q tests/test_workbench_runner_smoke.py`

### Task 17: Upgrade `industrial-adapt` recipe to use the new workbench runner

**Files:**
- Modify: `pyimgano/recipes/builtin/industrial_adapt.py`
- Test: `tests/test_recipe_industrial_adapt_maps.py`

**Behavior:**
- When model supports pixel maps and config `save_maps=true`, write maps and include them in JSONL.

Run: `python -m pytest -q tests/test_recipe_industrial_adapt_maps.py`

### Task 18: Add stable artifact schema versioning for train runs

**Files:**
- Modify: `pyimgano/reporting/report.py` (or workbench runner payload stamping)
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_schema_version.py`

**Behavior:**
- Ensure train/workbench run reports include:
  - `schema_version`
  - `timestamp_utc`
  - `pyimgano_version`

Run: `python -m pytest -q tests/test_workbench_schema_version.py`

### Task 19: Documentation for recipes + `pyimgano-train`

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Create: `docs/RECIPES.md`
- Modify: `README.md` (short section + link)

**Content:**
- Minimal JSON config example
- `--list-recipes` / `--recipe-info`
- Map saving + tiling knobs

Docs sanity: `python -m pytest -q tests/test_cli_smoke.py`

### Task 20: Release `v0.6.1`

Version bump + changelog + squash merge + tag.

---

## Phase 3 (Tasks 21–30): Micro-finetune (minimal) + checkpoints + “infer from run”

### Task 21: Define checkpoint layout + utility

**Files:**
- Create: `pyimgano/training/__init__.py`
- Create: `pyimgano/training/checkpointing.py`
- Test: `tests/test_training_checkpointing.py`

**Behavior:**
- Provide `save_checkpoint(detector, path)` best-effort:
  - prefer `detector.save_checkpoint(path)` when available
  - else if `detector` has `.model` torch module, save state_dict
  - else raise `NotImplementedError` with actionable message

Run: `python -m pytest -q tests/test_training_checkpointing.py`

### Task 22: Add micro-finetune runner (narrow scope)

**Files:**
- Create: `pyimgano/training/runner.py`
- Test: `tests/test_training_runner_smoke.py`

**Behavior:**
- A thin wrapper that:
  - sets seeds (reuse existing seed helpers)
  - calls `fit(...)` with recipe-configured kwargs (epochs/lr when supported)
  - returns metrics + timing

Run: `python -m pytest -q tests/test_training_runner_smoke.py`

### Task 23: Add builtin recipe `micro-finetune-autoencoder` (first supported finetune recipe)

**Files:**
- Create: `pyimgano/recipes/builtin/micro_finetune_autoencoder.py`
- Modify: `pyimgano/recipes/__init__.py` (register builtin)
- Test: `tests/test_recipe_micro_finetune_autoencoder_smoke.py`

**Behavior:**
- Trains a small supported set (start with `vision_ae` and/or `vision_vae`).
- Writes checkpoint under `checkpoints/`.
- Emits run report with checkpoint path metadata.

Run: `python -m pytest -q tests/test_recipe_micro_finetune_autoencoder_smoke.py`

### Task 24: Extend workbench config schema to include `training` section

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Test: `tests/test_workbench_training_config.py`

**Behavior:**
- `training` config includes:
  - `enabled` (bool)
  - `epochs` / `lr` (optional, model-dependent)
  - `checkpoint_name` (default `model.pt`)

Run: `python -m pytest -q tests/test_workbench_training_config.py`

### Task 25: Write checkpoints/artifacts from the workbench runner when training enabled

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_runner_checkpoints.py`

Run: `python -m pytest -q tests/test_workbench_runner_checkpoints.py`

### Task 26: Add `pyimgano-infer --from-run RUN_DIR` (best-effort)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Create: `pyimgano/workbench/load_run.py`
- Test: `tests/test_infer_cli_from_run.py`

**Behavior:**
- Loads `config.json` from run dir
- Resolves model + checkpoint (if present)
- Applies threshold calibration value (if persisted)
- Runs inference on `--input ...`

Run: `python -m pytest -q tests/test_infer_cli_from_run.py`

### Task 27: Harden `--from-run` error handling and messaging

**Files:**
- Modify: `pyimgano/workbench/load_run.py`
- Test: `tests/test_infer_cli_from_run_errors.py`

Run: `python -m pytest -q tests/test_infer_cli_from_run_errors.py`

### Task 28: (Optional) Add anomalib training recipe skeleton behind extra

**Files:**
- Create: `pyimgano/recipes/builtin/anomalib_train.py`
- Test: `tests/test_recipe_anomalib_train_optional.py`

**Notes:**
- Must be fully gated behind `pyimgano[anomalib]`.
- Tests should skip cleanly when anomalib is not installed.

### Task 29: Docs: micro-finetune + checkpoints + infer-from-run

**Files:**
- Modify: `docs/CLI_REFERENCE.md`
- Modify: `docs/RECIPES.md`

### Task 30: Release `v0.6.2`

Version bump + changelog + squash merge + tag.

---

## Phase 4 (Tasks 31–40): UX polish + examples + integration coverage

### Task 31: Add `pyimgano-train --dry-run` (validate + print effective config)

**Files:**
- Modify: `pyimgano/train_cli.py`
- Test: `tests/test_train_cli_dry_run.py`

Run: `python -m pytest -q tests/test_train_cli_dry_run.py`

### Task 32: Add `pyimgano-train --export-infer-config` artifact

**Files:**
- Modify: `pyimgano/train_cli.py`
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Behavior:**
- Writes `artifacts/infer_config.json` containing only what inference needs.

Run: `python -m pytest -q tests/test_workbench_export_infer_config.py`

### Task 33: Persist seeds/provenance into `config.json` + report payload

**Files:**
- Modify: `pyimgano/workbench/config.py`
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_repro_provenance.py`

Run: `python -m pytest -q tests/test_workbench_repro_provenance.py`

### Task 34: Add `recipe_info` JSON output parity (text + `--json`)

**Files:**
- Modify: `pyimgano/train_cli.py`
- Test: `tests/test_train_cli_recipe_info_json.py`

Run: `python -m pytest -q tests/test_train_cli_recipe_info_json.py`

### Task 35: Add example configs (`examples/configs/*.json`)

**Files:**
- Create: `examples/configs/industrial_adapt_fast.json`
- Create: `examples/configs/industrial_adapt_maps_tiling.json`
- Create: `examples/configs/micro_finetune_autoencoder.json`
- Test: `tests/test_examples_configs_load.py`

Run: `python -m pytest -q tests/test_examples_configs_load.py`

### Task 36: Docs: “Industrial Workbench” positioning + quickstart

**Files:**
- Modify: `README.md`
- Create: `docs/WORKBENCH.md`

### Task 37: Add integration smoke: train → infer-from-run on tiny dataset

**Files:**
- Test: `tests/test_integration_workbench_train_then_infer.py`

Run: `python -m pytest -q tests/test_integration_workbench_train_then_infer.py`

### Task 38: Packaging + cache policy documentation

**Files:**
- Modify: `README.md` (cache/weights policy)
- Modify: `docs/RECIPES.md` (where checkpoints live, how to reuse)
- (Optional) Modify: `pyimgano/cache/__init__.py` / new helper module if needed

### Task 39: Final cleanup (lint/format) + full test sweep

Run:
- `python -m pytest -q`

### Task 40: Release `v0.6.3`

Version bump + changelog + squash merge + tag.

