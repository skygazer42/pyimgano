# Industrial Workflow Hardening Implementation Plan (40 Tasks)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `pyimgano` reliably usable in industrial pipelines by hardening the
workbench-first workflow: preflight validation, deliverable inference configs,
benchmark/workbench alignment, and stable artifacts.

**Architecture:** Add a workbench preflight module (`pyimgano.workbench.preflight`)
and a first-class inference-config loader for `pyimgano-infer --infer-config`.
Extend `pyimgano-benchmark` to run manifest datasets in paths mode with the same
split + pixel-metric gating semantics as workbench.

**Tech Stack:** Python, pytest, JSON/JSONL, NumPy, OpenCV (optional), pathlib.

---

## Locked-in Decisions

- Workbench-first: workbench artifacts are the canonical industrial contract.
- No HTTP service layer in this plan.
- Additive changes only (avoid breaking artifact schema; keep compatibility).
- Prefer clear errors + preflight over “best-effort silently”.

---

## Phase 0 (Tasks 1–2): Planning artifacts

### Task 1: Add design doc (done)

### Task 2: Add this implementation plan

**Files:**
- Create: `docs/plans/2026-02-24-industrial-workflow-hardening.md`

**Steps:**
- Add plan file.
- Commit: `docs: add industrial workflow hardening plan`

---

## Phase 1 (Tasks 3–12): Preflight validation (industrial data health)

### Task 3: Introduce `pyimgano.workbench.preflight` scaffolding

**Files:**
- Create: `pyimgano/workbench/preflight.py`
- Test: `tests/test_workbench_preflight_manifest.py`

**Steps:**
- Define JSON-friendly types:
  - `IssueSeverity = Literal["error","warning","info"]`
  - `PreflightIssue(code, severity, message, context)`
  - `PreflightReport(dataset, category, summary, issues)`
- Add a stub `run_preflight(config: WorkbenchConfig) -> PreflightReport`.
- Test: preflight returns a `PreflightReport` for a minimal manifest config.
- Commit.

### Task 4: Manifest preflight: record counts + label/mask coverage

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Test: `tests/test_workbench_preflight_manifest.py`

**Steps:**
- For manifest dataset, parse records and compute:
  - total records per category
  - per split counts (explicit split only; “auto” counted separately)
  - label distribution for explicit test rows
  - anomaly mask coverage for explicit/assigned test anomalies
- Add summary fields like:
  - `{"counts": {...}, "mask_coverage": {...}}`
- Commit.

### Task 5: Manifest preflight: missing files + duplicate paths

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Test: `tests/test_workbench_preflight_manifest.py`

**Steps:**
- Detect missing `image_path` files (resolve with manifest-dir first, then root fallback).
- Detect missing mask files when `mask_path` provided.
- Detect duplicates of resolved image path within a category.
- Emit issues with stable codes:
  - `MANIFEST_MISSING_IMAGE`
  - `MANIFEST_MISSING_MASK`
  - `MANIFEST_DUPLICATE_IMAGE`
- Commit.

### Task 6: Manifest preflight: group conflicts and leakage checks

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Test: `tests/test_workbench_preflight_manifest.py`

**Steps:**
- Validate group invariants for explicit split:
  - conflicting `split` inside same `group_id` → `error`
  - anomaly in `train`/`val` group → `error`
- Emit issue codes:
  - `MANIFEST_GROUP_SPLIT_CONFLICT`
  - `MANIFEST_GROUP_ANOMALY_IN_TRAIN`
- Commit.

### Task 7: Preflight for non-manifest datasets (best-effort)

**Files:**
- Modify: `pyimgano/workbench/preflight.py`
- Test: `tests/test_workbench_preflight_non_manifest.py`

**Steps:**
- For `custom`: reuse structure validation.
- For benchmark datasets: validate root exists + categories can be listed.
- Emit issue codes:
  - `DATASET_ROOT_MISSING`
  - `DATASET_CATEGORY_EMPTY`
- Commit.

### Task 8: Add `pyimgano-train --preflight` mode

**Files:**
- Modify: `pyimgano/train_cli.py`
- Test: `tests/test_train_cli_preflight.py`

**Steps:**
- Add `--preflight` flag (requires `--config`).
- Behavior:
  - prints JSON: `{"preflight": <report>}`
  - exit 0 when no `error` severity issues
  - exit 2 when any `error` severity issue exists
- Keep `--dry-run` output stable.
- Commit.

### Task 9: Document preflight in CLI docs

**Files:**
- Modify: `docs/CLI_REFERENCE.md`

**Steps:**
- Add a short `pyimgano-train --preflight` section with example.
- Commit.

### Task 10: Add a manifest-first industrial config template (preflight + run)

**Files:**
- Create: `examples/configs/manifest_industrial_workflow_balanced.json`
- Test: `tests/test_examples_configs_load.py` (if needed)

**Steps:**
- Add a minimal manifest config that runs with `industrial-adapt`.
- Commit.

### Task 11: Add dataset summary block to workbench reports

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_report_dataset_summary.py`

**Steps:**
- Add `dataset_summary` to per-category report:
  - counts for train/calibration/test
  - anomaly ratio in test
  - pixel metrics enabled/disabled (and reason)
- Commit.

### Task 12: Ensure manifest `meta` is also attached to per-image JSONL in benchmark (if supported)

**Files:**
- (Optional) Modify: `pyimgano/pipelines/run_benchmark.py` and loader
- Test: `tests/test_benchmark_manifest_meta_jsonl.py`

**Steps:**
- If benchmark manifest run is implemented in Phase 3, propagate `meta` into per-image JSONL.
- Commit.

---

## Phase 2 (Tasks 13–24): Inference delivery contract (`--infer-config`)

### Task 13: Add `pyimgano.inference.config` loader (infer-config parsing)

**Files:**
- Create: `pyimgano/inference/config.py`
- Test: `tests/test_infer_config_loader.py`

**Steps:**
- Implement:
  - `load_infer_config(path) -> dict`
  - `select_infer_category(payload, category)` for multi-category
  - checkpoint path resolution relative to config file dir
- Commit.

### Task 14: Add `pyimgano-infer --infer-config` and `--infer-category`

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Steps:**
- Add flags:
  - `--infer-config PATH` (mutually exclusive with `--model`/`--from-run`)
  - `--infer-category NAME` (when infer-config has multiple categories)
- Load model/device/preset/kwargs from config (allow CLI overrides to win).
- Apply threshold + checkpoint when present.
- Commit.

### Task 15: Integration: train → export infer-config → infer using infer-config

**Files:**
- Modify: `tests/test_integration_workbench_train_then_infer.py`

**Steps:**
- Add a test that:
  - runs `pyimgano-train --export-infer-config`
  - calls `pyimgano-infer --infer-config artifacts/infer_config.json`
- Commit.

### Task 16: Document infer-config usage

**Files:**
- Modify: `docs/CLI_REFERENCE.md`

**Steps:**
- Add `pyimgano-infer --infer-config` section.
- Commit.

### Task 17: Make exported infer-config portable (relative checkpoint paths)

**Files:**
- Modify: `pyimgano/workbench/runner.py` (export helper)
- Test: `tests/test_workbench_export_infer_config.py`

**Steps:**
- Ensure `checkpoint.path` is relative to `run_dir` when possible.
- Ensure infer-config loader can resolve it relative to the infer-config file.
- Commit.

### Task 18: Add “threshold provenance” to infer-config

**Files:**
- Modify: `pyimgano/workbench/runner.py`
- Test: `tests/test_workbench_export_infer_config.py`

**Steps:**
- Add fields describing threshold calibration:
  - quantile used and/or contamination-derived default
- Keep backward compat (optional field).
- Commit.

### Task 19: Add strict errors for ambiguous category selection

**Files:**
- Modify: `pyimgano/inference/config.py`
- Test: `tests/test_infer_config_loader.py`

**Steps:**
- If payload has multiple categories and none specified → error listing categories.
- Commit.

### Task 20: Add `pyimgano-infer --print-effective-config` (optional)

**Files:**
- Modify: `pyimgano/infer_cli.py`
- Test: `tests/test_infer_cli_infer_config.py`

**Steps:**
- Print the merged effective model/adaptation/threshold config as JSON and exit.
- Commit.

### Task 21: Add `--from-run` parity by reusing infer-config loader internally

**Files:**
- Modify: `pyimgano/infer_cli.py`

**Steps:**
- Refactor: `--from-run` code path builds an in-memory infer-config payload and reuses the same application logic.
- Commit.

### Task 22: Add docs “Industrial quick path” (manifest → preflight → train → export → infer)

**Files:**
- Create: `docs/INDUSTRIAL_QUICKPATH.md`

**Steps:**
- Provide a short step-by-step with commands.
- Commit.

### Task 23: Add a “deploy bundle” note (no weights in wheel; caching paths)

**Files:**
- Modify: `README.md`

**Steps:**
- Add a small section describing how to ship `infer_config.json` + checkpoint.
- Commit.

### Task 24: Add smoke test for infer-config with relative checkpoint outside run dir

**Files:**
- Create: `tests/test_infer_config_relative_checkpoint.py`

**Steps:**
- Write an infer-config file in tmp dir, checkpoint in sibling dir, ensure resolve works.
- Commit.

---

## Phase 3 (Tasks 25–34): Benchmark/workbench alignment for manifest

### Task 25: Make `pyimgano-benchmark` runnable for `--dataset manifest`

**Files:**
- Modify: `pyimgano/pipelines/mvtec_visa.py` (split loader)
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Steps:**
- Add dataset name `"manifest"` to the benchmark pipeline split loader (paths-only).
- Require `--manifest-path` when dataset is manifest.
- Commit.

### Task 26: Add manifest split policy flags to benchmark CLI

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Steps:**
- Add flags:
  - `--manifest-test-normal-fraction` (default 0.2)
  - `--manifest-split-seed` (defaults to `--seed` or 0)
- Commit.

### Task 27: Benchmark manifest should gate pixel metrics consistently

**Files:**
- Modify: `pyimgano/pipelines/mvtec_visa.py` (evaluation glue)
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Steps:**
- If manifest loader reports pixel skip reason, skip pixel metrics + attach reason in report.
- Commit.

### Task 28: Add dataset_summary for benchmark runs too

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_report_dataset_summary_benchmark.py`

**Steps:**
- Add `dataset_summary` to report payload (counts + anomaly ratio).
- Commit.

### Task 29: Ensure manifest benchmark per-image JSONL contains stable `input` and `meta` (if available)

**Files:**
- Modify: `pyimgano/pipelines/run_benchmark.py`
- Test: `tests/test_benchmark_manifest_meta_jsonl.py`

**Steps:**
- When dataset is manifest, attach `meta` to per-image rows (using manifest loader output).
- Commit.

### Task 30: Document benchmark manifest usage

**Files:**
- Modify: `docs/CLI_REFERENCE.md`

**Steps:**
- Add a snippet for `pyimgano-benchmark --dataset manifest --manifest-path ...`.
- Commit.

### Task 31: Ensure dataset catalog API supports explicit manifest_path everywhere

**Files:**
- Modify: `pyimgano/datasets/catalog.py`
- Test: `tests/test_dataset_catalog.py`

**Steps:**
- Ensure `list_dataset_categories(dataset="manifest", root="...", manifest_path="...")` works.
- Commit.

### Task 32: Add helpful errors for common manifest benchmark mistakes

**Files:**
- Modify: `pyimgano/cli.py`
- Test: `tests/test_cli_smoke_manifest_benchmark.py`

**Steps:**
- Missing manifest path: show actionable hint.
- Unsupported input_mode for manifest in benchmark: explicit error.
- Commit.

### Task 33: Add an example for benchmark manifest usage

**Files:**
- Create: `examples/manifest_benchmark_demo.sh` (optional)

**Steps:**
- Add a simple runnable example script (no downloads, just CLI shape).
- Commit.

### Task 34: Add regression tests for manifest list-categories vs benchmark list-categories

**Files:**
- Modify: `tests/test_cli_list_categories_manifest.py`

**Steps:**
- Ensure both discovery paths behave the same for manifest categories.
- Commit.

---

## Phase 4 (Tasks 35–40): Polish + release hygiene

### Task 35: Run formatting (best-effort)

**Steps:**
- Run: `python -m black pyimgano tests`
- Run: `python -m isort pyimgano tests`
- Commit formatting-only changes if any.

### Task 36: Run targeted tests (best-effort)

**Steps:**
- Run: `pytest -q tests/test_workbench_preflight_manifest.py tests/test_infer_cli_infer_config.py`

### Task 37: Run full test suite (best-effort)

**Steps:**
- Run: `pytest -q`

### Task 38: Update changelog

**Files:**
- Modify: `CHANGELOG.md`

**Steps:**
- Add an Unreleased entry summarizing preflight + infer-config + benchmark manifest support.
- Commit.

### Task 39: Optional patch version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `pyimgano/__init__.py`

**Steps:**
- Bump patch (e.g., `0.6.5`) if features are user-facing.
- Commit: `release: v0.6.5`.

### Task 40: Push

**Steps:**
- `git push origin main`
- Optional: `git tag v0.6.5 && git push origin v0.6.5`

