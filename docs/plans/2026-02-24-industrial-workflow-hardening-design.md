# PyImgAno Industrial Workflow Hardening — Design

**Date:** 2026-02-24  
**Status:** Approved (user: workbench-first, no HTTP service)  

## Context

`pyimgano` is already usable for industrial anomaly detection, but “industrial
usable” is more than having many models:

- Teams need **reproducible**, **diagnosable**, and **portable** workflows.
- Data rarely fits benchmark folder layouts; manifest JSONL helps, but real
  projects also need **preflight validation**, **dataset summaries**, and
  **explicit artifact contracts**.
- Once a run is trained/adapted, downstream inference should be a **delivery
  artifact** (config + checkpoint) rather than “copy a run directory and hope
  the structure stays consistent”.

The project currently has three main user-facing surfaces:

1) `pyimgano-benchmark` — ad-hoc benchmarking with run artifacts  
2) `pyimgano-train` — workbench recipe runs (artifact-first, reproducible)  
3) `pyimgano-infer` — offline inference (including `--from-run`)  

This hardening effort chooses a **workbench-first** philosophy:

- Workbench artifacts are the canonical contract for industrial work.
- Benchmark CLI is a convenience surface that should not diverge from workbench
  semantics.

## Goals

### 1) Industrial-grade data validation (preflight)

- Detect common data issues **before** running a recipe:
  - missing files
  - duplicated samples (same path repeated)
  - invalid/empty categories
  - invalid label/split combinations
  - group leakage risks / conflicting explicit splits within a group
  - mask coverage problems for pixel metrics (and quantify coverage)
- Provide a machine-readable summary (JSON) for CI and pipeline orchestration.

### 2) Consistent inference delivery contract (no run-dir coupling)

- Support running inference from a single exported file:
  - `artifacts/infer_config.json` (already exportable)
  - plus an on-disk checkpoint file (can live inside or outside the run dir)
- Add `pyimgano-infer --infer-config PATH` to run inference from exported config.

### 3) Benchmark/workbench alignment

- Keep dataset handling and category discovery consistent:
  - manifest dataset category listing should work everywhere
  - manifest dataset should be runnable in benchmark mode when feasible, or
    explicitly rejected with a “use workbench” message.

### 4) Stronger artifacts & reporting

- Reports should include:
  - dataset split summary (counts; anomaly ratio)
  - pixel-metric status with explicit reason when disabled
  - threshold provenance (where the threshold came from)
  - manifest metadata propagation into per-image JSONL when present
- Keep schema backwards compatible (additive fields; avoid breaking changes).

### 5) Documentation and templates

- “From 0 → industrial run → exported inference” documented as a shortest path.
- Provide config templates for common industrial workflows (manifest-first).

## Non-goals

- No HTTP service layer (FastAPI/etc.) in this iteration.
- No large “model zoo expansion” as a primary goal (quality and contracts first).
- No UI/frontend.
- No breaking changes to existing run artifact paths unless unavoidable.

## Options Considered

### Option A: Workbench-first hardening (recommended)

Make the workbench runner + artifacts the canonical contract. Add a preflight
module and build a deployable inference config path.

**Why:** industrial teams want stable artifacts and predictable behavior more
than extra knobs.

### Option B: CLI-first unification

Build a new umbrella CLI with subcommands and unify all behavior at the CLI
layer.

**Why not:** larger surface change and higher risk to existing users; it does
not inherently guarantee better artifacts.

### Option C: Schema-first strict spec

Freeze a JSON schema for reports and configs, then enforce it everywhere.

**Why not (for now):** useful later, but slows iteration and requires more
design investment than needed for this phase.

## Proposed Architecture

### 1) `pyimgano.workbench.preflight`

Add a preflight module that can run independently of training:

- Inputs:
  - `WorkbenchConfig`
  - dataset-specific sources (manifest path, root, category)
- Outputs:
  - `PreflightReport` (JSON-friendly)
  - `issues[]` with severity (`error`/`warning`/`info`) and stable codes

Preflight covers:

- Manifest dataset:
  - parse records (reuse `iter_manifest_records`)
  - per-category counts (train/val/test, label distribution)
  - missing files (image/mask)
  - duplicate image paths
  - group_id conflicts (split conflicts; anomaly-in-train/val)
  - pixel-metric viability:
    - whether masks exist
    - whether all anomaly test samples have masks
- Benchmark datasets (`mvtec`, `visa`, `btad`, `custom`):
  - root existence
  - category listing sanity
  - custom layout validation (already exists; reuse)

Integration:

- `pyimgano-train`:
  - keep `--dry-run` output stable
  - add a new `--preflight` mode (config required) that prints `{"preflight": ...}`
    and exits; errors → return code 2

### 2) Inference config as a first-class delivery artifact

Implement `pyimgano-infer --infer-config PATH`:

- Load exported infer config (`build_infer_config_payload` output).
- Support both:
  - `from_run` (optional convenience)
  - direct `checkpoint.path` resolution relative to the config file
- Multi-category infer configs:
  - require a category selection flag when multiple categories exist
  - use per-category threshold/checkpoint when present

### 3) Benchmark alignment

For `pyimgano-benchmark`:

- Add `--dataset manifest` runnable mode (paths-only) using
  `load_manifest_benchmark_split`.
- Reuse the same pixel gating rules (skip pixel metrics if anomaly masks are
  incomplete, with explicit reason).

This keeps benchmark behavior consistent with workbench semantics while still
allowing ad-hoc experimentation.

## Testing Strategy

- Unit tests:
  - preflight report contains expected counts and issues for synthetic manifests
  - infer-config loading supports relative checkpoint paths and category selection
  - benchmark manifest runs (dummy detector) produce stable artifacts
- Integration smokes:
  - manifest → `pyimgano-train` → `--export-infer-config` → `pyimgano-infer --infer-config`

## Acceptance Criteria

1) Users can validate data issues without starting training (`pyimgano-train --preflight`).
2) Exported inference configs can run without requiring `--from-run`.
3) Benchmark + workbench semantics for manifest splits and pixel metrics match.
4) Reports include explicit provenance for thresholds and pixel metric gating.
5) Documentation provides a shortest-path industrial recipe for manifest datasets.

