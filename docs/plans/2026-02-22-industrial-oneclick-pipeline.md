# Industrial One-Click Pipeline (CLI + Python API) — Implementation Plan (20 Tasks)

**Date:** 2026-02-22  
**Owner:** @codex  
**Depends on:** `docs/plans/2026-02-22-industrial-oneclick-pipeline-design.md`

## Scope (B → A → C)

This plan executes:

- **B (primary):** One-click industrial pipeline with run artifacts + category=all + custom dataset support.
- **A (secondary):** Provide a stable Python API entry point that acts as a “train+val runner” for industrial workflows.
- **C (secondary):** Make the package surface more torch/pyod-like by exposing dataset loaders under `pyimgano.datasets` and pipeline helpers under `pyimgano.pipelines`.

## Task List (20)

### Phase 1 — Core runner + reporting

1. Add a small reporting helper to write JSONL (`save_jsonl_records`) alongside `save_run_report`.
2. Add a run directory helper (`build_run_dir_name`, `ensure_run_dir`) with timestamp formatting and safe name sanitization.
3. Add a dataset-category discovery helper:
   - MVTec / LOCO / BTAD via class constants
   - AD2 / VisA via `list_categories(root)`
   - custom dataset returns `["custom"]`
4. Add a pipeline runner module `pyimgano/pipelines/run_benchmark.py`:
   - `run_benchmark_category(...)`
   - returns `{split, threshold, metrics, per_image_records}`
5. Implement train-calibrated score thresholds in the pipeline:
   - default quantile = `1 - contamination` when available else `0.995`
   - optional override `calibration_quantile`
6. Implement `run_benchmark_all(...)`:
   - iterates categories
   - aggregates per-category reports
   - computes overall mean/std for main metrics

### Phase 2 — CLI integration

7. Extend `pyimgano-benchmark` parser:
   - add `custom` dataset
   - add `--resize H W`
   - add `--output-dir`
   - add `--save-run/--no-save-run`
   - add `--per-image-jsonl/--no-per-image-jsonl`
   - add `--calibration-quantile`
   - add `--limit-train`, `--limit-test`
8. Update CLI validation rules:
   - require category unless dataset=custom
   - allow `--category all` for standard datasets
9. Wire CLI execution path:
   - call pipeline runner
   - print JSON summary to stdout (backward compatible)
   - write artifacts to disk if save-run enabled
10. Keep `--output` behavior:
    - if provided, write a copy of the top-level report to that exact path

### Phase 3 — Package surface (C)

11. Add `pyimgano/datasets/benchmarks.py` to expose benchmark dataset loaders (`MVTecDataset`, `VisADataset`, `load_dataset`, etc.) without moving existing implementation.
12. Re-export benchmark dataset loaders from `pyimgano/datasets/__init__.py` (torch-like surface).
13. Update `pyimgano/pipelines/__init__.py` to export new runner functions.
14. Update `pyimgano/pipelines/mvtec_visa.py` to import dataset factory from `pyimgano.datasets` (not `pyimgano.utils`), preserving compatibility.

### Phase 4 — Docs + hygiene

15. Update `docs/EVALUATION_AND_BENCHMARK.md` with one-click examples:
    - single category
    - category=all
    - custom dataset
16. Update `README.md` with a short “One-click benchmark” section + output layout.
17. Add `runs/` to `.gitignore` since runs are created by default.

### Phase 5 — Tests + release

18. Add unit tests for:
    - category listing helper
    - JSONL saving
19. Add CLI tests for:
    - `--category all` produces summary JSON and writes run artifacts to a temp output dir
    - `--dataset custom` works with minimal structure
20. Bump version + changelog entry, then commit, tag, push to `main`.

## Deliverables

- `pyimgano.pipelines.run_benchmark(...)` Python API
- `pyimgano-benchmark` upgraded to support `--category all` + run artifacts
- `report.json` + per-category `per_image.jsonl` output layout
- Docs and tests updated accordingly

