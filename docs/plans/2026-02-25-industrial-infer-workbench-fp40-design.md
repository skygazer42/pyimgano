# PyImgAno Industrial Inference + Workbench “FP40” Hardening — Design

**Date:** 2026-02-25  
**Status:** Approved (user: “continue”; strict backward compatibility; main-branch delivery)  

## Context

`pyimgano` already includes a large model registry (classical + deep, including
pixel-map models) and an industrial CLI surface:

- `pyimgano-train` (workbench recipes + exportable `infer_config.json`)
- `pyimgano-infer` (offline inference → JSONL, with optional anomaly maps)
- “defects export” support (mask + connected-component regions) with ROI-aware
  pixel-threshold calibration

For production industrial inspection, the biggest remaining pain points are
rarely “missing another model”, and more often:

1) **False positives and unstable defect masks/regions** (noise, edges, tiles,
   background / fixture areas, fragment regions, inconsistent ordering).
2) **Throughput & operational UX** (batching, I/O, tiling seams, profiling).
3) **Reproducibility / audit / delivery artifacts** (infer-config portability,
   provenance, validation, environment capture).

This design proposes a 40-task hardening plan focused on those three outcomes.

## Goals (priority order)

1) **Reduce false positives** and make `--defects` outputs more stable and
   controllable for industrial pipelines.
2) Improve **throughput** without changing results by default.
3) Improve **reproducibility & auditability** of workbench artifacts and
   `infer_config.json` delivery.

## Constraints (must-haves)

- **Strict backward compatibility**
  - Do not change existing JSONL semantics or CLI meanings.
  - Only add new fields, flags, and config keys.
  - Old configs must continue to load and behave as before.
- **Release cadence**
  - Execute **40 tasks** in **4 milestones**.
  - After tasks **#10 / #20 / #30 / #40**:
    - bump version
    - update changelog
    - tag `vX.Y.Z`
    - push to `main`
- **No service layer**
  - Offline CLIs only; no FastAPI/HTTP service in this plan.
- **Keep dependencies light**
  - Prefer stdlib + existing deps; new deps must be optional and justified.

## Non-goals

- Time-series / video tracking (speaker diarization analog is out-of-scope here).
- Shipping weights inside wheels (weights stay external on disk/cache).
- Large registry expansion as a primary goal (only fill gaps if necessary).

## Options considered

### Option A — Defects & infer-config hardening (recommended)

Invest in the post-processing and delivery layer: ROI/threshold/provenance,
mask/region stability, and deployable configs.

**Why:** directly targets industrial error modes (FPs, unstable outputs) with
minimal risk and strong testability.

### Option B — Add more SOTA models/recipes

Add new algorithm wrappers and training recipes.

**Why not as primary:** the repo already includes many SOTA entries; accuracy
gains are often dominated by postprocess/ROI/threshold for industrial FP issues.

### Option C — Full “PyOD/sklearn/torch-like” API unification

Deep refactor for uniform estimator contracts everywhere.

**Why not as primary:** valuable, but riskier and less directly connected to FP
stability. We will borrow only the most practical pieces (contract tests, clear
errors) in Milestone 4.

## Proposed Design

### A) Defects pipeline: stable, controllable industrial outputs

Additive configuration (CLI flags + `infer_config.json` keys) to improve defect
mask + region extraction without changing default behavior:

- **Edge/border suppression** (`border_ignore_px`) to reduce seam/edge artifacts.
- **Map smoothing** (`map_smoothing`) to suppress pixel noise pre-threshold.
- **Hysteresis thresholding** (`hysteresis`) to reduce salt-and-pepper FPs while
  keeping real defects connected.
- **Shape filters** (`shape_filters`) to remove “long thin” border strips and
  implausible fragments (aspect ratio, solidity, etc.).
- **Region merge** (`merge_nearby`) to combine close fragments into one defect.
- **Deterministic region ordering + ids** to make outputs stable across runs.
- **Optional overlays export** for quick FP root-cause inspection.

All features are **off by default** unless enabled via config/flags.

### B) Performance: throughput without result drift

Add optional (opt-in) inference performance knobs:

- streaming JSONL writing + bounded memory usage
- batch-size support when detector can batch
- tiling grid reuse and seam blending options
- optional AMP/autocast for supported torch models
- configurable worker parallelism for load/preprocess
- per-stage profiling output (load/preprocess/model/postprocess/save)

### C) Workbench & delivery: auditability and artifact contracts

Make exported `infer_config.json` more self-sufficient and easier to validate:

- add `schema_version` (additive) and compatibility parsing
- validate infer-config completeness / checkpoint references
- add a dedicated “validate infer-config” CLI
- expand provenance (`threshold_provenance`, `pixel_threshold_provenance`)
- expand environment capture (GPU/CUDA/torch versions when available)
- optional “deployment bundle” export (config + checkpoint + metadata)
- unify workbench → infer behavior so deploys match training settings

## Milestones (40 tasks / 4 releases)

### Milestone 1 (Tasks 1–10): Defects false-positive controls

1. `defects.border_ignore_px` (edge suppression)
2. `defects.map_smoothing` (median/gaussian/box; opt-in)
3. `defects.hysteresis` (high/low thresholds)
4. `defects.region_filters.shape` (aspect/solidity/compactness)
5. `defects.merge_nearby` (distance-based merge)
6. Deterministic region ordering + stable ids
7. Stable `max_regions` topK selection strategy
8. Optional image-space bbox mapping when possible
9. `--save-overlays` (opt-in debug artifacts)
10. Docs + example config for FP reduction preset

Release: version bump + tag + push.

### Milestone 2 (Tasks 11–20): Throughput & profiling

11. Stream JSONL writer hardening
12. Add `--batch-size` (safe fallback to 1)
13. Tiling buffer reuse
14. Tiling seam blending option
15. Add `--amp` (opt-in)
16. Add `--num-workers` for loading/preprocess
17. Preprocess pipeline reuse where safe
18. Stage timing profile in reports/logs (additive)
19. Optional mask encoding/compression improvements
20. Benchmark notes + docs for throughput tuning

Release: version bump + tag + push.

### Milestone 3 (Tasks 21–30): Infer-config delivery & audit

21. Add `infer_config.schema_version`
22. Infer-config validation on export/load
23. Add `pyimgano-validate-infer-config`
24. Enrich threshold provenance fields
25. Enrich environment report (GPU/CUDA/torch)
26. Add `--seed` for reproducibility (opt-in)
27. Export optional deployment bundle directory
28. Preflight/report alignment (additive)
29. Ensure workbench-exported infer settings replay in infer
30. Docs: “0 → run → infer_config → deployment”

Release: version bump + tag + push.

### Milestone 4 (Tasks 31–40): Ecosystem alignment + recipes + docs

31. Strengthen sklearn/PyOD adapter behaviors (errors, input modes)
32. Add contract tests for representative detectors
33. Clarify dataset API “recommended paths”
34. Add 2–3 industrial recipes (using existing models)
35. Add “false-positive debugging guide” (overlays + region stats)
36. Add opt-in illumination/contrast preprocessing pipeline knobs
37. Document “torch-like” package structure (how to extend models/datasets)
38. Update comparison docs (positioning vs PyOD/anomalib)
39. Update README (+ translations) for current industrial workflow
40. Packaging/publishing check (no VCS direct deps; update publishing docs)

Release: version bump + tag + push.

## Testing Strategy

- Unit tests for:
  - each new defects postprocess feature (deterministic synthetic maps)
  - deterministic region ordering/id behavior
  - tiling seam blending correctness (basic invariants)
  - infer-config schema and validation behaviors (backwards compatible)
- CLI tests:
  - `pyimgano-infer --defects` JSONL contains new optional fields only when enabled
  - old configs/flags unchanged behavior

## Acceptance Criteria

1) Default behavior unchanged unless new flags/config keys are set.  
2) New defects controls can be enabled and are covered by deterministic tests.  
3) `infer_config.json` becomes easier to validate and more portable for deployment.  
4) Each milestone release is versioned, tagged, and pushed to `main`.  

