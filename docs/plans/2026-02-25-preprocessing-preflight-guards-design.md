# Preprocessing Preflight Guards (Design)

**Date:** 2026-02-25

## Goal

Make the new industrial preprocessing block (`preprocessing.illumination_contrast`) safer to use in real projects by:

1) **Failing early** (preflight) when a configured model cannot accept numpy image inputs.
2) Providing a **clear runtime error** in `pyimgano-infer` instead of a cryptic stack trace.
3) Adding an example workbench config showing a recommended preprocessing setup.

This is a “hardening” pass after `v0.6.22`, which made preprocessing deploy-consistent across:
workbench → exported `infer_config.json` → `pyimgano-infer`.

## Context / Problem

`preprocessing.illumination_contrast` is implemented via a wrapper that loads images as `RGB/u8/HWC` numpy arrays and forwards them to the detector (`PreprocessingDetector`).

This works only for **numpy-capable** detectors. Many classical/path-oriented detectors expect `list[str]` and do their own image loading / feature extraction. With preprocessing enabled, those models receive numpy arrays and may crash in confusing ways.

## Non-goals

- Automatically rewriting user configs (e.g. switching `dataset.input_mode`).
- Making all models support numpy images.
- Adding CLI flags for preprocessing (infer-config / workbench is the source of truth).

## Approaches Considered

### A) Runtime “try and fallback”

Let preprocessing run; if the detector fails with numpy, retry without preprocessing.

Pros:
- “Works” in more cases without user changes.

Cons:
- Silent behavior differences (dangerous in industrial QA).
- Hard to make deterministic; can mask real errors.

### B) Capability-based hard guard (recommended)

Use model registry capabilities to decide whether preprocessing is allowed:
- If preprocessing configured AND model does not support numpy inputs → error in preflight + clear runtime error.

Pros:
- Deterministic, explicit, auditable.
- Catches misconfigurations before long runs.

Cons:
- Best-effort: capability detection relies on registry tags / base classes.

### C) Hybrid (guard + warning)

Like (B), but use:
- **error** when the model is known and lacks numpy support
- **warning** when the model is unknown/unregistered (cannot prove support)

This is effectively the same as (B) but avoids blocking custom/third-party models.

## Proposed Design

### 1) Workbench preflight guard

Add a new preflight check that runs before dataset validation:

- If `config.preprocessing.illumination_contrast` is set:
  - Best-effort import `pyimgano.models` (register models).
  - Query registry entry for `config.model.name`.
  - Compute capabilities (`input_modes`).
  - If `numpy` is not supported → add a **preflight error** with context:
    - `model`
    - `supported_input_modes`
    - `hint` to use a numpy-capable model (e.g. `vision_patchcore`, `vision_padim`, etc.)
  - If registry lookup fails (custom model) → add **warning** (cannot validate).

### 2) `pyimgano-infer` runtime clarity

When loading `--infer-config` / `--from-run` that contains preprocessing:

- Wrap the detector as today.
- If the first inference call fails with a “wrong input type” signature, surface a clearer message:
  - preprocessing requires a numpy-capable model
  - disable preprocessing or choose a model with `numpy` tag.

Implementation will be lightweight: a guard at wrap time that checks registry capabilities when possible.

### 3) Example config + docs

Add `examples/configs/industrial_adapt_preprocessing_illumination.json` demonstrating:

- `vision_patchcore` (numpy-capable)
- a minimal illumination/contrast block (white balance + CLAHE + gamma)

Update `docs/WORKBENCH.md` quickstart list to reference the new example.

## Testing Plan

- Unit test for preflight:
  - Register a dummy model with tags that do not include `numpy`.
  - Create a config with preprocessing enabled.
  - Assert preflight issues include the new error code.

No integration tests should be needed beyond that (runtime error paths are covered by existing `infer_cli` smoke tests).

