# CLI Presets — `industrial-balanced` (pyimgano-benchmark) — Design

**Date:** 2026-02-21  
**Status:** Approved  
**Owner:** @codex (with user approval)

## Background

`pyimgano-benchmark` exposes powerful model selection via `--model` + `--model-kwargs`, but
the UX for day-to-day industrial benchmarking is still too “expert-mode”:

- Users must remember good default hyperparameters per model.
- Passing JSON via `--model-kwargs` is cumbersome and error-prone.
- For “popular industrial AD” baselines, teams typically want a small set of known-good,
  speed/accuracy-balanced defaults to start from.

## Goals

1. Add a CLI-level `--preset` flag to apply **sensible defaults** for industrial visual
   anomaly detection.
2. Keep existing behavior unchanged unless the user explicitly supplies `--preset`.
3. Ensure deterministic and safe kwargs merging:
   - User-provided `--model-kwargs` must always override the preset.
   - CLI-provided “auto kwargs” (`--device`, `--contamination`, `--pretrained`) must still
     be applied when accepted by the target model constructor.
4. Improve out-of-the-box “balance” (accuracy vs speed) for the most commonly used
   industrial AD models.

## Non-Goals

- Automatically enabling `--pixel` or changing pixel post-processing defaults.
- Adding new detectors/algorithms.
- Enforcing a single “best” configuration for all datasets/hardware.
- Adding any new hard dependencies (everything stays optional).

## Proposed Approach

### A) Add `--preset` to `pyimgano-benchmark`

Add a new CLI argument:

- `--preset` (optional): name of a preset to apply before merging user kwargs.

If `--preset` is not set, behavior is unchanged.

### B) Implement one preset: `industrial-balanced`

Model-specific default kwargs (only applied when missing from `--model-kwargs`):

- `vision_patchcore`
  - `backbone="resnet50"` (lighter than WideResNet)
  - `coreset_sampling_ratio=0.05` (smaller memory bank, faster inference)
  - `n_neighbors=5` (stable kNN aggregation)
  - `knn_backend="faiss"` if available, else `"sklearn"` (optional speed-up)
- `vision_fastflow`
  - `epoch_num=10` (reduce training time vs default 20)
  - `n_flow_steps=6` (keep reasonable expressiveness)
  - `batch_size=32` (better GPU utilization)
- `vision_cflow`
  - `epochs=15` (reduce training time vs default 50)
  - `n_flows=4` (balanced speed/quality)
  - `batch_size=32`

Other models: no preset kwargs (empty dict).

### C) Kwargs Precedence and Filtering Rules

Merge order (highest precedence first):

1. `--model-kwargs` (user)
2. `--preset` kwargs (preset)
3. Auto kwargs inferred from CLI flags (`--device`, `--contamination`, `--pretrained`)

Filtering/validation:

- User kwargs are validated against the model constructor signature (existing behavior).
- Preset/auto kwargs are **best-effort** and filtered to only include keys accepted by the
  constructor (unless it accepts `**kwargs`).

### D) Optional Dependency Handling

For PatchCore’s `knn_backend` selection:

- Use `pyimgano.utils.optional_deps.optional_import("faiss")` to detect FAISS.
- Never raise if FAISS is missing; fall back to `"sklearn"`.

### E) Reporting

Keep output JSON stable; optionally include the preset name in the JSON payload for
reproducibility (if provided).

## Testing Strategy

1. Parser test: `--preset industrial-balanced` is accepted and does not `SystemExit`.
2. Preset application test: PatchCore receives preset kwargs when user does not override.
3. Precedence test: user kwargs override preset kwargs.
4. Optional-deps test: force FAISS available/unavailable via monkeypatch and verify chosen
   `knn_backend`.

## Documentation

Update:

- `README.md` (CLI usage example and explanation)
- `docs/QUICKSTART.md` (benchmark examples + recommended industrial presets)

