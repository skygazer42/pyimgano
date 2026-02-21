# Anomalib Checkpoint Benchmark Coverage (Design)

**Date:** 2026-02-21

## Context

`pyimgano` provides a unified API, dataset loaders, benchmarking, pixel-level evaluation, and reporting for
visual anomaly detection. In practice, many teams train industrial anomaly detectors using **anomalib** and
then need a reliable way to:

- run inference on a trained checkpoint,
- compute image-level and pixel-level metrics on MVTec AD / VisA-like datasets,
- postprocess/align anomaly maps,
- produce stable JSON reports.

This design adds first-class support for benchmarking anomalib checkpoints via `pyimgano-benchmark`,
while keeping anomalib as an **optional dependency**.

## Goals

1. **CLI-first anomalib checkpoint benchmarking**
   - Support `pyimgano-benchmark --model vision_*_anomalib --checkpoint-path ...`.
   - Support `--model-kwargs` JSON for advanced constructor parameters.

2. **Robust, consistent anomaly-map handling**
   - Normalize anomalib outputs to a 2D `float32` anomaly map for pixel metrics.
   - Keep existing pipeline alignment behavior (maps resized to GT mask size).

3. **Safe parameter injection**
   - Avoid passing unsupported `device` / `pretrained` kwargs to models that do not accept them.
   - Do not override user-provided values from `--model-kwargs`.

4. **Keep core install lightweight**
   - `import pyimgano.models` must remain safe when anomalib is not installed.
   - anomalib usage remains behind `pyimgano[anomalib]`.

## Non-Goals

- Re-implement anomalib training loops inside `pyimgano`.
- Guarantee exact metric parity with anomalib’s own benchmark runner (different defaults may apply).
- Add new datasets beyond what `pyimgano` already supports (this design focuses on checkpoint evaluation).

## Proposed UX

### Minimal CLI (recommended)

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore_anomalib \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --device cuda \
  --pixel \
  --output runs/mvtec_bottle_patchcore_anomalib.json
```

### Advanced CLI (kwargs)

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_anomalib_checkpoint \
  --checkpoint-path /path/to/checkpoint.ckpt \
  --model-kwargs '{"contamination": 0.05}' \
  --pixel
```

### Precedence and validation rules

- `--model-kwargs` must parse as a JSON object (`{...}`).
- `--checkpoint-path` sets `checkpoint_path` unless the user already provided it in `--model-kwargs`.
- If both are provided and conflict, error out with a clear message.
- For anomalib-backed models, `checkpoint_path` is required (either via `--checkpoint-path` or `--model-kwargs`).

## Architecture

### 1) Registry metadata

Update `pyimgano/models/anomalib_backend.py` registry entries to include:

- `metadata.backend = "anomalib"`
- `metadata.requires_checkpoint = True`
- `metadata.anomalib_model = "<patchcore|padim|...>"` for alias entries

This enables the CLI to detect “checkpoint-required” models without hard-coding name patterns.

### 2) CLI safe kwargs injection

`pyimgano/cli.py` will:

1. Parse `--model-kwargs` JSON into a dict.
2. Merge `--checkpoint-path` into `checkpoint_path` with conflict checking.
3. Add default kwargs (e.g. `device`, `contamination`, `pretrained`) only if they are not already set.
4. Validate kwargs against the selected model’s constructor signature:
   - If the constructor does **not** accept `**kwargs`, unknown keys become a user-facing error.

### 3) Anomaly map normalization for anomalib inferencer outputs

`VisionAnomalibCheckpoint.get_anomaly_map` will normalize the returned anomaly map to:

- 2D array (squeezed from `(1, H, W)` / `(H, W, 1)` etc.)
- `np.float32`

If the result cannot be interpreted as a 2D map, it will raise a clear error.

## Testing Strategy

Keep tests lightweight (no anomalib dependency required) by dependency injection:

- Wrapper tests inject a fake inferencer that returns:
  - dict-like predictions,
  - object-like predictions,
  - anomaly maps with various shapes.
- CLI tests validate:
  - `--model-kwargs` parsing,
  - checkpoint precedence/validation,
  - safe filtering of injected defaults.

## Documentation

- Update README and/or Quickstart with a “Train with anomalib, evaluate with pyimgano” snippet.
- Document `--checkpoint-path` and `--model-kwargs` in `pyimgano-benchmark --help`.

## Future Work (out of scope)

- Support `anomalib.engine.Engine.predict()` in addition to `TorchInferencer` for newer anomalib versions.
- Add dataset adapters for anomalib-native datasets (e.g. MVTec LOCO, RealIAD) and benchmark parity mode.

