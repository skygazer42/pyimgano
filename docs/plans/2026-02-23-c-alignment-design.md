# PyImgAno “C-mode” Alignment (PyOD + sklearn + torch-like) — Design

**Date:** 2026-02-23  
**Status:** Approved  
**Owner:** @luke / @codex  

## Context

`pyimgano` is a vertical toolkit for **visual anomaly detection**. It already provides:

- A large model registry (`pyimgano.models.create_model`)
- Industrial datasets + benchmark pipeline + run artifacts
- Numpy-first inference helpers and post-processing

The next step is to make the package feel “as dependable” as industrial users expect from:

- **PyOD / scikit-learn** (uniform estimator API + predictable behavior)
- **PyTorch/TensorFlow** ecosystems (repeatable runs, tooling, strong CLI surfaces)

The user goal is “C-mode”: support both worlds:

1) A stable, sklearn/PyOD-like *estimator contract* (fit/score/predict, predictable shapes, errors).  
2) A practical industrial pipeline layer (CLI runners, artifacts, category=all, custom datasets).

## Goals (Must-have)

1. **Contract-first API**:
   - detectors expose consistent `fit`, `decision_function`, `predict` behavior
   - explicit input mode support: `paths`, `numpy` (and optionally `torch`)
2. **Discoverability**:
   - `pyimgano-benchmark --model-info` describes constructor kwargs + capabilities
   - `--list-models` filtering remains stable
3. **CLI consistency**:
   - shared parsing/validation across `pyimgano-benchmark`, `pyimgano-infer`, `pyimgano-robust-benchmark`
   - shared JSONL/report writing helpers (no drift)
4. **Reproducibility**:
   - run artifacts include schema version + environment + config
   - deterministic seeds where feasible
5. **Minimal breaking changes**:
   - keep `pyimgano.models` registry API stable
   - keep legacy `pyimgano.detectors` compatibility module working

## Non-goals

- Rewriting every deep model implementation.
- Shipping model weights inside the wheel (weights stay external).
- Achieving full scikit-learn pipeline compatibility for every detector (we will provide an adapter).

## Architecture (3 Layers)

### Layer 1 — Contracts (Hard)

Define and test a minimal detector contract:

- `fit(X, y=None) -> self`
- `decision_function(X) -> np.ndarray shape (N,)`
- `predict(X) -> np.ndarray shape (N,)` with values in `{0,1}`

Optional capabilities:
- pixel maps: `get_anomaly_map(path)` / `predict_anomaly_map(batch)`
- serialization: `save()/load()` or pickle support (model-dependent)

This layer is enforced via contract tests.

### Layer 2 — Registry + Adapters (Stable API)

The registry remains the entry point. We add **introspection**:

- constructor signature + accepted kwargs
- metadata (`requires_checkpoint`, backend tags)
- computed capabilities (input modes, pixel-map support, save/load support)

Additionally, add a minimal sklearn-style adapter for registry models.

### Layer 3 — Pipelines + Reporting (Industrial UX)

Unify run artifacts + CLI UX across:

- benchmark
- infer
- robustness benchmark

Artifacts:
- `report.json`
- `config.json`
- `environment.json` (versions, platform)
- per-image JSONL (`per_image.jsonl`)

## Delivery / Release Strategy

- Execute **40 tasks** with **one commit per task**.
- After commits **#10 / #20 / #30 / #40**:
  - bump version
  - update changelog
  - create and push a tag (`v0.5.5`, `v0.5.6`, `v0.5.7`, `v0.5.8`)
- Work happens on a feature branch in a git worktree; we merge to `main` once complete.

## Acceptance Criteria

1. All tests pass in CI (Linux/macOS/Windows, Python 3.9–3.12).
2. `pyimgano-benchmark` / `pyimgano-infer` / `pyimgano-robust-benchmark` share parsing logic (no drift).
3. Reports are stable, versioned, and include environment metadata.
4. At least a handful of representative detectors pass the contract test suite.

