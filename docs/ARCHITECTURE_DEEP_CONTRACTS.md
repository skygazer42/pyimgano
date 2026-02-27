# Deep Model Contracts

This document describes the **deep/vision detector contract** used in `pyimgano`.

The goal is to make deep models behave like sklearn-style detectors while still
supporting industrial vision needs:

- image-level scoring (`decision_function`)
- optional pixel-level outputs (`get_anomaly_map`)
- deterministic tiny/smoke modes for tests
- no implicit network downloads (airgapped-friendly)

## What “Deep” Means Here

In this repo, a “deep model” usually means a detector that:

- uses Torch/Torchvision during inference/training
- consumes image inputs (paths or in-memory arrays)
- learns a representation (reconstruction/distillation/flow/…)

Deep detectors are expected to remain compatible with the base detector
semantics (`BaseDetector`):

- higher score = more anomalous
- contamination-based thresholding is available via base classes
- `predict()` returns `{0,1}` labels

## Base Classes

Key base classes (conceptual):

- `BaseDetector`:
  - contamination thresholding
  - common sklearn-like convenience methods
- `BaseVisionDetector`:
  - classical vision detectors that depend on a feature extractor
- `BaseVisionDeepDetector`:
  - deep vision detectors that use Torch training/eval loops

The deep base class lives in `pyimgano/models/baseCv.py`.

## Required Methods (Deep Contract)

Deep detectors built on `BaseVisionDeepDetector` are expected to implement:

1. `build_model(self) -> torch.nn.Module`
   - constructs the Torch module (no downloads by default)

2. `training_forward(self, batch) -> float`
   - one optimization step; returns scalar loss

3. `evaluating_forward(self, batch) -> np.ndarray | torch.Tensor`
   - computes a per-sample anomaly score for a batch
   - output must be convertible to a 1D numpy array

The base class provides:

- `fit(X)` which prepares a dataset + dataloader and runs the training loop
- `decision_function(X)` which runs eval and returns a score per sample

## Inputs

Deep detectors should accept:

- `X` as a list of file paths (`str`/`Path`)
- `X` as a list of `np.ndarray` images (uint8 HWC)

This is important for industrial pipelines where images may already be decoded.

## Outputs

Image-level scoring:

- `decision_function(X)` returns `np.ndarray` with shape `(N,)`
- scores must follow `higher = more anomalous`

Optional pixel outputs:

- models that can produce localization maps should expose:
  - `get_anomaly_map(x)` returning a single 2D map `(H,W)` (float32)

## Tiny Mode (Tests)

Some deep detectors provide `tiny=True`:

- smaller networks / reduced dimensions
- intended for unit tests and quick smoke runs
- should be deterministic given the same seed

## No Implicit Downloads

Industrial deployments are frequently offline or airgapped.

Rules:

- default constructors must not trigger network downloads
- torchvision-based extractors/backbones must use `weights=None` unless the user opts in

We enforce this with tests that monkeypatch `torch.hub.load_state_dict_from_url`.

## Caching (Best-Effort)

For repeated scoring runs on the same image paths, `BaseVisionDeepDetector`
supports an optional **eval tensor cache**:

- `det.set_eval_cache(cache_dir)`
- cache keys incorporate file metadata + a transform fingerprint
- intended for deterministic eval transforms (not random train augmentation)

See: `pyimgano/cache/deep_embeddings.py`.

