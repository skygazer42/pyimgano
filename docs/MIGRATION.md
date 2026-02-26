# Migration Guide

This guide summarizes the most important compatibility notes and recommended
paths when upgrading `pyimgano`.

> Tip: If you are new to the project, you can ignore this file and start with:
> - `README.md`
> - `docs/QUICKSTART.md`
> - `docs/CLI_REFERENCE.md`

---

## Stable API Surface (Recommended)

`pyimgano` is converging on a few “anchor APIs” that are designed to feel
familiar if you use scikit-learn / PyOD / torch-style libraries:

- **Models (registry-driven)**
  - `pyimgano.models.list_models()`
  - `pyimgano.models.create_model(name, **kwargs)`
  - `pyimgano.models.model_info(name)` (JSON-friendly metadata + capabilities)
- **Datasets**
  - `pyimgano.datasets.load_dataset(...)`
  - `pyimgano.datasets.MVTecDataset`, `pyimgano.datasets.VisADataset`, ...
- **Industrial runners / artifacts**
  - CLI: `pyimgano-benchmark`, `pyimgano-infer`, `pyimgano-robust-benchmark`
  - Pipeline: `pyimgano.pipelines.run_benchmark`

---

## PyOD Removal (Post v0.6.23)

Historically, `pyimgano` shipped several `vision_*` wrappers around the PyOD
library. Starting with the changes introduced after **v0.6.23 (2026-02-25)**,
`pyimgano` no longer depends on PyOD at runtime.

What changed:

- `pyod` was removed from package dependencies (`pyproject.toml`,
  `requirements.txt`).
- Ported/implemented a large set of classical detectors natively around
  `pyimgano.models.base_detector.BaseDetector` (stable thresholding +
  `predict()` / `predict_proba()` semantics).
- The default registry no longer auto-registers PyOD-only wrappers.

Removed registry model names (PyOD-only wrappers):

- `vision_cd`
- `vision_auto_encoder`
- `vision_anogan`
- `vision_dif`
- `vision_lunar`
- `vision_so_gaal`
- `vision_so_gaal_new`
- `vision_mo_gaal`
- `vision_xgbod`

Recommended replacements:

- Classical feature-vector baselines: `vision_ecod`, `vision_copod`,
  `vision_iforest`, `vision_knn`, `vision_pca`, `vision_kde`, `vision_gmm`,
  `vision_hbos`, `vision_mcd`, `vision_ocsvm`, `vision_kpca`, `vision_inne`.
- Deep feature-space detector: `vision_deep_svdd` (for tabular features or
  image-derived embeddings).
- Image-native reconstruction / pixel-map models: `ae_resnet_unet`, `vae_conv`,
  `vision_patchcore`, `vision_padim`, `vision_spade`, ...

If you previously passed PyOD-specific keyword arguments, expect small naming
differences on native implementations, for example:

- `epoch_num` -> `epochs`
- `hidden_neuron_list` -> `hidden_neurons`

## Legacy API (`pyimgano.detectors`)

Older versions of the project documented classes under `pyimgano.detectors`
(e.g. `IsolationForestDetector`, `AutoencoderDetector`).

- The module is still available as a **compatibility layer**.
- New code should prefer `pyimgano.models.create_model(...)` + registry names
  (e.g. `vision_patchcore`, `vision_ecod`, `vision_padim`).

---

## CLI + Run Artifacts (v0.5.4+)

### `pyimgano-benchmark` now writes run artifacts by default

By default, `pyimgano-benchmark` writes a run directory under `runs/`:

```
runs/<ts>_<dataset>_<model>/
  report.json
  config.json
  environment.json
  categories/<cat>/report.json
  categories/<cat>/per_image.jsonl
```

If you want “stdout-only” behavior:

- Use `--no-save-run` to disable artifact writing.
- Use `--output report.json` if you want a single JSON report file in addition
  to stdout.

See: `docs/EVALUATION_AND_BENCHMARK.md` and `docs/CLI_REFERENCE.md`.

### Input modes (`--input-mode`)

Some workflows require decoded frames in memory (e.g. corruption robustness),
while others are fine passing file paths to detectors.

- `pyimgano-benchmark --input-mode paths|numpy`
  - `paths`: pass file paths to detectors (default)
  - `numpy`: decode images first and pass `np.ndarray` frames (model dependent)
- `pyimgano-robust-benchmark --input-mode numpy|paths`
  - `numpy`: required for corruptions (default)
  - `paths`: clean-only evaluation (corruptions skipped)

---

## Model Discovery / Introspection

The CLI now exposes a stable discovery surface:

- `pyimgano-benchmark --list-models`
- `pyimgano-benchmark --list-models --tags vision,deep`
- `pyimgano-benchmark --model-info vision_patchcore`
- `pyimgano-benchmark --model-info vision_patchcore --json`

Programmatically:

- `pyimgano.models.list_models(tags=[...])`
- `pyimgano.models.model_info(name)` (JSON-friendly payload)

---

## Serialization (Classical Detectors Only)

`pyimgano` provides helpers for persisting **classical** detectors:

- `pyimgano.serialization.save_detector(detector, path)`
- `pyimgano.serialization.load_detector(path)`

Notes:

- This uses pickle under the hood; **never load pickle files from untrusted
  sources**.
- Deep learning detectors are intentionally not covered by this helper (their
  state is usually tied to torch modules + checkpoints).

---

## sklearn Integration

If you want `clone()`-friendly wrappers around registry models, use:

- `pyimgano.sklearn_adapter.RegistryModelEstimator`

See: `docs/SKLEARN_INTEGRATION.md`.
