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

