# Examples Index

This index groups examples by the question you are trying to answer first, not by filename alone.

| File | Goal | Dependencies | Offline-safe | Notes |
|---|---|---|---|---|
| `quick_start.py` | Fastest Python API introduction | base install | yes | Smallest general-purpose entrypoint |
| `registry_quickstart.py` | Discover registry models and metadata | base install | yes | Good first stop after `pyim --list` |
| `industrial_infer_numpy.py` | Numpy-first industrial inference path | base install | yes | Best fit for backend/service integrations |
| `industrial_tiling_infer.py` | High-resolution tiled inference | base install | yes | Use when frames are too large for a single pass |
| `benchmark_example.py` | Programmatic benchmark invocation | base install | yes | Pairs well with starter benchmark configs |
| `feature_extractors_demo.py` | Inspect feature extractor choices | base install | yes | Useful before classical pipelines |
| `embedding_plus_core_ecod.py` | Embeddings plus a lightweight classical scorer | base install | yes | CPU-friendly baseline route |
| `embedding_plus_core_mahalanobis_shrinkage.py` | Embeddings plus stronger Gaussian-distance scoring | base install | yes | Good when you already have embeddings |
| `openclip_plus_core_knn.py` | CLIP-style embeddings plus classical kNN | `pyimgano[clip,torch]` | no | Foundation-model flavored route |
| `openclip_mvtec_visa.py` | OpenCLIP industrial evaluation flow | `pyimgano[clip,torch]` | no | Use when comparing multimodal backbones |
| `softpatch_mvtec_visa.py` | SoftPatch localization workflow | `pyimgano[torch]` | no | Good for noisy-normal datasets |
| `anomalydino_mvtec_visa.py` | AnomalyDINO few-shot workflow | `pyimgano[torch]` | no | Useful when normal data is scarce |
| `synthesis_generate_dataset_demo.py` | Generate synthetic anomaly data | base install | yes | Useful for smoke tests and demos |

Recommended starting order:

1. `quick_start.py`
2. `registry_quickstart.py`
3. `industrial_infer_numpy.py`
4. `benchmark_example.py`

If you prefer CLI-first onboarding, start from `docs/START_HERE.md` instead.

## Goal-Based Routes

If you want examples that match the new `pyim --goal ...` discovery routes, use this map:

- `pyim --goal first-run --json`
  Baseline: `embedding_plus_core_ecod.py`
  Optional backend upgrade: `openclip_plus_core_knn.py`
- `pyim --goal pixel-localization --json`
  Baseline: `embedding_plus_core_ecod.py`
  Stronger optional backend: `openclip_plus_core_knn.py`
- `pyim --goal deployable --json`
  Baseline: `industrial_infer_numpy.py`
  Optional backend route: `openclip_plus_core_knn.py`

The pattern is intentional:

- baseline: low-friction CPU or base-install route you can run first
- optional backend: stronger route that typically needs `pyimgano[clip,torch]` or another extra
