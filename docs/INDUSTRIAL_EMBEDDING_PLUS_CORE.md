# Industrial Route: Embedding + Core

This document describes a practical industrial pipeline that is intentionally
stable and maintainable:

1. extract **deep embeddings** (Torch/Torchvision or optional OpenCLIP)
2. run a **classical “core” detector** on the embedding matrix (`core_*`)
3. (optional) calibrate/standardize scores for thresholding / ensembles

Why this route works well:

- embeddings give strong generalization for many texture/appearance defects
- classical cores are easy to debug, ensemble, and calibrate
- no pixel model training required for a strong baseline

## Key Concepts

- **Feature extractor**: returns `np.ndarray` with shape `(N, D)`
  - examples: `torchvision_backbone`, `torchvision_backbone_gem`, `torchvision_multilayer`, `torchvision_vit_tokens`
  - industrial deployment: `torchscript_embed` (bring-your-own TorchScript checkpoint)
  - industrial deployment: `onnx_embed` (bring-your-own ONNX backbone; runs via onnxruntime)
  - optional: `openclip_embed`

- **Core detector**: consumes a feature matrix `X: np.ndarray (N, D)`
  - examples: `core_ecod`, `core_copod`, `core_knn`, `core_knn_cosine`, `core_knn_cosine_calibrated`, `core_iforest`
  - robust/statistical: `core_mahalanobis_shrinkage`, `core_cosine_mahalanobis`, `core_mcd`, `core_pca_md`
  - local/graph: `core_lof`, `core_lid`, `core_oddoneout`, `core_mst_outlier`
  - simple density: `core_extra_trees_density`
  - deep-on-embeddings: `core_torch_autoencoder`

- **Vision wrapper**: consumes images, internally extracts features, then calls the core
  - recommended: `vision_embedding_core`
  - generic: `vision_feature_pipeline`
  - convenience wrappers (examples): `vision_resnet18_ecod`, `vision_resnet18_knn_cosine_calibrated`, `vision_resnet18_mahalanobis_shrinkage`
  - deployment wrappers (examples): `vision_torchscript_ecod`, `vision_torchscript_knn_cosine_calibrated`
  - deployment wrappers (examples): `vision_onnx_ecod`, `vision_onnx_knn_cosine_calibrated`
  - no-torch structural wrappers (examples): `vision_structural_ecod`, `vision_structural_copod`

## Recommended Baselines

1. `vision_embedding_core` + `core_ecod`
   - fast and surprisingly strong on many industrial datasets
   - minimal hyperparameters

2. `vision_embedding_core` + `core_knn`
   - simple “distance to normals” baseline
   - can be sensitive to embedding scale; consider normalization/standardization

3. `vision_feature_pipeline` + `structural` + `core_iforest`
   - deterministic handcrafted baseline
   - useful when Torch is not available

4. Preconfigured "industrial wrappers" (minimal kwargs)
   - ResNet18 embeddings route (offline-safe, no implicit downloads)
     - `vision_resnet18_ecod` / `vision_resnet18_copod`
     - `vision_resnet18_knn_cosine_calibrated` (stable [0,1] score scale via unsupervised calibration)
     - `vision_resnet18_mahalanobis_shrinkage` / `vision_resnet18_cosine_mahalanobis` (Gaussian distance baselines)
     - `vision_resnet18_oddoneout` / `vision_resnet18_lof` (local anomaly baselines)
     - `vision_resnet18_pca_md` / `vision_resnet18_extra_trees_density` (sanity-check baselines)
     - `vision_resnet18_mcd` (robust covariance; useful when you suspect heavy tails / outliers)
     - `vision_resnet18_mst_outlier` (graph baseline)
   - TorchScript embeddings route (deployment-friendly; requires `checkpoint_path`)
     - `vision_torchscript_ecod` / `vision_torchscript_copod`
     - `vision_torchscript_knn_cosine_calibrated` / `vision_torchscript_cosine_mahalanobis`
     - `vision_torchscript_oddoneout` / `vision_torchscript_lof`
   - Structural CPU route (no torch; useful as an offline/CPU fallback)
     - `vision_structural_ecod` / `vision_structural_copod` / `vision_structural_iforest`
     - `vision_structural_knn` / `vision_structural_lof`
     - `vision_structural_pca_md` / `vision_structural_mcd` / `vision_structural_extra_trees_density`
   - Deep-on-embeddings (small deep capacity without a heavy pixel model)
     - `vision_resnet18_torch_ae` (reconstruction error on embeddings)

## Quick Menu (Industrial Baselines)

This is a practical “start here” menu for common industrial scenarios.

| Goal | Start with | Why / notes |
|------|------------|-------------|
| **Fast, parameter-free** | `vision_resnet18_ecod` / `vision_resnet18_copod` | Great first-pass baselines; minimal knobs |
| **Stable score scale across lines/cameras** | `vision_resnet18_knn_cosine_calibrated` | Unsupervised score standardization makes thresholds more portable |
| **Gaussian distance baseline** | `vision_resnet18_mahalanobis_shrinkage` | Strong, stable covariance baseline for embeddings |
| **Directional embeddings (magnitude noisy)** | `vision_resnet18_cosine_mahalanobis` | L2-normalized Mahalanobis (cosine-style) |
| **Local/cluster anomalies** | `vision_resnet18_oddoneout` / `vision_resnet18_lof` | Neighborhood comparison / density baselines |
| **Robust covariance (outliers/heavy tails)** | `vision_resnet18_mcd` | Robust estimator; avoid extremely high-dimensional features |
| **Pure CPU fallback (no torch)** | `vision_structural_ecod` / `vision_structural_copod` | Deterministic, dependency-light baseline |

### TorchScript deployment note

All `vision_torchscript_*` wrappers require a checkpoint file:

- Install: `pip install "pyimgano[torch]"`
- CLI: pass `--checkpoint-path /path/to/model.pt`
- Python: pass `checkpoint_path="/path/to/model.pt"`

This keeps production inference reproducible and airgapped-friendly.

## CLI/Preset Shortcuts

Two convenient ways to keep command lines short:

1. **Use wrapper model names directly** (works in `pyimgano-infer` and `pyimgano-benchmark`).
2. **Use JSON-ready preset names** (works in `pyimgano-benchmark` via `--model <preset-name>`):
   - see `pyimgano/presets/industrial_classical.py` (e.g. `industrial-structural-ecod`, `industrial-embed-mahalanobis-shrinkage-rank`)
   - see `pyimgano/cli_presets.py` for CLI-only aliases (e.g. `industrial-embedding-core-balanced`)

## Safe Defaults (No Downloads)

For industrial reliability, our defaults are chosen to avoid implicit weight downloads.

In particular:

- `torchvision_*` extractors default to `pretrained=False`
- `openclip_embed` defaults to `pretrained=None`

If you want pretrained weights, you must opt in explicitly.

## TorchScript Deployment (Bring Your Own Checkpoint)

For production environments where you want a stable, reproducible embedding model
without relying on upstream model registries or implicit downloads, you can export
your embedding model to TorchScript and use:

- `embedding_extractor="torchscript_embed"`
- `embedding_kwargs={"checkpoint": "/path/to/model.pt", ...}`

This keeps the “deep embedding + classical core” route practical in airgapped / CI
environments: the only required artifact is your checkpoint file.

## ONNX Deployment (Bring Your Own Backbone)

If you want a deployment-friendly embedding route that does not depend on Torch
at inference time, you can export a torchvision backbone to ONNX and use:

- Install: `pip install "pyimgano[onnx]"` (runtime) and `pip install "pyimgano[torch]"` (export)
- `embedding_extractor="onnx_embed"`
- `embedding_kwargs={"checkpoint_path": "/path/to/model.onnx", ...}`

ONNX export helper:

```bash
pyimgano-export-onnx \
  --backbone resnet18 \
  --no-pretrained \
  --out /path/to/resnet18_embed.onnx
```

Then run a wrapper model that expects `checkpoint_path`:

- `vision_onnx_ecod`
- `vision_onnx_knn_cosine_calibrated`

## Score Hygiene

All detectors in `pyimgano` follow:

- higher score = more anomalous

If you combine heterogeneous detectors, consider using:

- `core_score_standardizer` / `vision_score_standardizer`
- `core_score_ensemble` / `vision_score_ensemble`

## Where To Look Next

- Examples:
  - `examples/embedding_plus_core_ecod.py`
  - `examples/openclip_plus_core_knn.py`
- Pipeline model:
  - `pyimgano/models/vision_embedding_core.py`
- Feature extractors:
  - `pyimgano/features/torchvision_backbone.py`
  - `pyimgano/features/torchvision_backbone_gem.py`
  - `pyimgano/features/torchvision_multilayer.py`
  - `pyimgano/features/torchvision_vit_tokens.py`
  - `pyimgano/features/onnx_embed.py`
  - `pyimgano/onnx_export_cli.py`
