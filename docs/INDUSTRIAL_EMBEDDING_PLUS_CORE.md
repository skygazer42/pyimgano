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
  - optional: `openclip_embed`

- **Core detector**: consumes a feature matrix `X: np.ndarray (N, D)`
  - examples: `core_ecod`, `core_knn`, `core_iforest`, `core_lid`, `core_mst_outlier`, `core_crossmad`, `core_torch_autoencoder`

- **Vision wrapper**: consumes images, internally extracts features, then calls the core
  - recommended: `vision_embedding_core`
  - generic: `vision_feature_pipeline`
  - convenience: `vision_resnet18_ecod`, `vision_resnet18_iforest`, `vision_resnet18_knn`, `vision_resnet18_torch_ae`

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
   - `vision_resnet18_ecod` / `vision_resnet18_iforest` / `vision_resnet18_knn`
   - `vision_resnet18_torch_ae` (reconstruction error on embeddings; useful when you want a small amount of deep capacity without a heavy pixel model)

## Safe Defaults (No Downloads)

For industrial reliability, our defaults are chosen to avoid implicit weight downloads.

In particular:

- `torchvision_*` extractors default to `pretrained=False`
- `openclip_embed` defaults to `pretrained=None`

If you want pretrained weights, you must opt in explicitly.

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
