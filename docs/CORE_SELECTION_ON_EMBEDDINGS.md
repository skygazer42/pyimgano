# Core Selection on Embeddings (Industrial Defaults)

This document explains how to choose a `core_*` detector when your inputs are
**deep embeddings** (e.g. from `torchvision_backbone` or OpenCLIP).

Scope:
- **image-level anomaly scoring** (a single score per image)
- **feature-matrix first**: `core_*` consumes an `np.ndarray` (or torch tensor) of shape `(N, D)`
- offline-by-default (no implicit weight downloads)

If you are looking for **pixel anomaly maps**, see the “pixel-first” baselines
and reference-based pipelines instead.

---

## Recommended baseline: standardize scores

Even “good” core detectors can output very different score ranges depending on:
- embedding backbone + pooling choice
- batch normalization / preprocessing
- sample count and feature dimension

In industrial pipelines (and especially when doing comparisons or ensembles),
it is often useful to standardize scores **without labels**.

We recommend wrapping your chosen core detector with:

- `core_score_standardizer` with `method="rank"` (empirical CDF → `[0, 1]`)

Python:

```python
from pyimgano.models.registry import create_model

det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchvision_backbone",
    embedding_kwargs={
        "backbone": "resnet18",
        "pretrained": False,  # important: no implicit downloads
        "pool": "avg",
        "device": "cpu",
    },
    core_detector="core_score_standardizer",
    core_kwargs={
        "base_detector": "core_mahalanobis_shrinkage",
        "base_kwargs": {},
        "method": "rank",
    },
)
```

CLI (benchmark):

```bash
pyimgano-benchmark --dataset manifest --manifest-path ./manifest.jsonl --category bottle \
  --model industrial-embedding-core-balanced --device cpu --no-pretrained
```

---

## Which core_* should I start with?

For embeddings, these are the most stable “industrial-first” choices:

### 1) `core_mahalanobis_shrinkage` (recommended default)

Use when:
- you want a strong baseline with minimal knobs
- your embedding dimension is high and/or your sample count is modest

Why:
- covariance shrinkage (Ledoit–Wolf) is much more stable than raw covariance

Notes:
- Mahalanobis assumes an (approximately) unimodal “normal” embedding cluster.
  If your normal data is clearly multimodal, consider a kNN-style detector.

### 2) `core_knn_cosine` (great with normalized embeddings)

Use when:
- your embeddings are (or can be) L2-normalized
- you expect normal data to be multimodal

Why:
- cosine distance is a natural metric for many deep embeddings
- simple, explainable, and usually hard to “break” with scaling

Knobs:
- `n_neighbors`: larger → smoother; smaller → more sensitive to local outliers
- `method`: `largest|mean|median` aggregation of neighbor distances
- `normalize=True`: row-wise L2 normalization (recommended)

### 3) `core_ecod` (fast, parameter-free baseline)

Use when:
- you want a “no-tuning” baseline to sanity-check your embedding pipeline

Why:
- ECOD is fast and robust as an early benchmark

---

## Practical guidance (industrial heuristics)

### If scores look unstable across runs
- Make sure your embedding extractor is deterministic:
  - fixed `seed` where applicable
  - deterministic transforms (avoid random augmentations during scoring)
- Use `core_score_standardizer(method="rank")` to stabilize scale.

### If anomalies are tiny but “spiky”
- Try `pool="max"` or `pool="gem"` in `torchvision_backbone`.

### If the normal set is clearly multimodal (multiple valid subtypes)
- Prefer kNN-style cores (`core_knn_cosine`, `core_lof`, `core_lid`) over a single Gaussian model.

### If you need to compare detectors fairly
- Standardize scores (rank) and report metrics with a single threshold (SegF1) when masks exist.

