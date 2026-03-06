# Algorithm Selection Guide

This guide helps you choose the right anomaly detection algorithm for your use case.

## Quick Selection Table

| Your Priority | Recommended Algorithms | Why |
|---------------|------------------------|-----|
| **Best Overall Performance** | ECOD, COPOD, IForest | Top performers in ADBench benchmark |
| **No Parameter Tuning** | ECOD, COPOD | Completely parameter-free |
| **Speed (Large Datasets)** | COPOD, KNN, ECOD | O(n×d) complexity, highly optimized |
| **Interpretability** | ECOD, PCA, KNN | Clear mathematical foundation |
| **Local / Cluster Anomalies** | LoOP, LDOF, LOF | Local neighborhood geometry |
| **Simple Statistical Baselines** | Mahalanobis, RZScore, PCA-MD | Strong sanity-check baselines |
| **Small Datasets** | KNN, LOF, OCSVM | Work well with limited data |
| **High-Dimensional Data** | ECOD, COPOD, PCA | Designed for many features |
| **Deep Learning** | Deep SVDD, VAE, AE | Learn complex patterns |
| **Production Ready** | ECOD, COPOD, IForest | Fast, stable, well-tested |

## Pixel Localization (Recommended)

If your goal is **industrial inspection** with **pixel-level localization** (anomaly maps / heatmaps),
start with these “pixel-first” detectors (designed for MVTec AD / VisA-style datasets).

| Detector | When to start | Notes |
|----------|---------------|-------|
| `vision_patchcore` | Strong default pixel localization | Training-free (memory bank) baseline; supports `feature_projection_dim` + `memory_bank_dtype` for speed/memory tradeoffs |
| `vision_softpatch` | “Noisy normal” training set | Filters outlier patches in the memory bank for robustness |
| `vision_anomalydino` | Few-shot / small normal set | DINOv2 patch embeddings + kNN; pass a custom embedder for offline usage |
| `vision_mambaad` | Sequence-model reconstruction | Requires `pyimgano[mamba]`; trains a small Mamba SSM to reconstruct normal patch embeddings |
| `vision_openclip_patchknn` | Semantics-driven patch kNN | Requires `pyimgano[clip]` (OpenCLIP) or an injected embedder |
| `vision_dinomaly_anomalib`, `vision_cfa_anomalib` | You already use anomalib | Inference wrappers; require `pyimgano[anomalib]` + a trained checkpoint |

### Aligned template-style inspection (CPU-friendly)

If your station is stable (pose/background mostly fixed) and you want a fast pixel-map baseline:

| Detector | Start when | Notes |
|----------|------------|-------|
| `vision_pixel_gaussian_map` | You have stable alignment | Per-pixel mean+std z-score map; very fast CPU baseline |
| `vision_pixel_mad_map` | “Noisy normal” training set | Robust per-pixel median+MAD z-score map |
| `vision_pixel_mean_absdiff_map` | You want the simplest baseline | Per-pixel mean template abs-diff map in `[0,1]` |
| `ssim_template_map` | Structure stable, illumination varies | SSIM template similarity map turned into anomaly map |
| `vision_phase_correlation_map` | Small XY misalignment | Phase-corr alignment before abs-diff map |

See also:

- `docs/STRUCTURE_TEMPLATE_BASELINES.md` for when to use `ssim_template`, `ssim_struct`, and structural-feature baselines like `lof_structure`.

## Industrial Integration (numpy-first)

If your upstream system already gives you decoded numpy frames (video pipelines, cameras, backend services),
use the **numpy-first** helpers:

- `pyimgano.inputs.ImageFormat` + `normalize_numpy_image` (explicit formats, no guessing)
- `pyimgano.inference.infer` + `calibrate_threshold`

Recommended starting points:

- `vision_patchcore`, `vision_softpatch`, `vision_anomalydino` (pixel maps + strong baselines)
- `vision_score_ensemble` (robustness via rank-normalized score ensembling)
- `vision_embedding_core` when you already have or want explicit embeddings plus a simple `core_*` detector

Discovery tip:

```bash
pyimgano-benchmark --list-models --tags numpy
pyimgano-benchmark --list-models --tags numpy,pixel_map
```

## Algorithm Categories

### 🔥 Top-Tier Algorithms (Start Here)

#### ECOD (Empirical Cumulative Distribution-based)
**Best for: First-time users, production systems**

```python
detector = models.create_model(
    "vision_ecod",
    feature_extractor=extractor,
    contamination=0.1,
    n_jobs=-1
)
```

**Pros:**
- ✅ Parameter-free (no tuning needed)
- ✅ Top benchmark performance
- ✅ Highly interpretable
- ✅ Fast O(n×d) complexity
- ✅ Works well on high-dimensional data

**Cons:**
- ❌ Needs meaningful numeric features; for vision workflows this depends on your feature extractor

**When to use:** Default choice for most applications

---

#### COPOD (Copula-Based Outlier Detection)
**Best for: Speed-critical applications**

```python
detector = models.create_model(
    "vision_copod",
    feature_extractor=extractor,
    contamination=0.1
)
```

**Pros:**
- ✅ Parameter-free
- ✅ Very fast
- ✅ Excellent benchmark results
- ✅ O(n×d) complexity

**Cons:**
- ❌ Needs meaningful numeric features; for vision workflows this depends on your feature extractor

**When to use:** When speed is critical, large datasets

---

#### Isolation Forest
**Best for: General-purpose anomaly detection**

```python
detector = models.create_model(
    "vision_iforest",
    feature_extractor=extractor,
    n_estimators=100,
    contamination=0.1
)
```

**Pros:**
- ✅ Robust and reliable
- ✅ Top benchmark performer
- ✅ Handles high dimensions well
- ✅ Widely used and trusted

**Cons:**
- ❌ Random forest requires tuning n_estimators
- ❌ Memory intensive for large datasets

**When to use:** Proven reliability needed

---

### 📊 Classic Algorithms

#### KNN (K-Nearest Neighbors)
**Best for: Simple, interpretable results**

```python
detector = models.create_model(
    "vision_knn",
    feature_extractor=extractor,
    n_neighbors=10,
    method='largest'  # or 'mean', 'median'
)
```

**Pros:**
- ✅ Very interpretable
- ✅ Simple and well-understood
- ✅ No training needed (lazy learning)
- ✅ Multiple scoring methods

**Cons:**
- ❌ Slow on large datasets
- ❌ Memory intensive (stores all training data)
- ❌ Requires choosing k

**When to use:** Small datasets, interpretability important

---

#### PCA (Principal Component Analysis)
**Best for: Linear patterns, dimensionality reduction**

```python
detector = models.create_model(
    "vision_pca",
    feature_extractor=extractor,
    n_components=0.95,  # Keep 95% variance
    whiten=True
)
```

**Pros:**
- ✅ Classic, well-understood
- ✅ Fast
- ✅ Interpretable (principal components)
- ✅ Good for linear patterns

**Cons:**
- ❌ Assumes linear relationships
- ❌ May miss nonlinear anomalies

**When to use:** Data has linear structure

---

#### LOF (Local Outlier Factor)
**Best for: Density-based detection**

```python
detector = models.create_model(
    "vision_lof",
    feature_extractor=extractor,
    n_neighbors=20
)
```

**Pros:**
- ✅ Detects local anomalies
- ✅ Works well with clusters
- ✅ Classic algorithm

**Cons:**
- ❌ Requires tuning n_neighbors
- ❌ Slow on large datasets

**When to use:** Data has varying densities

---

#### KDE (Kernel Density Estimation)
**Best for: Simple density baseline**

```python
detector = models.create_model(
    "vision_kde",
    feature_extractor=extractor,
    bandwidth=1.0,
    contamination=0.1,
)
```

**Pros:**
- ✅ Strong non-parametric baseline
- ✅ Works well for low/medium-dimensional features

**Cons:**
- ❌ Can be slow on very large datasets
- ❌ Sensitive to bandwidth choice

**When to use:** You want a density baseline for extracted features

---

#### GMM (Gaussian Mixture Model)
**Best for: Parametric density baseline**

```python
detector = models.create_model(
    "vision_gmm",
    feature_extractor=extractor,
    n_components=1,
    contamination=0.1,
)
```

**Pros:**
- ✅ Fast, classic statistical baseline
- ✅ Can model multi-modal normal data with `n_components>1`

**Cons:**
- ❌ Sensitive to `n_components` and initialization

**When to use:** Normal data is multi-modal and you want a simple baseline

---

#### SOS (Stochastic Outlier Selection)
**Best for: Probabilistic neighborhood baseline**

```python
detector = models.create_model(
    "vision_sos",
    feature_extractor=extractor,
    perplexity=4.5,
    contamination=0.1,
)
```

**When to use:** You want a probabilistic kNN-style baseline without heavy tuning

---

#### SOD (Subspace Outlier Detection)
**Best for: Subspace-style outliers**

```python
detector = models.create_model(
    "vision_sod",
    feature_extractor=extractor,
    n_neighbors=20,
    ref_set=10,
    contamination=0.1,
)
```

**When to use:** Features have many irrelevant dimensions and outliers appear in subspaces

---

#### Robust MAD (Median Absolute Deviation)
**Best for: Fast, robust baseline for feature vectors**

```python
detector = models.create_model(
    "vision_mad",
    feature_extractor=extractor,
    aggregation="max",  # max|mean|l2
    contamination=0.1,
)
```

**When to use:** You want a very fast, robust baseline that tolerates outliers in the training set

---

### 🧠 Deep Learning Algorithms

#### Deep SVDD
**Best for: Complex nonlinear patterns**

```python
detector = models.create_model(
    "vision_deep_svdd",
    feature_extractor=extractor,
    epochs=50,
    hidden_neurons=[128, 64]
)
```

**Pros:**
- ✅ Learns complex patterns
- ✅ End-to-end trainable
- ✅ Powerful representation

**Cons:**
- ❌ Requires more data
- ❌ Slower training
- ❌ More parameters to tune

**When to use:** Large datasets, complex patterns

---

#### VAE (Variational Autoencoder)
**Best for: Probabilistic modeling**

```python
detector = models.create_model(
    "vision_vae",
    feature_extractor=extractor,
    encoder_neurons=[128, 64],
    latent_dim=32,
    epochs=100
)
```

**Pros:**
- ✅ Probabilistic framework
- ✅ Generates samples
- ✅ Smooth latent space

**Cons:**
- ❌ Requires substantial data
- ❌ Complex to tune
- ❌ Slower training

**When to use:** Need probabilistic scores, generative model

---

## Decision Tree

```
Do you need parameter-free solution?
├─ YES → Use ECOD or COPOD
└─ NO
   └─ Is speed critical?
      ├─ YES → Use COPOD or KNN
      └─ NO
         └─ Is interpretability important?
            ├─ YES → Use ECOD, PCA, or KNN
            └─ NO
               └─ Large dataset with complex patterns?
                  ├─ YES → Use Deep SVDD or VAE
                  └─ NO → Use Isolation Forest
```

## Benchmark Results

Based on [ADBench](https://github.com/Minqi824/ADBench) (30 algorithms on 57 datasets):

### Top Performers (Average Rank)
1. **ECOD** - Rank 3.5/30
2. **COPOD** - Rank 4.2/30
3. **IForest** - Rank 5.8/30
4. **LODA** - Rank 6.1/30

### Speed Comparison (Relative)
- **Fastest:** COPOD (1.0×)
- **Very Fast:** ECOD (1.2×), KNN (1.5×)
- **Fast:** IForest (2.0×), PCA (2.1×)
- **Moderate:** LOF (3.5×), OCSVM (4.0×)
- **Slow:** Deep SVDD (10×+), VAE (15×+)

## Use Case Examples

### Industrial Quality Control
**Scenario:** Detect defects in manufactured products

**Recommended:** ECOD or COPOD
- Fast enough for real-time inspection
- No parameter tuning (stable in production)
- High accuracy on visual defects

```python
detector = models.create_model("vision_ecod", ...)
```

---

### Medical Imaging
**Scenario:** Identify abnormal X-rays/scans

**Recommended:** Deep SVDD or VAE
- Can learn subtle patterns
- Probabilistic scores useful for doctors
- Worth the computational cost

```python
detector = models.create_model("vision_deep_svdd", ...)
```

---

### Security/Surveillance
**Scenario:** Detect unusual behavior in video frames

**Recommended:** COPOD or IForest
- Real-time processing needed
- High accuracy required
- Robust to varying conditions

```python
detector = models.create_model("vision_copod", ...)
```

---

### Research/Experimentation
**Scenario:** Exploring different approaches

**Recommended:** Try multiple
- ECOD (baseline)
- Deep SVDD (deep learning)
- Your domain-specific method

```python
for algo in ["vision_ecod", "vision_copod", "vision_deep_svdd"]:
    detector = models.create_model(algo, ...)
    # Compare results
```

---

## Parameter Tuning Guide

### Contamination
**What it is:** Expected proportion of outliers (0 < contamination < 0.5)

**How to set:**
- **Known outlier ratio:** Use that value
- **Unknown:** Start with 0.1 (10%)
- **Very rare defects:** Try 0.01-0.05
- **Balanced dataset:** Try 0.1-0.2

### Number of Neighbors (KNN, LOF)
**What it is:** How many neighbors to consider

**How to set:**
- **Small datasets (<100):** Try 5-10
- **Medium datasets (100-1000):** Try 10-20
- **Large datasets (>1000):** Try 20-50
- **Rule of thumb:** √n where n = dataset size

### Number of Components (PCA)
**What it is:** How many principal components to keep

**How to set:**
- **Variance-based:** 0.90-0.99 (keep 90-99% variance)
- **Fixed number:** 10-50 components
- **Rule of thumb:** min(n_samples/10, n_features/2)

## Common Pitfalls

### ❌ Don't
- Use deep learning on small datasets (<1000 samples)
- Ignore parameter tuning for KNN/LOF
- Use slow algorithms for real-time applications
- Forget to validate on test data

### ✅ Do
- Start with ECOD or COPOD
- Use cross-validation for parameter selection
- Benchmark multiple algorithms
- Monitor performance in production
- Use appropriate feature extractors

## Further Resources

- **scikit-learn Outlier Detection:** https://scikit-learn.org/stable/modules/outlier_detection.html
- **ADBench Paper:** https://arxiv.org/abs/2206.09426
- **ECOD Paper:** Li et al., TKDE 2022
- **COPOD Paper:** Li et al., ICDM 2020

---

**Need help?** Check examples in `examples/` directory or open an issue on GitHub.
