# Deep Learning Models for Visual Anomaly Detection

This guide covers the state-of-the-art deep learning algorithms available in PyImgAno for visual anomaly detection.

## 📊 Quick Comparison

| Algorithm | Year | Speed | Accuracy | Memory | Training Time | Use Case |
|-----------|------|-------|----------|--------|---------------|----------|
| **AnomalyDINO** ⭐ | 2025 | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Minutes | Few-shot, foundation-style |
| **MambaAD** ⭐ | 2024 | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Minutes | Patch reconstruction, sequence modeling |
| **SimpleNet** ⭐ | 2023 | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Low | Minutes | Production, Fast deployment |
| **PatchCore** ⭐ | 2022 | ⚡⚡ | ⭐⭐⭐⭐⭐ | Medium | No training | High accuracy needed |
| **SoftPatch** ⭐ | - | ⚡⚡ | ⭐⭐⭐⭐ | Medium | No training | Noisy-normal robustness, localization |
| **PaDiM** | 2021 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Low | No training | Edge devices |
| **STFPM** | 2021 | ⚡⚡ | ⭐⭐⭐⭐ | Medium | Hours | Balanced performance |
| **FastFlow** | 2021 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Medium | Hours | Fast inference |
| **EfficientAd** | 2023 | ⚡⚡⚡ | ⭐⭐⭐⭐ | Low | Hours | Resource-constrained |

---

## 🏆 State-of-the-Art Models

### 1. SimpleNet (CVPR 2023) ⭐⭐⭐

**Ultra-fast training with SOTA performance**

```python
from pyimgano.models import create_model

detector = create_model(
    'vision_simplenet',
    epochs=10,           # Only 10 epochs needed!
    batch_size=8,
    feature_dim=384,
    device='cuda'
)

detector.fit(train_images)
scores = detector.decision_function(test_images)
```

**Key Features:**
- ⚡ **Ultra-fast**: 10x faster training than PatchCore
- 🎯 **SOTA accuracy**: Competitive or better than complex methods
- 💾 **Lightweight**: Only ~1M trainable parameters
- 🚀 **Simple**: No complex architecture, easy to understand

**When to Use:**
- Production environments requiring fast deployment
- Limited computational resources
- Need to quickly iterate on models
- Real-time applications

**Technical Details:**
- Pre-trained WideResNet50 backbone (frozen)
- Small adapter network (~1M parameters)
- Local neighborhood discriminator
- Cosine similarity-based anomaly scoring

**Performance:**
- MVTec AD: ~99% AUROC on most categories
- Training time: ~10 minutes on GPU for typical dataset
- Inference: ~50ms per image on GPU

---

### 2. PatchCore (CVPR 2022) ⭐⭐⭐

**Training-free, memory-based SOTA detection**

```python
detector = create_model(
    'vision_patchcore',
    backbone='wide_resnet50',
    coreset_sampling_ratio=0.1,  # Keep 10% of patches
    n_neighbors=9,
    device='cuda'
)

detector.fit(train_images)  # No training, just feature extraction
scores = detector.decision_function(test_images)

# Get pixel-level anomaly map
anomaly_map = detector.get_anomaly_map('test_image.jpg')
```

**Key Features:**
- 🎯 **SOTA accuracy**: Top performance on MVTec AD benchmark
- 🚫 **No training**: Only feature extraction and memory bank creation
- 📍 **Localization**: Excellent pixel-level anomaly localization
- 🧠 **Memory-based**: Uses coreset sampling for efficiency

**When to Use:**
- Highest accuracy is critical
- Need precise anomaly localization
- Can't afford training time
- Have sufficient memory for feature bank

**Technical Details:**
- Locally aware patch-level representations
- Greedy coreset subsampling
- k-NN based anomaly scoring
- Multi-scale feature aggregation

**Performance:**
- MVTec AD: 99.6% image-level AUROC
- Pixel-level: 98.1% AUROC
- No training time, only feature extraction
- Memory: Depends on coreset ratio

---

### 2.5 SoftPatch (robust patch-memory) ⭐ NEW

**Patch-memory detector tuned for “noisy normal” training data (industrial inspection reality)**

`vision_softpatch` follows a Patch-kNN workflow similar to PatchCore/AnomalyDINO, but adds a
training-time memory-bank filtering step to reduce the impact of outlier patches in your
normal/reference set.

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_softpatch",
    # Remove the top 10% most outlier-looking train patches from the memory bank.
    train_patch_outlier_quantile=0.10,
    # Optional: reduce the remaining memory bank for speed.
    coreset_sampling_ratio=0.25,
)

detector.fit(train_images)
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

**When to Use:**
- Your “normal” training set likely includes some bad frames / contaminated samples
- You want strong pixel localization but more robustness than a plain patch-memory baseline

**Key Parameters:**
- `train_patch_outlier_quantile`: fraction of training patches removed as outliers (0.0–<1.0)
- `coreset_sampling_ratio`: additional sampling ratio for the remaining memory bank (0.0–1.0)

---

### 3. AnomalyDINO (WACV 2025) ⭐ NEW

**Few-shot friendly foundation-style detector (DINOv2 patch embeddings + kNN)**

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_anomalydino",
    device="cuda",        # optional
    contamination=0.1,
)

# Fit on normal/reference images (builds a patch memory bank)
detector.fit(train_images)

scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

**Key Features:**
- 🧠 **Foundation backbone**: DINOv2 patch embeddings
- 🎯 **Few-shot**: Works well with small normal/reference sets
- 📍 **Localization**: Produces pixel-level anomaly maps
- ⚡ **Inference-first**: No gradient training in the default workflow

**Notes:**
- The default embedder uses `torch.hub` to load DINOv2 weights on first run.
- For offline/enterprise usage, pass a custom `embedder=...`.
- For faster kNN search on large memory banks, install `pyimgano[faiss]`.

---

### 3.5 MambaAD (NeurIPS 2024) ⭐ NEW

**Sequence-model reconstruction on frozen patch embeddings**

`vision_mambaad` follows a practical MambaAD-style workflow:
- extract fixed-grid patch embeddings (default: DINOv2)
- train a small Mamba SSM to reconstruct normal patch patterns
- use reconstruction error for image scores + pixel anomaly maps

```python
from pyimgano.models import create_model

# Requires:
#   pip install "pyimgano[mamba]"
detector = create_model(
    "vision_mambaad",
    device="cuda",   # or "cpu"
    epochs=5,
    batch_size=8,
    lr=1e-3,
)

detector.fit(train_images)
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

**When to Use:**
- You want a learnable detector that can model long-range patch dependencies
- You need pixel-level maps but prefer a reconstruction-style training loop

---

### 4. STFPM (BMVC 2021)

**Student-Teacher feature pyramid matching**

```python
detector = create_model(
    'vision_stfpm',
    backbone='resnet18',
    layers=['layer1', 'layer2', 'layer3'],
    epochs=100,
    batch_size=32,
    lr=0.4,
    device='cuda'
)

detector.fit(train_images)
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map('test_image.jpg')
```

**Key Features:**
- 👨‍🏫 **Knowledge distillation**: Student learns from frozen teacher
- 🏔️ **Multi-scale**: Feature pyramid for different resolutions
- 📊 **Good accuracy**: Competitive performance
- 💡 **Interpretable**: Clear teacher-student paradigm

**When to Use:**
- Balanced accuracy and speed needed
- Multi-scale feature matching is beneficial
- Interpretable method preferred
- Medium-sized datasets

**Technical Details:**
- Pre-trained ResNet18 teacher (frozen)
- Trainable ResNet18 student
- MSE loss for feature alignment
- Multi-layer feature extraction

**Performance:**
- MVTec AD: ~97% image-level AUROC
- Training time: 2-3 hours on GPU
- Inference: ~100ms per image

---

## 🎯 Model Selection Guide

### By Use Case

**🚀 Production Deployment (Speed + Accuracy)**
```
1. SimpleNet      - Ultra-fast, SOTA accuracy
2. PaDiM          - Fast, good for edge devices
3. EfficientAd    - Resource-efficient
```

**🎯 Maximum Accuracy**
```
1. PatchCore      - Best overall accuracy
2. SimpleNet      - Competitive, much faster
3. STFPM          - Good accuracy, interpretable
```

**📍 Anomaly Localization**
```
1. PatchCore      - Best pixel-level accuracy
2. SoftPatch      - More robust when train normals are noisy
3. STFPM          - Good multi-scale localization
4. PaDiM          - Fast localization
```

**⚡ Fast Training**
```
1. SimpleNet      - 10 epochs (~10 min)
2. PatchCore      - No training needed
3. PaDiM          - No training needed
```

**💾 Low Memory**
```
1. SimpleNet      - Small adapter network
2. PaDiM          - Compact statistics
3. EfficientAd    - Memory-efficient
```

---

## 🔌 Optional Backends

### anomalib checkpoint wrapper (inference-only)

If you already have checkpoints trained with Intel's `anomalib`, you can load them directly and reuse
PyImgAno's unified API / post-processing / evaluation.

Install:

```bash
pip install pyimgano[anomalib]
```

Usage:

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_anomalib_checkpoint",  # or: vision_patchcore_anomalib, vision_padim_anomalib, ...
    checkpoint_path="/path/to/anomalib.ckpt",
    device="cuda",
    contamination=0.1,
)

detector.fit(train_images)  # calibrates a score threshold only
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

### OpenCLIP detectors (prompt-score / patch-kNN)

PyImgAno ships two lightweight CLIP-style detectors backed by `open_clip_torch` (imported as `open_clip`).

Install:

```bash
pip install "pyimgano[clip]"
```

Prompt-score detector (prompt-based scoring + anomaly maps):

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_openclip_promptscore",
    device="cuda",  # or "cpu"
    contamination=0.1,
    class_name="screw",
    openclip_model_name="ViT-B-32",
    openclip_pretrained="laion2b_s34b_b79k",
)

detector.fit(train_images)  # calibrates a score threshold only
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

Patch-kNN detector (OpenCLIP ViT patch embeddings + kNN memory bank):

```python
detector = create_model(
    "vision_openclip_patchknn",
    device="cuda",
    contamination=0.1,
    openclip_model_name="ViT-B-32",
    openclip_pretrained="laion2b_s34b_b79k",
)

detector.fit(train_images)  # builds patch memory bank
scores = detector.decision_function(test_images)
anomaly_map = detector.get_anomaly_map(test_images[0])
```

Notes:
- OpenCLIP weights are cached by torch (default: `~/.cache/torch`, configurable via `TORCH_HOME`).
- If you want unit-test friendly behavior or custom feature extraction, pass `embedder=...` and
  `text_features_normal=`/`text_features_anomaly=` (prompt-score).


## 📖 Detailed Algorithm Descriptions

### PaDiM (Patch Distribution Modeling)

**ICPR 2021 - Fast and lightweight**

```python
detector = create_model(
    'padim',
    backbone='resnet18',
    d_reduced=128,
    device='cpu'  # Works well on CPU
)
```

- Uses pre-trained CNN features
- Models patch-level distributions
- Gaussian distribution for normal patterns
- Mahalanobis distance for scoring

**Best for:** Edge devices, CPU inference, fast deployment

---

### FastFlow

**ICLR 2022 - Fast normalizing flow**

```python
detector = create_model(
    'fastflow',
    backbone='resnet18',
    device='cuda'
)
```

- Normalizing flow for distribution modeling
- Fast training and inference
- Good accuracy-speed trade-off

**Best for:** When both training and inference speed matter

---

### EfficientAd

**2023 - Resource-efficient detection**

```python
detector = create_model(
    'efficientad',
    device='cpu'
)
```

- Designed for limited resources
- Good accuracy with low memory
- Suitable for embedded systems

**Best for:** IoT, embedded systems, resource constraints

---

## 🔧 Advanced Usage

### Custom Backbones

```python
# PatchCore with ResNet50
detector = create_model(
    'vision_patchcore',
    backbone='resnet50',  # or 'wide_resnet50'
    device='cuda'
)

# SimpleNet with different feature dimensions
detector = create_model(
    'vision_simplenet',
    backbone='wide_resnet50',
    feature_dim=512,  # Larger feature space
    device='cuda'
)
```

### Memory Optimization

```python
# PatchCore with aggressive coreset sampling
detector = create_model(
    'vision_patchcore',
    coreset_sampling_ratio=0.01,  # Keep only 1%
    device='cuda'
)

# STFPM with smaller batch size
detector = create_model(
    'vision_stfpm',
    batch_size=8,  # Reduce memory usage
    device='cuda'
)
```

### GPU Acceleration

```python
import torch

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create model with GPU
detector = create_model('vision_simplenet', device=device)

# Mixed precision training (for STFPM, SimpleNet)
# Note: Requires additional setup in training loop
```

### Anomaly Visualization

```python
import matplotlib.pyplot as plt
import cv2

# Get anomaly map
anomaly_map = detector.get_anomaly_map('test_image.jpg')

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
img = cv2.imread('test_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
axes[0].imshow(img)
axes[0].set_title('Original Image')

# Anomaly heatmap
im = axes[1].imshow(anomaly_map, cmap='jet')
axes[1].set_title('Anomaly Heatmap')
plt.colorbar(im, ax=axes[1])

# Overlay
overlay = img.copy()
heatmap_colored = cv2.applyColorMap(
    (anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_JET
)
overlay = cv2.addWeighted(overlay, 0.6, heatmap_colored, 0.4, 0)
axes[2].imshow(overlay)
axes[2].set_title('Overlay')

plt.tight_layout()
plt.show()
```

---

## 📊 Benchmarks

### MVTec AD Dataset (Image-level AUROC)

| Category | SimpleNet | PatchCore | PaDiM | STFPM |
|----------|-----------|-----------|-------|-------|
| Carpet   | 99.1%     | 99.6%     | 99.1% | 98.7% |
| Grid     | 99.8%     | 99.9%     | 98.2% | 98.9% |
| Leather  | 100%      | 100%      | 99.2% | 99.3% |
| Tile     | 99.6%     | 99.7%     | 97.3% | 98.1% |
| Wood     | 99.7%     | 99.8%     | 99.1% | 99.0% |
| **Avg**  | **99.2%** | **99.6%** | **98.7%** | **98.4%** |

### Speed Comparison (per image)

| Model | Training Time | Inference (CPU) | Inference (GPU) |
|-------|---------------|-----------------|-----------------|
| SimpleNet | ~10 min | 500ms | 50ms |
| PatchCore | None | 200ms | 80ms |
| PaDiM | None | 100ms | 40ms |
| STFPM | 2-3 hours | 300ms | 100ms |
| FastFlow | 1-2 hours | 150ms | 60ms |

*Note: Times are approximate and depend on hardware/dataset*

---

## 🎓 Best Practices

### 1. Start Simple, Scale Up

```python
# Start with SimpleNet for quick experimentation
detector = create_model('vision_simplenet', epochs=10)
detector.fit(train_images)

# If accuracy not sufficient, try PatchCore
detector = create_model('vision_patchcore', coreset_sampling_ratio=0.1)
detector.fit(train_images)
```

### 2. Use Appropriate Coreset Ratios

```python
# Small dataset (<100 images): Use higher ratio
detector = create_model('vision_patchcore', coreset_sampling_ratio=0.5)

# Medium dataset (100-1000 images): Balanced
detector = create_model('vision_patchcore', coreset_sampling_ratio=0.1)

# Large dataset (>1000 images): Lower ratio
detector = create_model('vision_patchcore', coreset_sampling_ratio=0.01)
```

### 3. Validate on Held-out Set

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Split data
train_imgs, val_imgs = train_test_split(normal_images, test_size=0.2)

# Train
detector.fit(train_imgs)

# Validate
val_scores = detector.decision_function(val_imgs + anomaly_imgs)
val_labels = [0] * len(val_imgs) + [1] * len(anomaly_imgs)

auroc = roc_auc_score(val_labels, val_scores)
print(f"Validation AUROC: {auroc:.4f}")
```

### 4. Save and Load Models

```python
import pickle

# Train and save
detector = create_model('vision_simplenet', epochs=10)
detector.fit(train_images)

with open('detector.pkl', 'wb') as f:
    pickle.dump(detector, f)

# Load and use
with open('detector.pkl', 'rb') as f:
    detector = pickle.load(f)

scores = detector.decision_function(test_images)
```

---

## 🔍 Troubleshooting

### Out of Memory (OOM)

**Solution 1: Reduce batch size**
```python
detector = create_model('vision_stfpm', batch_size=4)  # Instead of 32
```

**Solution 2: Use CPU**
```python
detector = create_model('vision_simplenet', device='cpu')
```

**Solution 3: Lower coreset ratio**
```python
detector = create_model('vision_patchcore', coreset_sampling_ratio=0.01)
```

### Slow Training

**Solution 1: Use SimpleNet**
```python
# 10x faster than other methods
detector = create_model('vision_simplenet', epochs=10)
```

**Solution 2: Reduce epochs**
```python
detector = create_model('vision_stfpm', epochs=50)  # Instead of 100
```

**Solution 3: Use GPU**
```python
detector = create_model('vision_simplenet', device='cuda')
```

### Low Accuracy

**Solution 1: Try PatchCore**
```python
detector = create_model('vision_patchcore')  # Best accuracy
```

**Solution 2: Increase training data**
```python
# Ensure sufficient normal samples (>50 recommended)
```

**Solution 3: Increase feature dimensions**
```python
detector = create_model('vision_simplenet', feature_dim=512)
```

---

## 📚 References

1. **SimpleNet (CVPR 2023)**
   - Liu, Z., Zhou, Y., Xu, Y., & Wang, Z.
   - "SimpleNet: A Simple Network for Image Anomaly Detection and Localization"

2. **PatchCore (CVPR 2022)**
   - Roth, K., et al.
   - "Towards Total Recall in Industrial Anomaly Detection"

3. **STFPM (BMVC 2021)**
   - Wang, G., Han, S., Ding, E., & Huang, D.
   - "Student-Teacher Feature Pyramid Matching for Anomaly Detection"

4. **PaDiM (ICPR 2021)**
   - Defard, T., et al.
   - "PaDiM: A Patch Distribution Modeling Framework"

5. **FastFlow (ICLR 2022)**
   - Yu, J., et al.
   - "FastFlow: Unsupervised Anomaly Detection and Localization"

---

## 🤝 Contributing

Want to add a new deep learning algorithm? See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Popular algorithms to consider:
- DRAEM (reconstruction-based)
- Reverse Distillation
- CFlow-AD (conditional normalizing flow)
- DSR (Discriminative Self-supervised Reconstruction)

---

## 📞 Support

- **Documentation**: [docs/](.)
- **Issues**: [GitHub Issues](https://github.com/jhlu2019/pyimgano/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jhlu2019/pyimgano/discussions)
