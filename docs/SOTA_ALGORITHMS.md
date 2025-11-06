# State-of-the-Art Algorithms in PyImgAno

PyImgAno includes the latest state-of-the-art (SOTA) algorithms from top computer vision conferences. This document provides an overview of the cutting-edge methods available.

## 🏆 Latest SOTA Algorithms (2023-2024)

### WinCLIP (CVPR 2023) ⭐⭐⭐

**Paper**: "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation"
**Key Innovation**: Zero-shot anomaly detection using CLIP's visual-language understanding

**Highlights**:
- ✅ **Zero-shot capability** - No anomaly samples needed for training
- ✅ **Few-shot learning** - Works with minimal normal samples
- ✅ **Strong localization** - Window-based attention for precise anomaly maps
- ✅ **No fine-tuning** - Leverages pre-trained CLIP directly

**When to use**:
- When you have very limited training data
- For rapid prototyping without training
- When you can describe anomalies in text
- For multi-class anomaly detection

**Example**:
```python
from pyimgano.models import create_model

# Zero-shot detection
detector = create_model(
    "winclip",
    clip_model="ViT-B/32",
    k_shot=0  # Zero-shot
)

detector.set_class_name("screw")  # Describe the object
scores = detector.predict_proba(test_images)

# With anomaly localization
anomaly_maps = detector.predict_anomaly_map(test_images)
```

### SimpleNet (CVPR 2023) ⭐⭐⭐

**Paper**: "SimpleNet: A Simple Network for Image Anomaly Detection and Localization"
**Key Innovation**: Ultra-fast one-stage detection with comparable accuracy to complex methods

**Highlights**:
- ⚡ **Ultra-fast** - 100+ FPS on single GPU
- 🎯 **High accuracy** - Matches PatchCore performance
- 💾 **Memory efficient** - Small model size
- 🚀 **Easy to train** - Simple architecture, fast convergence

**When to use**:
- Real-time applications
- Resource-constrained environments
- When speed is critical
- Industrial inspection systems

### DifferNet (WACV 2023) ⭐⭐

**Paper**: "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
**Key Innovation**: Learns to detect anomalies via learnable difference with k-NN

**Highlights**:
- 🧠 **Learnable differences** - Trains a network to compute meaningful differences
- 🎯 **k-NN augmented** - Combines k-NN with deep learning
- 📍 **Good localization** - Multi-scale feature comparison
- 🔧 **Flexible** - Works with various backbones

**When to use**:
- When you need both detection and localization
- For subtle anomalies
- When you have sufficient normal samples
- For fine-grained defect detection

**Example**:
```python
detector = create_model(
    "differnet",
    backbone="wide_resnet50",
    k_neighbors=5,
    train_difference=True  # Learn difference module
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

## 📅 Recent SOTA Algorithms (2020-2022)

### CutPaste (CVPR 2021) ⭐⭐

**Paper**: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
**Key Innovation**: Self-supervised learning via cutting and pasting image patches

**Highlights**:
- 🎨 **Self-supervised** - No anomaly samples needed
- 🔄 **Simple augmentation** - Easy to implement and understand
- 🎯 **Effective** - Strong performance on industrial datasets
- 🚀 **Fast training** - Converges quickly

**When to use**:
- When you only have normal samples
- For texture-based anomaly detection
- When you want interpretable augmentations
- For defect detection in manufacturing

**Example**:
```python
detector = create_model(
    "cutpaste",
    backbone="resnet18",
    augment_type="normal",  # or "scar", "3way"
    epochs=256
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

### SPADE (ECCV 2020) ⭐⭐⭐

**Paper**: "Sub-Image Anomaly Detection with Deep Pyramid Correspondences"
**Key Innovation**: Multi-scale k-NN feature matching with deep pyramid for excellent localization

**Highlights**:
- 🎯 **Excellent localization** - Pixel-perfect anomaly maps
- 📊 **Multi-scale** - Deep pyramid features capture various scales
- ⚡ **Fast inference** - Efficient k-D tree for k-NN
- 🏆 **High accuracy** - SOTA on MVTec AD localization
- 💾 **Memory efficient** - Only stores normal features

**When to use**:
- When pixel-level localization is critical
- For defects of varying sizes
- Industrial quality control with detailed maps
- When you need both detection and precise segmentation

**Example**:
```python
detector = create_model(
    "spade",
    backbone="wide_resnet50",
    k_neighbors=50,
    feature_levels=["layer1", "layer2", "layer3"]
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
anomaly_maps = detector.predict_anomaly_map(test_images)
```

### RIAD (2020) ⭐⭐

**Paper**: "Reconstruction by Inpainting for Visual Anomaly Detection"
**Key Innovation**: Self-supervised learning via image decomposition and inpainting

**Highlights**:
- 🎨 **Self-supervised** - No anomaly samples needed
- 🔄 **Inpainting-based** - Learns to reconstruct masked regions
- 🎯 **Effective** - Good performance on texture anomalies
- 🚀 **Simple architecture** - U-Net based reconstruction

**When to use**:
- When you only have normal samples
- For texture and surface defect detection
- When you want self-supervised learning
- For manufacturing quality control

**Example**:
```python
detector = create_model(
    "riad",
    backbone="resnet18",
    grid_size=8,
    epochs=100
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

### MemSeg (2022) ⭐⭐

**Paper**: "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder"
**Key Innovation**: Memory-guided segmentation with attention mechanisms

**Highlights**:
- 🧠 **Memory bank** - Stores prototypical normal patterns
- 🎯 **Attention-based** - Learns to query relevant memories
- 📍 **Good segmentation** - Produces detailed anomaly maps
- 🔧 **Flexible** - Adapts to various defect types

**When to use**:
- When you need semantic understanding
- For complex anomaly patterns
- When memory-based approaches are preferred
- Research and comparison

**Example**:
```python
detector = create_model(
    "memseg",
    memory_size=1000,
    feature_dim=512,
    k_neighbors=3
)

detector.fit(normal_images)
scores = detector.predict_proba(test_images)
```

### PatchCore (CVPR 2022) ⭐⭐⭐

**Paper**: "Towards Total Recall in Industrial Anomaly Detection"
**Key Innovation**: Coreset-based memory bank for efficient feature matching

**Highlights**:
- 🏆 **SOTA accuracy** - Best performance on MVTec AD
- 💾 **Memory efficient** - Coreset reduces memory footprint
- 📍 **Precise localization** - Patch-level anomaly maps
- ⚡ **Fast inference** - Efficient nearest neighbor search

**When to use**:
- When accuracy is critical
- For pixel-level anomaly localization
- Industrial quality control
- Benchmark comparisons

### STFPM (BMVC 2021) ⭐⭐

**Paper**: "Student-Teacher Feature Pyramid Matching"
**Key Innovation**: Multi-scale student-teacher knowledge distillation

**Highlights**:
- 🎓 **Knowledge distillation** - Teacher-student framework
- 🔍 **Multi-scale** - Feature pyramid for different scales
- 🎯 **Strong localization** - Pixel-wise anomaly maps
- 🚀 **End-to-end training** - Simple optimization

**When to use**:
- For multi-scale anomaly detection
- When you need detailed localization
- For defects of varying sizes
- Educational/research purposes

### DRAEM (ICCV 2021) ⭐⭐

**Paper**: "DRAEM: A Discriminatively Trained Reconstruction Embedding"
**Key Innovation**: Discriminative reconstruction with synthetic anomalies

**Highlights**:
- 🎭 **Synthetic anomalies** - Generates realistic defects
- 🔍 **Discriminative** - Not just reconstruction-based
- 📍 **Segmentation** - Pixel-level anomaly maps
- 🎯 **Robust** - Works well on various datasets

**When to use**:
- When you need segmentation maps
- For texture-based anomalies
- When reconstruction alone isn't enough
- Research and comparison

### CFlow-AD (WACV 2022) ⭐⭐

**Paper**: "CFLOW-AD: Real-Time Unsupervised Anomaly Detection"
**Key Innovation**: Conditional normalizing flows for anomaly scoring

**Highlights**:
- ⚡ **Real-time** - Fast inference
- 📊 **Probabilistic** - Principled likelihood-based scoring
- 🔄 **Flexible** - Normalizing flows are expressive
- 🎯 **Good performance** - Competitive with PatchCore

**When to use**:
- Real-time applications
- When you want probabilistic scores
- For research on normalizing flows
- When speed and accuracy both matter

### FastFlow (AAAI 2022) ⭐⭐

**Paper**: "FastFlow: Unsupervised Anomaly Detection via 2D Normalizing Flows"
**Key Innovation**: 2D normalizing flows for fast and accurate detection

**Highlights**:
- ⚡ **Very fast** - Faster than PatchCore
- 🎯 **High accuracy** - Near SOTA performance
- 💡 **Innovative** - 2D flows for spatial modeling
- 🚀 **Easy to train** - Stable optimization

**When to use**:
- When you need speed without sacrificing accuracy
- For large-scale deployment
- Industrial real-time inspection
- Research on flows

## 🏷️ Weakly-Supervised Algorithms

### DevNet (KDD 2019) ⭐⭐

**Paper**: "Deep Anomaly Detection with Deviation Networks"
**Key Innovation**: Deviation loss for weakly-supervised learning with few anomaly labels

**Highlights**:
- 🏷️ **Weakly-supervised** - Needs only few labeled anomaly samples
- 🎯 **Deviation loss** - Learns to score based on deviation from normal
- 🚀 **Good generalization** - Works well with limited labels
- 🔧 **Flexible architecture** - Customizable network depth

**When to use**:
- When you have a few labeled anomaly samples
- For scenarios where unsupervised methods struggle
- When you can afford minimal labeling effort
- Research on weakly-supervised learning

**Example**:
```python
# DevNet requires labeled data (0=normal, 1=anomaly)
detector = create_model(
    "devnet",
    backbone="resnet18",
    hidden_dims=[128, 64],
    margin=5.0
)

# y contains both normal (0) and anomaly (1) labels
detector.fit(train_images, y=train_labels)
scores = detector.predict_proba(test_images)
```

**Note**: Unlike most other algorithms in PyImgAno which are unsupervised, DevNet requires a small number of labeled anomaly samples during training. This makes it suitable for scenarios where obtaining a few anomaly labels is feasible.

## 📊 Algorithm Comparison

### Performance on MVTec AD

| Algorithm | AUC-ROC (Image) | AUC-ROC (Pixel) | Speed (FPS) | Year |
|-----------|----------------|-----------------|-------------|------|
| **WinCLIP** | ~95% | ~98% | 5-10 | 2023 |
| **SimpleNet** | ~99% | ~98% | 100+ | 2023 |
| **PatchCore** | **99.6%** | **98.7%** | 30-50 | 2022 |
| **DifferNet** | ~97% | ~97% | 20-40 | 2023 |
| **SPADE** | ~98% | **~99%** | 40-60 | 2020 |
| **CutPaste** | ~96% | N/A | 50+ | 2021 |
| **RIAD** | ~96% | ~97% | 30-50 | 2020 |
| **MemSeg** | ~97% | ~98% | 20-40 | 2022 |
| **STFPM** | ~97% | ~98% | 40-60 | 2021 |
| **DRAEM** | ~98% | ~98% | 30-50 | 2021 |
| **FastFlow** | ~99% | ~98% | 60-80 | 2022 |
| **CFlow-AD** | ~98% | ~97% | 50-70 | 2022 |
| **DevNet*** | ~95% | N/A | 40-60 | 2019 |

*Note: Performance varies by category and implementation*
*DevNet is weakly-supervised and requires labeled anomaly samples*

### Speed vs Accuracy Trade-off

```
High Accuracy, Slower:
├── PatchCore (99.6% AUC, 30-50 FPS) ⭐ Best overall
├── SPADE (98% AUC, 40-60 FPS) - Excellent localization
├── DRAEM (98% AUC, 30-50 FPS)
└── FastFlow (99% AUC, 60-80 FPS)

Balanced:
├── SimpleNet (99% AUC, 100+ FPS) ⭐ Best speed/accuracy
├── STFPM (97% AUC, 40-60 FPS)
├── MemSeg (97% AUC, 20-40 FPS) - Memory-guided
└── DifferNet (97% AUC, 20-40 FPS)

Fast, Good Accuracy:
├── CFlow-AD (98% AUC, 50-70 FPS)
├── CutPaste (96% AUC, 50+ FPS)
└── RIAD (96% AUC, 30-50 FPS) - Self-supervised

Special:
├── WinCLIP (95% AUC, 5-10 FPS) - Zero-shot capable
└── DevNet (95% AUC, 40-60 FPS) - Weakly-supervised
```

### When to Use Each Algorithm

**For maximum accuracy**:
→ PatchCore, FastFlow, SPADE

**For real-time (>50 FPS)**:
→ SimpleNet, CutPaste, FastFlow

**For best localization**:
→ SPADE, PatchCore, STFPM

**For zero-shot/few-shot**:
→ WinCLIP

**For self-supervised learning**:
→ CutPaste, RIAD, DRAEM

**For pixel-level localization**:
→ SPADE, PatchCore, STFPM, DRAEM

**For weakly-supervised (with few labels)**:
→ DevNet

**For memory-based approaches**:
→ MemSeg, PatchCore

**For research/education**:
→ DifferNet, CFlow-AD, FastFlow, RIAD

## 🔬 Algorithm Deep Dive

### CutPaste: Self-Supervised Learning

**How it works**:
1. Takes normal images
2. Cuts random rectangular patches
3. Pastes them at random locations (optionally rotated)
4. Trains classifier to distinguish original vs augmented
5. At test time, uses learned features for anomaly detection

**Variations**:
- **Normal CutPaste**: Regular rectangular patches
- **Scar CutPaste**: Thin elongated patches (for scratch-like defects)
- **3-way CutPaste**: Three-class classification (normal, normal cutpaste, scar cutpaste)

**Pros**:
- Simple and interpretable
- No anomaly data needed
- Fast training
- Good for texture anomalies

**Cons**:
- May not capture all anomaly types
- Depends on augmentation quality
- Limited localization

### WinCLIP: Zero-Shot Detection

**How it works**:
1. Uses pre-trained CLIP model
2. Defines text prompts for "normal" and "anomaly"
3. Extracts image and text features
4. Compares similarity using sliding windows
5. Anomaly score based on relative similarity

**Variations**:
- **Zero-shot**: Uses text prompts only
- **Few-shot**: Learns from k normal examples
- **Multi-scale**: Applies at different resolutions

**Pros**:
- No training required (zero-shot)
- Works with minimal data (few-shot)
- Flexible text-based control
- Strong localization

**Cons**:
- Requires CLIP installation
- Slower inference
- Depends on text prompt quality
- May not work for all defect types

### DifferNet: Learnable Differences

**How it works**:
1. Builds memory bank of normal features
2. For test image, finds k-nearest neighbors
3. Learns a difference module to compare features
4. Computes anomaly score from learned differences

**Key Components**:
- **Feature Extractor**: Pre-trained backbone (ResNet, Wide ResNet)
- **Memory Bank**: Stored normal features with k-D tree
- **Difference Module**: Learnable CNN for feature comparison
- **Multi-scale**: Uses multiple feature layers

**Pros**:
- Learns meaningful differences
- Good localization
- Flexible backbone
- Combines k-NN with deep learning

**Cons**:
- Requires training difference module
- Memory intensive (stores features)
- Slower than simple methods

### SPADE: Deep Pyramid Correspondences

**How it works**:
1. Extracts multi-scale features from normal images using pre-trained backbone
2. Builds k-D tree index for each feature level
3. For test image, extracts features at same scales
4. Finds k-nearest neighbors at each level
5. Computes distance-based anomaly map
6. Applies Gaussian smoothing for final map

**Key Components**:
- **Multi-scale Features**: Uses multiple layers (e.g., layer1, layer2, layer3)
- **k-NN Search**: Fast k-D tree for efficient nearest neighbor lookup
- **Alignment**: Optional feature alignment for better matching
- **Gaussian Smoothing**: Post-processing for smooth anomaly maps

**Pros**:
- Excellent pixel-level localization
- Multi-scale captures various defect sizes
- Fast inference with k-D tree
- No training required (only feature extraction)
- Memory efficient

**Cons**:
- Requires substantial normal data for good coverage
- Memory usage scales with normal set size
- Depends on pre-trained backbone quality

### RIAD: Reconstruction by Inpainting

**How it works**:
1. Decomposes normal images into grid-based masked regions
2. Trains U-Net to reconstruct masked areas from context
3. For test images, masks regions and reconstructs
4. Compares reconstruction with original
5. High reconstruction error indicates anomaly

**Key Components**:
- **Image Decomposer**: Creates grid masks for training
- **U-Net**: Encoder-decoder for reconstruction
- **Self-supervised Loss**: MSE between reconstruction and original
- **Multi-position**: Masks different positions for robustness

**Pros**:
- Self-supervised (no anomaly data needed)
- Simple and interpretable
- Good for texture/surface anomalies
- Effective reconstruction-based approach

**Cons**:
- May reconstruct anomalies if they're simple
- Limited to texture-based defects
- Requires training time
- Grid size affects performance

### MemSeg: Memory-Guided Segmentation

**How it works**:
1. Extracts features from normal images
2. Stores prototypical patterns in memory bank
3. For test image, queries k-nearest memories
4. Uses attention to weight relevant memories
5. Produces segmentation map highlighting anomalies

**Key Components**:
- **Memory Bank**: Stores normal feature prototypes
- **Feature Extractor**: Backbone for feature extraction
- **Attention Mechanism**: Learns to query relevant memories
- **Segmentation Head**: Produces pixel-level predictions

**Pros**:
- Memory-guided provides interpretability
- Good segmentation capability
- Attention-based querying
- Flexible and adaptable

**Cons**:
- Memory size affects performance
- Requires careful memory initialization
- More complex than simpler methods
- Training time for attention mechanism

### DevNet: Deviation Networks

**How it works**:
1. Takes normal AND few anomaly samples with labels
2. Trains network to predict anomaly scores
3. Uses deviation loss: minimize normal scores, maximize anomaly deviation
4. Normal samples should have low scores
5. Anomaly samples should deviate from normal by a margin

**Key Components**:
- **Feature Extractor**: Pre-trained backbone (frozen)
- **Scoring Network**: MLP that outputs anomaly score
- **Deviation Loss**: Custom loss encouraging deviation
- **Margin**: Hyperparameter controlling separation

**Pros**:
- Works with few anomaly labels
- Better than unsupervised when labels available
- Good generalization
- Flexible architecture

**Cons**:
- Requires labeled anomaly samples (weakly-supervised)
- Not purely unsupervised
- Performance depends on label quality
- May overfit to labeled anomalies

## 📚 Usage Examples

### Complete Workflow with CutPaste

```python
import numpy as np
from pyimgano.models import create_model
from sklearn.metrics import roc_auc_score

# 1. Load normal training data
normal_images = load_normal_images("train/good/")  # (N, H, W, 3)

# 2. Create CutPaste detector
detector = create_model(
    "cutpaste",
    backbone="resnet18",
    augment_type="3way",  # Use 3-way classification
    pretrained=True,
    epochs=256,
    batch_size=96,
    learning_rate=0.03,
)

# 3. Train
detector.fit(normal_images)

# 4. Test
test_images = load_test_images("test/")
test_labels = load_test_labels("test/")

# 5. Predict
scores = detector.predict_proba(test_images)
predictions = detector.predict(test_images)

# 6. Evaluate
auc = roc_auc_score(test_labels, scores)
print(f"AUC-ROC: {auc:.4f}")
```

### Zero-Shot Detection with WinCLIP

```python
from pyimgano.models import create_model

# 1. Create WinCLIP detector (no training needed!)
detector = create_model(
    "winclip",
    clip_model="ViT-B/32",
    window_size=224,
    k_shot=0  # Zero-shot
)

# 2. Set class name for text prompts
detector.set_class_name("screw")

# 3. Predict directly (no training!)
scores = detector.predict_proba(test_images)

# 4. Get pixel-level anomaly maps
anomaly_maps = detector.predict_anomaly_map(test_images)

# 5. Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(test_images[0])
plt.title("Original")

plt.subplot(132)
plt.imshow(anomaly_maps[0], cmap='hot')
plt.title("Anomaly Map")

plt.subplot(133)
overlay = test_images[0] * 0.5 + anomaly_maps[0][:,:,None] * 0.5
plt.imshow(overlay.astype(np.uint8))
plt.title("Overlay")
plt.show()
```

### Multi-Algorithm Ensemble

```python
from pyimgano.models import create_model
import numpy as np

# Create multiple detectors
detectors = {
    "cutpaste": create_model("cutpaste", backbone="resnet18"),
    "differnet": create_model("differnet", backbone="resnet18"),
    "simplenet": create_model("simplenet"),
}

# Train all
for name, detector in detectors.items():
    print(f"Training {name}...")
    detector.fit(normal_images)

# Ensemble prediction
all_scores = []
for name, detector in detectors.items():
    scores = detector.predict_proba(test_images)
    all_scores.append(scores)

# Average ensemble
ensemble_scores = np.mean(all_scores, axis=0)

# Weighted ensemble (if you know which works better)
weights = [0.4, 0.3, 0.3]  # CutPaste, DifferNet, SimpleNet
weighted_scores = np.average(all_scores, axis=0, weights=weights)
```

## 🎯 Best Practices

### For Production Deployment

1. **Choose the right algorithm**:
   - Real-time: SimpleNet, FastFlow
   - Maximum accuracy: PatchCore, FastFlow
   - Limited data: WinCLIP (zero/few-shot), CutPaste

2. **Optimize for your use case**:
   - Adjust image resolution (smaller = faster)
   - Use appropriate backbone (ResNet18 vs ResNet50)
   - Enable GPU for deep learning methods

3. **Validate thoroughly**:
   - Test on representative data
   - Check edge cases
   - Measure actual throughput

### For Research

1. **Benchmark properly**:
   - Use standard datasets (MVTec AD, BTAD)
   - Report multiple metrics (AUC-ROC image, AUC-ROC pixel, F1)
   - Include timing and memory usage

2. **Compare fairly**:
   - Use same data preprocessing
   - Same evaluation protocol
   - Report confidence intervals

3. **Ablation studies**:
   - Test different backbones
   - Vary hyperparameters
   - Compare components

## 📖 References

### Papers

1. **WinCLIP**: Jeong et al. "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.

2. **SimpleNet**: Liu et al. "SimpleNet: A Simple Network for Image Anomaly Detection and Localization." CVPR 2023.

3. **DifferNet**: Rudolph et al. "Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows." WACV 2021.

4. **CutPaste**: Li et al. "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization." CVPR 2021.

5. **PatchCore**: Roth et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022.

6. **STFPM**: Wang et al. "Student-Teacher Feature Pyramid Matching for Anomaly Detection." BMVC 2021.

7. **DRAEM**: Zavrtanik et al. "DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection." ICCV 2021.

8. **FastFlow**: Yu et al. "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows." AAAI 2022.

9. **CFlow-AD**: Gudovskiy et al. "CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows." WACV 2022.

10. **SPADE**: Cohen & Hoshen. "Sub-Image Anomaly Detection with Deep Pyramid Correspondences." ECCV 2020.

11. **RIAD**: Zavrtanik et al. "Reconstruction by Inpainting for Visual Anomaly Detection." Pattern Recognition 2021.

12. **MemSeg**: Yang et al. "Memorizing Normality to Detect Anomaly: Memory-augmented Deep Autoencoder for Unsupervised Anomaly Detection." ICCV 2021.

13. **DevNet**: Pang et al. "Deep Anomaly Detection with Deviation Networks." KDD 2019.

### Datasets

- **MVTec AD**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **BTAD**: https://avires.dimi.uniud.it/papers/btad/btad.zip
- **VisA**: https://github.com/amazon-science/spot-diff

## 🚀 Future Additions

We plan to add more SOTA algorithms:

- **RegAD** (CVPR 2024): Registration-based anomaly detection
- **UniAD** (NeurIPS 2022): Unified anomaly detection framework
- **PyramidFlow**: Multi-scale flow models
- **APRIL-GAN**: Adversarial prior based anomaly detection
- **ReverseDistillation++**: Enhanced reverse distillation

Stay tuned for updates!

## 💬 Contributing

Have a SOTA algorithm you'd like to see in PyImgAno?

- Open an issue on GitHub
- Submit a pull request
- Check our [Contributing Guide](../CONTRIBUTING.md)

We welcome implementations of new algorithms!
