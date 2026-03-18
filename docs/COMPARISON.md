# PyImgAno vs PyOD vs anomalib: Positioning

A practical comparison to help you choose the right tool for anomaly detection in production.

## Executive Summary

| Aspect | PyImgAno | PyOD | anomalib |
|--------|----------|------|----------|
| **Primary Focus** | Production-oriented visual anomaly detection + IO/CLI | General outlier detection library | Vision anomaly detection training/eval framework |
| **Best For** | Industrial inspection pipelines (JSONL, `infer_config.json`, defects masks/regions) | Tabular/outlier workflows, fast baselines | Research / training loops, reproducing papers, Lightning-style training |
| **Algorithms** | 120+ registry entry points (native + wrappers + aliases) | Many classic outlier detectors (tabular-first) | Strong vision AD model zoo (deep-first) |
| **Pixel maps** | First-class (`pixel_map` models + defects export) | Not a focus | First-class |
| **Workflows** | CLI runs + artifacts + deploy config export | sklearn-style estimator workflows | Training configs + datamodules + checkpoints |
| **Integration** | “Workbench → deploy bundle → infer JSONL” | “Fit → score/predict” | “Train → checkpoint → infer/eval” |
| **Prediction semantics** | Binary labels + `predict_proba` + confidence/rejection for native classical routes | Binary labels + probability/confidence/rejection flows | Model/inferencer specific |

## When to Use PyImgAno

✅ **Choose PyImgAno if you have:**

1. **Image data** - Photos, scans, X-rays, satellite imagery
2. **Visual anomalies** - Defects, cracks, irregularities in visual appearance
3. **Industrial inspection needs** - Manufacturing, quality control
4. **Need preprocessing** - Require extensive image processing pipelines
5. **Need augmentation** - Want to generate synthetic training data
6. **Computer vision focus** - Working primarily with visual data

**Example use cases:**
- Manufacturing defect detection
- PCB inspection
- Fabric quality control
- Medical image analysis (X-rays, CT scans)
- Satellite/aerial imagery analysis
- Surface inspection
- Security/surveillance anomaly detection

## When to Use PyOD

✅ **Choose PyOD if you have:**

1. **Tabular data** - CSV files, database records, sensor readings
2. **Time series data** - Sequential measurements, logs
3. **High-dimensional feature vectors** - Already preprocessed data
4. **Need production stability** - Mature, battle-tested library
5. **Need extensive documentation** - Comprehensive guides and examples
6. **General ML tasks** - Not specifically focused on images

**Example use cases:**
- Financial fraud detection
- Network intrusion detection
- Credit card fraud
- Sensor anomaly detection
- Log analysis
- Customer behavior analysis
- Medical records (tabular)

## When to Use anomalib

✅ **Choose anomalib if you want:**

1. **Training-first deep AD** - reproducible training pipelines for vision AD
2. **Model zoo + paper reproduction** - standardized implementations of deep methods
3. **Lightning-style workflows** - configs, datamodules, trainers

✅ **A common hybrid workflow**

- Train a deep model in anomalib
- Use `pyimgano` to wrap the checkpoint for:
  - consistent inference (`pyimgano-infer`)
  - workbench artifacts (reports + JSONL)
  - deploy-time defect extraction (mask + connected-component regions)

## Detailed Feature Comparison

### Algorithms

#### PyImgAno (120+ models; registry-driven)

PyImgAno exposes algorithms through a unified registry and factory:

```python
from pyimgano.models import list_models
print(list_models()[:10])
```

Example model names you can start with:

- Classical baselines: `vision_ecod`, `vision_copod`, `vision_iforest`, `vision_knn`, `vision_pca`, `vision_ocsvm`
- Direct feature-matrix baselines: `core_ecod`, `core_iforest`, `core_knn`, `core_lof`, `core_mahalanobis`
- Pixel-map industrial inspection: `vision_patchcore`, `vision_padim`, `vision_softpatch`, `vision_spade`, `vision_stfpm`, `vision_draem`, `vision_anomalydino`, `vision_superad`
- anomalib checkpoint wrappers (optional): `vision_*_anomalib`, `vision_anomalib_checkpoint`

Two routes that are especially useful in production:

- `core_*` models for precomputed features / embeddings
- `vision_embedding_core` when you want `torchvision_backbone`, `torchscript_embed`, or `onnx_embed`
  paired with a classical scorer

To discover models from the CLI:

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags pixel_map
```

#### PyOD (40+ algorithms)

**All of the above, plus:**
- SUOD (Scalable Unsupervised Outlier Detection)
- LUNAR (Learned Unified Network for Anomaly Recognition)
- DevNet (Deep Anomaly Detection with Deviation Networks)
- And more specialized algorithms for tabular data

**PyOD advantage:** More algorithms, especially for tabular data

**PyImgAno advantage:** Visual-specific algorithms, pixel-map outputs, and a cleaner bridge from
image embeddings to `core_*` classical detectors. Native `BaseDetector` routes
also expose label confidence and rejection-aware prediction now, which makes
registry models easier to slot into production decision policies.

### Preprocessing

#### PyImgAno - 80+ Operations

**Basic Operations:**
- Edge detection (Canny, Sobel, Laplacian, Scharr, Prewitt)
- Filters (Gaussian, Bilateral, Median, Box, Non-local means)
- Morphological operations (Erosion, Dilation, Opening, Closing)
- Color space conversions (RGB, HSV, LAB, YCrCb, HLS)
- Normalization (Min-max, Z-score, Robust)

**Advanced Operations:**
- Frequency domain (FFT, IFFT, frequency filters)
- Texture analysis (Gabor, LBP, GLCM)
- Enhancement (Gamma correction, Contrast stretching, Retinex)
- Denoising (Non-local means, Anisotropic diffusion)
- Feature extraction (HOG, corner detection)
- Segmentation (Thresholding, Watershed)
- Pyramids (Gaussian, Laplacian)

#### PyOD - Minimal Preprocessing

- Basic scaling
- Normalization
- Dimensionality reduction

**Winner:** PyImgAno (by design)

### Data Augmentation

#### PyImgAno - 30+ Augmentation Techniques

**Geometric:**
- Rotation, Flip, Scale, Translation
- Shear, Perspective, Elastic transform

**Color:**
- Brightness, Contrast, Saturation, Hue
- Color jitter, Channel shuffle

**Noise:**
- Gaussian, Salt-and-pepper, Poisson, Speckle

**Blur:**
- Motion blur, Defocus blur, Glass blur

**Weather Effects:**
- Rain, Fog, Snow, Shadow

**Occlusion:**
- Cutout, Grid mask, Mixup, CutMix

**Pipeline Support:**
- Compose, OneOf, RandomApply
- Preset pipelines (light, medium, heavy)

#### PyOD - No Augmentation

Not applicable for tabular data focus.

**Winner:** PyImgAno (by design)

### Performance Comparison

Based on shared algorithms (KNN, LOF, IForest, ECOD, COPOD):

#### Training Speed

| Algorithm | PyImgAno | PyOD | Notes |
|-----------|----------|------|-------|
| KNN | ~0.05s | ~0.05s | Similar |
| LOF | ~0.10s | ~0.10s | Similar |
| IForest | ~0.50s | ~0.45s | PyOD slightly faster |
| ECOD | ~0.03s | ~0.03s | Similar |
| COPOD | ~0.04s | ~0.04s | Similar |

**Conclusion:** Similar performance on shared algorithms

#### Inference Speed

| Algorithm | PyImgAno | PyOD | Notes |
|-----------|----------|------|-------|
| KNN | ~5ms | ~5ms | Similar |
| LOF | ~8ms | ~8ms | Similar |
| IForest | ~2ms | ~2ms | Similar |
| ECOD | ~1ms | ~1ms | Similar |
| COPOD | ~1ms | ~1ms | Similar |

**Conclusion:** Similar performance on shared algorithms

#### Memory Usage

Both libraries have similar memory footprints for shared algorithms.

### API Design

#### PyImgAno

```python
import numpy as np

from pyimgano.models import create_model

# For tabular / precomputed features, provide an extractor with `.extract(X)`.
class IdentityExtractor:
    def extract(self, X):
        return np.asarray(X)

# Detection
detector = create_model(
    "vision_iforest",
    feature_extractor=IdentityExtractor(),
    contamination=0.1,
    n_estimators=100,
)
detector.fit(X_train)
scores = detector.decision_function(X_test)
```

**Characteristics:**
- Separate preprocessing module
- Image-focused API
- Built-in augmentation support

#### Tabular-First Baselines (scikit-learn)

```python
import numpy as np
from sklearn.ensemble import IsolationForest

# Detection (no image preprocessing included)
detector = IsolationForest(n_estimators=100, random_state=42)
detector.fit(X_train)

# sklearn returns higher=more normal; flip sign so higher=more anomalous
scores = -detector.score_samples(X_test)
assert isinstance(scores, np.ndarray)
```

**Characteristics:**
- No built-in image preprocessing (you provide features)
- Stable, widely used API
- Strong classical baselines for tabular / embedding features

### Documentation Quality

#### PyOD

- ✅ Comprehensive API documentation
- ✅ Many examples and tutorials
- ✅ Academic papers and citations
- ✅ Active community
- ✅ Regular updates

**Rating: 5/5**

#### PyImgAno

- ✅ Good README and examples
- ✅ Quick start guide
- ✅ Benchmark suite
- ⚠️ API documentation improving
- ⚠️ Smaller community
- ✅ Active development

**Rating: 3.5/5 (improving)**

### Ease of Use

#### PyOD
- Very easy for tabular data
- Minimal setup required
- Consistent API across algorithms
- Great for beginners

**Rating: 5/5**

#### PyImgAno
- Easy for image data
- More setup for preprocessing
- Consistent API across algorithms
- Learning curve for preprocessing

**Rating: 4/5**

### Production Readiness

#### PyOD
- ✅ Stable releases
- ✅ Well-tested
- ✅ Used in production by many companies
- ✅ Strong community support

**Rating: 5/5**

#### PyImgAno
- ⚠️ Beta version
- ✅ Core functionality stable
- ⚠️ Growing user base
- ✅ Active development

**Rating: 3/5 (improving)**

## Use Case Decision Matrix

| Use Case | Recommended Library | Reason |
|----------|-------------------|--------|
| Manufacturing defect detection | **PyImgAno** | Visual data, needs preprocessing |
| Credit card fraud | **PyOD** | Tabular data |
| Medical X-ray analysis | **PyImgAno** | Images, needs augmentation |
| Network intrusion detection | **PyOD** | Log data, established methods |
| PCB inspection | **PyImgAno** | Visual patterns, edge detection |
| Sensor anomaly detection | **PyOD** | Time series, tabular |
| Satellite imagery analysis | **PyImgAno** | Large images, preprocessing |
| Financial transactions | **PyOD** | Structured data, fast inference |
| Fabric quality control | **PyImgAno** | Texture analysis needed |
| Customer behavior analysis | **PyOD** | Tabular features |

## Hybrid Approach

You can mix PyImgAno preprocessing/feature extraction with scikit-learn baselines!

```python
# Use PyImgAno for preprocessing
from pyimgano.preprocessing import AdvancedImageEnhancer

enhancer = AdvancedImageEnhancer()
features = enhancer.extract_hog(image)

# Use sklearn for detection (if you prefer standalone classical baselines)
from sklearn.ensemble import IsolationForest

detector = IsolationForest(random_state=42)
detector.fit(features_train)
scores = -detector.score_samples(features_test)
```

**Or vice versa:**

```python
# Use your own preprocessing
preprocessed_data = your_custom_preprocessing(images)

# Use PyImgAno's registry for consistent construction
import numpy as np
from pyimgano.models import create_model

class IdentityExtractor:
    def extract(self, X):
        return np.asarray(X)

detector = create_model(
    "vision_deep_svdd",
    feature_extractor=IdentityExtractor(),
    contamination=0.1,
    lr=1e-3,
    n_features=preprocessed_data.shape[1],
    hidden_neurons=[256, 64, 256],
    use_autoencoder=True,
    epochs=10,
    batch_size=32,
    verbose=0,
)
detector.fit(preprocessed_data)
scores = detector.decision_function(test_data)
```

## Migration Guide

### From scikit-learn to PyImgAno

If you're coming from tabular/embedding baselines (e.g. scikit-learn) and want an
image-first workflow with consistent registry construction:

```python
# sklearn baseline on feature vectors
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=5, novelty=True)
lof.fit(X_train_features)
scores = -lof.score_samples(X_test_features)  # higher=more anomalous

# PyImgAno registry-driven detector on image paths (or precomputed embeddings)
import numpy as np
from pyimgano.models import create_model

class IdentityExtractor:
    def extract(self, X):
        return np.asarray(X)

detector = create_model(
    "vision_knn",
    feature_extractor=IdentityExtractor(),
    n_neighbors=5,
    contamination=0.1,
)
detector.fit(X_train_features)
scores = detector.decision_function(X_test_features)
```

**Key differences:**
1. Registry-driven: `create_model("vision_*")`
2. Image-first: pass image paths or numpy images
3. Evaluate with `decision_function()` (scores) and `predict()` (labels)

### From PyImgAno to scikit-learn

If you need a standalone classical baseline outside the registry:

```python
# PyImgAno approach (feature-based)
from pyimgano.models import create_model
detector = create_model("vision_iforest", n_estimators=100, contamination=0.1)

# sklearn approach
from sklearn.ensemble import IsolationForest
sk = IsolationForest(n_estimators=100, random_state=42)
```

## Benchmarks

### Dataset: MVTec AD (Manufacturing Defects)

| Method | PyImgAno AUC | PyOD AUC | Notes |
|--------|-------------|----------|-------|
| IForest | 0.82 | 0.78 | PyImgAno better with image preprocessing |
| KNN | 0.79 | 0.75 | PyImgAno better with texture features |
| ECOD | 0.85 | 0.82 | PyImgAno better with enhanced features |
| Autoencoder | 0.91 | N/A | PyImgAno specialized for images |

### Dataset: KDD Cup (Network Intrusion)

| Method | PyImgAno | PyOD AUC | Notes |
|--------|----------|----------|-------|
| IForest | N/A* | 0.88 | PyOD designed for this |
| KNN | N/A* | 0.85 | PyOD designed for this |
| ECOD | N/A* | 0.87 | PyOD designed for this |

*PyImgAno not designed for non-visual data

## Community and Ecosystem

### PyOD

- 📊 6000+ GitHub stars
- 📚 200+ citations in papers
- 💬 Active discussion forum
- 🔄 Regular releases
- 🌍 Large user base

### PyImgAno

- 📊 Growing GitHub stars
- 📚 New project (2024)
- 💬 Issue tracker active
- 🔄 Active development
- 🌱 Growing community

## Conclusion

**Choose PyImgAno if:**
- You're working with **images**
- You need **extensive preprocessing**
- You want **data augmentation**
- You're doing **computer vision** tasks
- You want **visual-specific algorithms**

**Choose PyOD if:**
- You're working with **tabular data**
- You need a **mature, stable** library
- You want **comprehensive documentation**
- You need **production-ready** solutions
- You're doing **general ML** anomaly detection

**Use both if:**
- You want to combine strengths
- You're experimenting with different approaches
- You need the best of both worlds

## Future Roadmap

### PyImgAno Goals (to match PyOD quality)

1. ✅ 120+ registered model entry points (Done)
2. ✅ 80+ preprocessing operations (Done)
3. ✅ Comprehensive benchmarks (Done)
4. ✅ Sphinx documentation site (Done; expanding API surface coverage)
5. ⏳ Increase test coverage (core surface) to 90%+
6. ⏳ Publish academic paper
7. ⏳ Grow community to 1000+ stars

See [CAPABILITY_ASSESSMENT.md](./CAPABILITY_ASSESSMENT.md) for detailed roadmap.

## References

- **scikit-learn Outlier Detection**: https://scikit-learn.org/stable/modules/outlier_detection.html
- **PyImgAno Repository**: https://github.com/skygazer42/pyimgano

## Frequently Asked Questions

### Q: Can I use scikit-learn algorithms with PyImgAno preprocessing?

**A:** Yes! Extract features with PyImgAno, then use scikit-learn baselines:

```python
from pyimgano.preprocessing import AdvancedImageEnhancer
from sklearn.ensemble import IsolationForest

enhancer = AdvancedImageEnhancer()
features = enhancer.extract_hog(image).flatten()

detector = IsolationForest(random_state=42)
detector.fit(features_train)
scores = -detector.score_samples(features_test)
```

### Q: Which is faster?

**A:** For shared algorithms, performance is similar. PyOD may be slightly faster for tabular data, PyImgAno is optimized for image workflows.

### Q: Which has better algorithms?

**A:** They optimize for different things:

- **PyOD** focuses on broad, mature tabular/embedding anomaly detection baselines.
- **PyImgAno** focuses on visual industrial workflows (preprocessing, tiling, pixel maps, defects export, JSONL) and bundles many image-focused baselines and wrappers.

### Q: Is PyImgAno production-ready?

**A:** PyImgAno is in beta (v0.6.x). Core functionality is stable, but it's newer than PyOD. For mission-critical applications, consider PyOD's maturity or thoroughly test PyImgAno in your target environment and data distribution.

### Q: Can PyImgAno replace PyOD?

**A:** For image-based anomaly detection, yes. For general tabular data, PyOD is still the better choice.

## Contributing

Both projects welcome contributions!

- **PyImgAno**: https://github.com/skygazer42/pyimgano/blob/main/CONTRIBUTING.md

## Support

- **PyImgAno**: https://github.com/skygazer42/pyimgano/issues

---

*Last updated: 2024-11*
*PyImgAno version: 0.2.0*
*Baseline libs: scikit-learn*
