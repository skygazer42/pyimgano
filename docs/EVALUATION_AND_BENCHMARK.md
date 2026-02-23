## Evaluation and Benchmarking Guide

This guide covers PyImgAno's evaluation metrics and benchmarking tools for comprehensive performance analysis.

---

## 📊 Evaluation Metrics

PyImgAno provides a complete suite of evaluation metrics for anomaly detection.

### Quick Start

```python
from pyimgano import evaluate_detector, models

# Train detector
detector = models.create_model('vision_ecod')
detector.fit(train_images)

# Get anomaly scores
scores = detector.decision_function(test_images)

# Evaluate
results = evaluate_detector(test_labels, scores)

# Print results
from pyimgano import print_evaluation_summary
print_evaluation_summary(results)
```

---

## 🎯 Available Metrics

### Image-Level Metrics

#### AUROC (Area Under ROC Curve)
```python
from pyimgano import compute_auroc

auroc = compute_auroc(y_true, y_scores)
print(f"AUROC: {auroc:.4f}")
```

**Interpretation:**
- 1.0 = Perfect separation
- 0.5 = Random classifier
- < 0.5 = Worse than random

#### Average Precision (AP)
```python
from pyimgano import compute_average_precision

ap = compute_average_precision(y_true, y_scores)
print(f"AP: {ap:.4f}")
```

**When to use:** Imbalanced datasets (few anomalies)

#### Classification Metrics
```python
from pyimgano import compute_classification_metrics

metrics = compute_classification_metrics(y_true, y_pred)
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### Optimal Threshold Detection

```python
from pyimgano import find_optimal_threshold

# Optimize for F1 score
threshold, f1 = find_optimal_threshold(y_true, y_scores, metric='f1')

# Optimize for Youden's J statistic
threshold, j = find_optimal_threshold(y_true, y_scores, metric='youden')
```

---

## 🔬 Comprehensive Evaluation

```python
from pyimgano import evaluate_detector

results = evaluate_detector(
    y_true=test_labels,
    y_scores=test_scores,
    threshold=None,  # Auto-find optimal
    find_best_threshold=True
)

# Results include:
# - AUROC
# - Average Precision
# - Optimal threshold
# - Precision, Recall, F1
# - Confusion matrix
print(f"AUROC: {results['auroc']:.4f}")
print(f"F1: {results['metrics']['f1']:.4f}")
```

---

## 📈 Benchmarking

Compare multiple algorithms systematically.

## 🏭 One-Click Industrial Benchmark (CLI + Run Artifacts) ⭐ NEW

`pyimgano-benchmark` supports an “industrial one-click” mode that:

- runs a **single category** or **all categories** (`--category all`)
- calibrates a **score threshold from train/normal scores** (industrial default)
- writes a stable run layout on disk (JSON report + per-image JSONL)

### Single Category (writes run artifacts by default)

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --device cuda
```

### All Categories (aggregates mean/std)

```bash
pyimgano-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category all \
  --model vision_patchcore \
  --preset industrial-balanced \
  --device cuda
```

### Custom Dataset

Expected structure:

```
my_dataset/
  train/normal/*.png
  test/normal/*.png
  test/anomaly/*.png
  # optional masks:
  ground_truth/anomaly/*_mask.png
```

Run:

```bash
pyimgano-benchmark \
  --dataset custom \
  --root /path/to/my_dataset \
  --model vision_ecod \
  --device cpu
```

### Output Layout

By default, runs are written under `runs/`:

```
runs/<ts>_<dataset>_<model>/
  report.json
  config.json
  categories/
    <category>/report.json
    <category>/per_image.jsonl
```

### Useful Flags

- `--output-dir /path/to/run_dir`: place artifacts in a specific directory
- `--no-save-run`: disable artifact writing (stdout JSON only)
- `--no-per-image-jsonl`: keep `report.json` only (skip per-image records)
- `--resize H W`: resize dataset images/masks during loading (default `256 256`)
- `--calibration-quantile 0.995`: override score threshold quantile
- `--limit-train N`, `--limit-test N`: quick smoke runs

### Basic Benchmark

```python
from pyimgano import AlgorithmBenchmark

# Define algorithms
algorithms = {
    'ECOD': {'model_name': 'vision_ecod', 'contamination': 0.1},
    'PatchCore': {'model_name': 'vision_patchcore', 'device': 'cuda'},
    'SimpleNet': {'model_name': 'vision_simplenet', 'epochs': 10},
}

# Run benchmark
benchmark = AlgorithmBenchmark(algorithms)
results = benchmark.run(
    train_images=train_paths,
    test_images=test_paths,
    test_labels=test_labels
)

# Print summary
benchmark.print_summary()
```

### Quick Benchmark

For fast comparisons with default settings:

```python
from pyimgano import quick_benchmark

results = quick_benchmark(
    train_images=train_paths,
    test_images=test_paths,
    test_labels=test_labels,
    algorithms=['ECOD', 'COPOD', 'PatchCore']
)
```

### Benchmark Output

```
================================================================================
BENCHMARK SUMMARY
================================================================================

Performance Metrics:
--------------------------------------------------------------------------------
Algorithm               AUROC       AP       F1  Precision   Recall
--------------------------------------------------------------------------------
ECOD                   0.9523   0.9145   0.9032     0.8750   0.9333
PatchCore              0.9956   0.9923   0.9800     0.9767   0.9833
SimpleNet              0.9912   0.9876   0.9756     0.9650   0.9863

Timing Results:
--------------------------------------------------------------------------------
Algorithm            Train (s)  Inference (s)  Per Image (ms)
--------------------------------------------------------------------------------
ECOD                      2.34          0.15            75.0
PatchCore                 0.00          0.85           425.0
SimpleNet                12.50          0.12            60.0

Best Performers:
--------------------------------------------------------------------------------
Best AUROC:          PatchCore (0.9956)
Best F1:             PatchCore (0.9800)
Fastest Training:    PatchCore (0.00s)
Fastest Inference:   SimpleNet (60.0ms/image)
```

### Save Results

```python
# Save to JSON
benchmark.save_results('benchmark_results.json')

# Get rankings
auroc_rankings = benchmark.get_rankings('auroc')
speed_rankings = benchmark.get_rankings('inference_per_image')
```

---

## 🎨 Visualization

### Anomaly Heatmap

```python
from pyimgano.visualization import plot_anomaly_map

detector = models.create_model('vision_patchcore')
detector.fit(train_images)

anomaly_map = detector.get_anomaly_map('test_image.jpg')

plot_anomaly_map(
    'test_image.jpg',
    anomaly_map,
    threshold=0.5,
    save_path='anomaly_viz.png'
)
```

### ROC Curve

```python
from pyimgano.visualization import plot_roc_curve

plot_roc_curve(
    y_true=test_labels,
    y_scores=test_scores,
    label='ECOD',
    save_path='roc_curve.png'
)
```

### Compare Multiple Detectors

```python
from pyimgano.visualization import compare_detectors

scores_dict = {
    'ECOD': ecod_scores,
    'PatchCore': patchcore_scores,
    'SimpleNet': simplenet_scores,
}

compare_detectors(
    y_true=test_labels,
    scores_dict=scores_dict,
    save_path='comparison.png'
)
```

### Score Distribution

```python
from pyimgano.visualization import plot_score_distribution

normal_scores = detector.decision_function(normal_images)
anomaly_scores = detector.decision_function(anomaly_images)

plot_score_distribution(
    normal_scores,
    anomaly_scores,
    save_path='score_dist.png'
)
```

---

## 📊 Complete Example

```python
from pyimgano import (
    AlgorithmBenchmark,
    evaluate_detector,
    models,
)
from pyimgano.visualization import compare_detectors, plot_anomaly_map

# 1. Define algorithms to compare
algorithms = {
    'Classical ML': {
        'ECOD': {'model_name': 'vision_ecod'},
        'COPOD': {'model_name': 'vision_copod'},
    },
    'Deep Learning': {
        'PatchCore': {
            'model_name': 'vision_patchcore',
            'coreset_sampling_ratio': 0.1
        },
        'SimpleNet': {
            'model_name': 'vision_simplenet',
            'epochs': 10
        },
    }
}

# 2. Run comprehensive benchmark
all_algorithms = {**algorithms['Classical ML'], **algorithms['Deep Learning']}
benchmark = AlgorithmBenchmark(all_algorithms)

results = benchmark.run(
    train_images=train_images,
    test_images=test_images,
    test_labels=test_labels,
    verbose=True
)

# 3. Print summary
benchmark.print_summary()

# 4. Detailed evaluation of best performer
best_algo = benchmark.get_rankings('auroc')[0][0]
print(f"\n{'=' * 60}")
print(f"Detailed Analysis: {best_algo}")
print(f"{'=' * 60}\n")

best_detector = models.create_model(all_algorithms[best_algo]['model_name'])
best_detector.fit(train_images)
best_scores = best_detector.decision_function(test_images)

from pyimgano import print_evaluation_summary
evaluation = evaluate_detector(test_labels, best_scores)
print_evaluation_summary(evaluation)

# 5. Visualizations
# ROC comparison
scores_dict = {}
for name, cfg in all_algorithms.items():
    det = models.create_model(cfg['model_name'])
    det.fit(train_images)
    scores_dict[name] = det.decision_function(test_images)
compare_detectors(test_labels, scores_dict, save_path='comparison.png')

# Anomaly map (for models that support it)
if hasattr(best_detector, 'get_anomaly_map'):
    anomaly_map = best_detector.get_anomaly_map(test_images[0])
    plot_anomaly_map(test_images[0], anomaly_map, save_path='heatmap.png')

print("\n✓ Evaluation complete!")
```

---

## 🎯 Best Practices

### 1. Choose Right Metrics

**AUROC** - Overall performance, threshold-independent
```python
auroc = compute_auroc(y_true, y_scores)
```

**F1-Score** - Balance between precision and recall
```python
threshold, f1 = find_optimal_threshold(y_true, y_scores, metric='f1')
```

**Average Precision** - Better for imbalanced datasets
```python
ap = compute_average_precision(y_true, y_scores)
```

### 2. Validate on Held-out Set

```python
from sklearn.model_selection import train_test_split

# Split normal training data
train_imgs, val_imgs = train_test_split(normal_images, test_size=0.2)

# Train on subset
detector.fit(train_imgs)

# Validate
val_scores = detector.decision_function(val_imgs + anomaly_images)
val_labels = [0] * len(val_imgs) + [1] * len(anomaly_images)

results = evaluate_detector(val_labels, val_scores)
print(f"Validation AUROC: {results['auroc']:.4f}")
```

### 3. Cross-Validation for Small Datasets

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
aurocs = []

for train_idx, val_idx in kfold.split(normal_images):
    train_fold = [normal_images[i] for i in train_idx]
    val_fold = [normal_images[i] for i in val_idx]

    detector = models.create_model('vision_ecod')
    detector.fit(train_fold)

    scores = detector.decision_function(val_fold + anomaly_images)
    labels = [0] * len(val_fold) + [1] * len(anomaly_images)

    auroc = compute_auroc(labels, scores)
    aurocs.append(auroc)

print(f"Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
```

### 4. Statistical Significance Testing

```python
from scipy import stats

# Compare two detectors
scores_a = detector_a.decision_function(test_images)
scores_b = detector_b.decision_function(test_images)

auroc_a = compute_auroc(test_labels, scores_a)
auroc_b = compute_auroc(test_labels, scores_b)

# Bootstrap confidence intervals
n_bootstrap = 1000
aurocs_a_boot = []
aurocs_b_boot = []

for _ in range(n_bootstrap):
    indices = np.random.choice(len(test_labels), len(test_labels), replace=True)
    aurocs_a_boot.append(compute_auroc(test_labels[indices], scores_a[indices]))
    aurocs_b_boot.append(compute_auroc(test_labels[indices], scores_b[indices]))

# Compare
print(f"Detector A: {auroc_a:.4f} (95% CI: {np.percentile(aurocs_a_boot, [2.5, 97.5])})")
print(f"Detector B: {auroc_b:.4f} (95% CI: {np.percentile(aurocs_b_boot, [2.5, 97.5])})")
```

---

## 📚 References

- **AUROC**: Fawcett, T. (2006). "An introduction to ROC analysis." Pattern Recognition Letters.
- **Average Precision**: Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves." ICML.
- **PRO Score**: Bergmann, P., et al. (2020). "The MVTec Anomaly Detection Dataset." IJCV.

---

## 🆘 Troubleshooting

### Issue: AUROC is NaN

**Cause**: Only one class in test set

**Solution**:
```python
if len(np.unique(test_labels)) < 2:
    print("Warning: Need both normal and anomaly samples for AUROC")
```

### Issue: Low performance on all metrics

**Possible causes:**
1. Insufficient training data
2. Train/test distribution mismatch
3. Wrong algorithm for data type

**Solution**: Try different algorithms and check data quality

### Issue: Benchmark runs too slowly

**Solution**: Use quick_benchmark with fast algorithms
```python
results = quick_benchmark(
    train_images, test_images, test_labels,
    algorithms=['ECOD', 'COPOD']  # Fast algorithms only
)
```

---

For more examples, see:
- [examples/quick_start.py](../examples/quick_start.py)
- [examples/benchmark_example.py](../examples/benchmark_example.py)
- [examples/visualization_example.py](../examples/visualization_example.py)
