# Recipes: Embeddings + Core

This doc contains small, practical recipes for the **embedding + classical core**
route (see `docs/INDUSTRIAL_EMBEDDING_PLUS_CORE.md`).

The intent is to provide copy-pastable starting points that:

- run offline by default (no implicit model downloads)
- are deterministic given a seed
- keep artifacts lightweight (no large bundled weights/assets)

## Recipe 1: Torchvision Embeddings + ECOD (Quick Baseline)

Python usage:

```python
from pyimgano.models.registry import create_model

det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchvision_backbone",
    embedding_kwargs={"backbone": "resnet18", "pretrained": False, "pool": "avg", "device": "cpu"},
    core_detector="core_ecod",
    core_kwargs={},
)

det.fit(train_paths)
scores = det.decision_function(test_paths)
labels = det.predict(test_paths)
```

Notes:
- Set `pretrained=True` only if you expect network access or have cached weights.
- For speed, reduce image sizes upstream (dataset loader resize, or pre-resize on disk).

## Recipe 2: Structural Features (No Torch) + IsolationForest

```python
from pyimgano.models.registry import create_model

det = create_model(
    "vision_feature_pipeline",
    contamination=0.1,
    feature_extractor={"name": "structural", "kwargs": {"max_size": 512}},
    core_detector="core_iforest",
    core_kwargs={"n_estimators": 200, "n_jobs": 1},
)

det.fit(train_paths)
scores = det.decision_function(test_paths)
```

## Recipe 3: Optional OpenCLIP Embeddings + kNN

This requires `open_clip_torch` (optional extra).

```python
from pyimgano.models.registry import create_model

det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="openclip_embed",
    # keep pretrained=None unless you explicitly want weights
    embedding_kwargs={"model_name": "ViT-B-32", "pretrained": None, "device": "cpu"},
    core_detector="core_knn",
    core_kwargs={"method": "largest", "n_neighbors": 5},
)
```

## Recipe 4: Synthetic Data → Embedding + Core (E2E)

Use `pyimgano-synthesize` to generate a tiny dataset + manifest, then run a
pipeline detector on it (see `tests/test_e2e_synth_embedding_core_pipeline.py`).

CLI:

```bash
pyimgano-synthesize \
  --in-dir /path/to/normal_images \
  --out-root ./out_synth_dataset \
  --category synthetic \
  --presets scratch stain tape \
  --blend alpha \
  --alpha 0.9 \
  --n-train 50 \
  --n-test-normal 10 \
  --n-test-anomaly 10 \
  --seed 0
```

Python:

```python
import numpy as np

from pyimgano.datasets.manifest import load_manifest_benchmark_split
from pyimgano.models import create_model
from pyimgano.synthesize_cli import synthesize_dataset

# 1) Synthesize a dataset + manifest.
synthesize_dataset(
    in_dir="/path/to/normal_images",
    out_root="./out_synth_dataset",
    category="synthetic",
    preset="scratch",
    presets=["scratch", "stain", "tape"],  # mixture sampling
    seed=0,
    n_train=50,
    n_test_normal=10,
    n_test_anomaly=10,
)

# 2) Load split from manifest.
split = load_manifest_benchmark_split(
    manifest_path="./out_synth_dataset/manifest.jsonl",
    root_fallback="./out_synth_dataset",
    category="synthetic",
    resize=(224, 224),
    load_masks=False,
)

# 3) Embedding + classical core pipeline.
det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchvision_backbone",
    embedding_kwargs={"backbone": "resnet18", "pretrained": False, "device": "cpu"},
    core_detector="core_ecod",
)

det.fit(split.train_paths)
scores = np.asarray(det.decision_function(split.test_paths), dtype=np.float64)
print(scores[:5])
```
