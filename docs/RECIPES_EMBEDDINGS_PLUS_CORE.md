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
    embedding_kwargs={
        "backbone": "resnet18",
        "pretrained": False,
        "pool": "avg",
        "device": "cpu",
        # Optional: when weights-provided transforms are unavailable, this is the
        # fallback resize used by the extractor.
        "image_size": 224,
        # Optional: disk cache keyed by (path + mtime + extractor config).
        # Speeds up repeated scoring runs.
        # "cache_dir": "./.cache/pyimgano_embeds",
    },
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

## Recipe 1b: TorchScript Embeddings + ECOD (Bring Your Own Checkpoint)

If you have a production embedding model exported to TorchScript (e.g. `model.pt`)
you can plug it into the same `vision_embedding_core` pipeline.

```python
from pyimgano.models.registry import create_model

det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchscript_embed",
    embedding_kwargs={
        # REQUIRED: a local TorchScript checkpoint path (no downloads).
        "checkpoint": "/abs/or/rel/path/to/model.pt",
        "device": "cpu",
        "batch_size": 16,
        # Preprocess defaults mirror torchvision ImageNet-style normalization.
        "image_size": 224,
        # Optional: disk cache keyed by (path + mtime + extractor config).
        # "cache_dir": "./.cache/pyimgano_torchscript_embeds",
    },
    core_detector="core_ecod",
)
```

### Pooling modes (`torchvision_backbone`)

`torchvision_backbone` supports multiple pooling modes via `embedding_kwargs["pool"]`:

- `avg` (default): global average pool (good general baseline).
- `max`: global max pool (can emphasize small sharp defects).
- `gem`: GeM pooling over a chosen conv node (strong baseline for embeddings in some setups).
  - Requires a conv backbone (e.g. ResNet). Configure `pool_node` (default: `layer4`),
    `gem_p` and `gem_eps` if needed.
- `cls`: CLS-token pooling for torchvision ViT backbones (e.g. `vit_b_16`).
  - For ViT, `image_size` must match the model’s configured size (usually `224`).

Example: GeM pooling on ResNet layer4

```python
embedding_kwargs={
    "backbone": "resnet18",
    "pretrained": False,
    "pool": "gem",
    "pool_node": "layer4",
    "gem_p": 3.0,
    "device": "cpu",
}
```

Example: CLS pooling on ViT

```python
embedding_kwargs={
    "backbone": "vit_b_16",
    "pretrained": False,
    "pool": "cls",
    "image_size": 224,
    "device": "cpu",
}
```

### Speed knobs (best-effort, safe defaults)

When running on CUDA, these options can help:

- `channels_last=True`: tries to use NHWC memory format.
- `amp=True`: best-effort autocast (CUDA only; silently falls back).
- `compile=True`: best-effort `torch.compile` (CUDA only; silently falls back).

### Disk caching

There are two caching layers you can use:

- **Extractor-native embedding cache**: set `embedding_kwargs["cache_dir"]` for `torchvision_backbone`.
- **CLI feature cache**: use `pyimgano-features --cache-dir ...` to cache extracted features
  for any extractor that consumes path inputs.

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
