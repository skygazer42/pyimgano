# Tutorial: Embeddings Plus Classical Core

This tutorial shows the common industrial route:

`images -> embedding extractor -> core_* detector -> anomaly score`

It is a good fit when:

- you want deep features but a simple scoring layer
- you need offline-safe deployment options
- you want to swap detectors without retraining an end-to-end model

## 1. Choose The Embedding Layer

Common extractors:

- `torchvision_backbone`
- `torchvision_backbone_gem`
- `torchscript_embed`
- `onnx_embed`
- `openclip_embed` (optional extra)

If you are using Python classes directly, the registered
`torchvision_backbone` extractor is implemented by
`TorchvisionBackboneExtractor`.

See `docs/FEATURE_EXTRACTORS.md` for the full extractor list.

## 2. Start With `vision_embedding_core`

`vision_embedding_core` is the shortest path for this pattern.

```python
from pyimgano.models import create_model

det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchvision_backbone",
    embedding_kwargs={
        "backbone": "resnet18",
        "pretrained": False,
        "device": "cpu",
    },
    core_detector="core_knn",
    core_kwargs={"n_neighbors": 5},
)
```

Why this route works well:

- the feature stage stays explicit
- the scoring stage stays simple and replaceable
- `pretrained=False` keeps the default path offline-safe

## 3. Swap The Core Detector, Not The Whole Pipeline

Once embeddings are reasonable, try a few `core_*` scorers against the same
feature space:

- `core_knn` for local distance
- `core_lof` for local density mismatch
- `core_ecod` for a parameter-light baseline
- `core_mahalanobis` or `core_mcd` for Gaussian-ish embeddings
- `core_kde_ratio` for density-contrast style scoring

Example:

```python
det = create_model(
    "vision_embedding_core",
    contamination=0.1,
    embedding_extractor="torchscript_embed",
    embedding_kwargs={
        "checkpoint_path": "/models/embedder.ts",
        "device": "cpu",
    },
    core_detector="core_ecod",
    core_kwargs={},
)
```

## 4. Use `vision_feature_pipeline` For Full Control

When you want the general composition primitive, use
`vision_feature_pipeline` directly:

```python
from pyimgano.models import create_model

det = create_model(
    "vision_feature_pipeline",
    contamination=0.1,
    feature_extractor={
        "name": "torchvision_backbone",
        "kwargs": {"backbone": "resnet18", "pretrained": False, "device": "cpu"},
    },
    core_detector="core_knn",
    core_kwargs={"n_neighbors": 5},
)
```

This becomes especially useful when you want a JSON-friendly config for the
workbench or custom recipe layers.

## 5. Example Scripts In This Repo

Start from:

- `examples/embedding_plus_core_ecod.py`
- `examples/embedding_plus_core_mahalanobis_shrinkage.py`
- `examples/feature_pipeline_core_detectors.py`
- `examples/torchvision_embeddings_classical_demo.py`

These examples already cover the most common deployment-friendly composition
patterns.

## 6. Practical Guidance

Recommended order:

1. Get embeddings stable and offline-safe.
2. Start with `core_ecod` or `core_knn`.
3. If you need more robustness, try `core_mcd`, `core_lof`, or `core_oddoneout`.
4. If you later need localization, move to patch-based or pixel-map models rather
   than forcing an image-level core detector to do pixel work.

Operational notes:

- `vision_embedding_core` is image-level unless you deliberately choose a
  patch/pixel pipeline.
- Explicit checkpoint paths are preferred for deploy environments.
- For production, pair this route with exported `infer_config.json` or a deploy
  bundle from the workbench.

See also:

- `docs/TUTORIAL_CLASSICAL_ON_EMBEDDINGS.md`
- `docs/CORE_MODELS.md`
- `docs/RECIPES_EMBEDDINGS_PLUS_CORE.md`
