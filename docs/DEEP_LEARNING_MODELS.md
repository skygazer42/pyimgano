# Deep Learning Models

PyImgAno exposes native models, compact variants, experimental proxies, and
external checkpoint adapters through one registry. Inspect fidelity before
using a paper name in reports:

```python
from pyimgano.models import model_info

metadata = model_info("vision_patchcore")["metadata"]
print(metadata["paper_fidelity"])
print(metadata["implementation_status"])
```

The generated [Model Index](MODEL_INDEX.md) lists every registered name. The
[Neural Model Fidelity](SOTA_ALGORITHMS.md) guide explains the classification.

## Recommended native implementations

| Model | Fidelity | Main use |
|---|---|---|
| `vision_patchcore` | `core-aligned` | Patch memory and localization |
| `vision_padim` | `core-aligned` | Per-location Gaussian modeling |
| `vision_stfpm` | `core-aligned` | Student/teacher feature pyramids |
| `vision_differnet` | `paper-adaptation` | Flow detector with a reduced transform ensemble |
| `vision_reverse_distillation` | `paper-adaptation` | ResNet-18 replacement for the paper's WRN50/OCBE path |
| `vision_draem` | `paper-adaptation` | Paper networks/schedule; simplified fallback synthesis unless DTD images are supplied |
| `vision_simplenet` | `paper-adaptation` | Compact feature projection/training recipe |
| `vision_spade` | `core-aligned` | Image retrieval and deep correspondences |
| `vision_cutpaste` | `core-aligned` | CutPaste self-supervision |

`pretrained=False` is the offline-safe default for most native vision models.
That avoids implicit downloads but does not provide the pretrained features
used by the papers. Enable or inject audited weights explicitly for meaningful
experiments.

## Minimal workflow

```python
from pyimgano.models import create_model

detector = create_model(
    "vision_patchcore",
    pretrained=True,
    coreset_sampling_ratio=0.1,
    device="cuda",
)
detector.fit(normal_image_paths)

scores = detector.decision_function(test_image_paths)
maps = detector.predict_anomaly_map(test_image_paths)
```

Published numbers additionally depend on weights, dataset split,
preprocessing, training schedule, score normalization, and metric protocol.
`core-aligned` is an algorithm-structure statement, not a benchmark claim.

## When the native entry is partial or experimental

- `vision_fastflow`, `vision_cflow`, `vision_dfm`, `vision_fcdd`, and
  `vision_softpatch` are compact/partial variants.
- `vision_draem` uses the paper networks and schedule, but its fallback anomaly
  synthesis is not the DTD protocol unless anomaly-source images are supplied.
- `vision_reverse_distillation` and `vision_simplenet` retain their papers' main
  idea but use materially different native architectures or training pipelines.
  Prefer an external backend where exact upstream checkpoint compatibility is
  required.
- `vision_ast`, `vision_promptad`, `vision_realnet`, `vision_inctrl`,
  `vision_glad`, `vision_oneformore`, `vision_panda`, `vision_regad`,
  `vision_riad`, and `vision_winclip` are experimental proxies.
- Family adapters such as `vision_aaclip`, `vision_univad`, and
  `vision_visionad` expose injectable scoring hooks; they are not native paper
  reproductions.

For supported upstream implementations, prefer model-specific anomalib entries
such as `vision_fastflow_anomalib`, `vision_cflow_anomalib`, and
`vision_winclip_anomalib`. These require an upstream-trained checkpoint.

## Detector contract

A detector must provide:

- `fit(X, y=None) -> self`
- `decision_function(X) -> (N,)`, with higher values meaning more anomalous

Pixel-localization models should also provide `get_anomaly_map(image) -> (H,W)`
or `predict_anomaly_map(X) -> (N,H,W)`.

Deep registry entries must declare their paper relationship:

```python
import numpy as np

from pyimgano.models.registry import register_model


@register_model(
    "my_detector",
    tags=("vision", "deep"),
    metadata={
        "description": "My generic reconstruction baseline",
        "paper_fidelity": "not-applicable",
        "implementation_status": "generic-reconstruction-baseline",
    },
)
class MyDetector:
    def __init__(self, *, contamination: float = 0.1):
        self.contamination = float(contamination)

    def fit(self, x, y=None):
        del y
        scores = self.decision_function(x)
        self.threshold_ = float(np.quantile(scores, 1.0 - self.contamination))
        return self

    def decision_function(self, x):
        return np.zeros(len(list(x)), dtype=np.float32)
```

If the implementation is only related to a paper, use `related_paper` with
`paper_fidelity="partial"` or `"inspired"`. Do not set `paper` or `sota`.

Supervision is derived conservatively: external checkpoint wrappers report
`backend-defined`; native normal-only deep detectors default to `one-class`
unless a more specific tag such as `self-supervised`, `weakly-supervised`, or
`zero-shot` is present.

## Optional dependency boundary

Keep large optional imports inside construction or runtime methods, provide an
actionable install hint, and never download weights implicitly. External
checkpoint wrappers should declare `paper_fidelity="external-backend"`,
`requires_checkpoint=True`, and a `weights_source`.
