# Deep Learning Models

PyImgAno exposes native models, compact variants, experimental proxies, and
external checkpoint adapters through one registry. Inspect fidelity before
using a paper name in reports:

```python
from pyimgano.models.registry import model_info

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
| `vision_softpatch` | `core-aligned` | Noisy-training patch denoising and weighted memory |
| `vision_padim` | `core-aligned` | Paper R18/WR50-2 patch distributions |
| `vision_stfpm` | `core-aligned` | Paper ResNet-18 feature-pyramid matching |
| `vision_cflow` | `core-aligned` | Multi-scale conditional normalizing flows |
| `vision_dfm` | `paper-adaptation` | PCA and full Gaussian likelihood on one deep layer |
| `core_deep_svdd` | `paper-adaptation` | Both paper objectives on a generic feature MLP |
| `vision_devnet` | `paper-adaptation` | Paper image network and detection path; localization is not exposed |
| `vision_differnet` | `paper-adaptation` | Paper detection path; gradient-map localization is not exposed |
| `vision_reverse_distillation` | `core-aligned` | Paper WRN50-2 teacher, OCBE, and reverse-WRN decoder |
| `vision_draem` | `paper-adaptation` | Paper networks/schedule; simplified fallback synthesis unless DTD images are supplied |
| `vision_simplenet` | `core-aligned` | Paper patch embedding, adapter, feature noise, and discriminator |
| `vision_spade` | `core-aligned` | WRN50-2 ImageNet-V1 retrieval and squared-L2 deep-pyramid correspondences |
| `vision_cutpaste` | `core-aligned` | ResNet-18 CutPaste 3-way learning and Gaussian-density scoring |
| `vision_efficientad` | `paper-adaptation` | Paper S/M PDNs, dual-head student, 64-D autoencoder, three losses, calibrated dual maps |

`pretrained=False` is the offline-safe default for most native vision models,
including SimpleNet. Reverse Distillation instead defaults to the paper's
pretrained teacher. Cache audited weights or explicitly opt in before benchmark
runs; random weights are only structurally valid.

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

## When the native entry is adapted or experimental

- `vision_softpatch` follows the paper's WideResNet50-2 `layer2`/`layer3`
  PatchCore embedding, position-wise LOF (`k=6`), 15% patch denoising, 10%
  greedy coreset, stored outlier weights, weighted nearest-neighbor score, max
  image reduction, and sigma-4 map smoothing. Set `pretrained=True` for the
  paper's ImageNet weights; the offline default preserves structure only.
- `vision_dfm` now implements the paper's Gaussian branch: one independently
  selected ResNet layer, factor-4 average pooling, 99.5%-variance PCA, MLE
  covariance in PCA space, and the full Gaussian negative log-likelihood. Its
  normal-only detector contract and offline-safe weight default remain local
  adaptations of the paper's supervised class-conditional experiments.
- `vision_cflow` follows the authors' ResNet path: frozen layer2--layer4
  features, sinusoidal 2-D conditions, one eight-block conditional flow per
  scale, normalized likelihood training, and summed multi-scale probability
  maps. Its offline default does not download ImageNet weights, and published
  MVTec results also require the authors' category-specific input sizes and
  benchmark protocol.
- `vision_fastflow` now follows the paper's ResNet18/WideResNet50-2 stages
  1--3, native feature widths, eight ActNorm/permutation/affine-coupling steps,
  3x3-only or alternating 3x3/1x1 subnets, 2-D likelihood objective, and
  multi-scale probability maps. It remains a paper adaptation because the
  paper does not publish all stabilization, probability-map normalization,
  image-reduction, or augmentation details and the local offline default does
  not download ImageNet weights.
- `vision_draem` uses the paper networks and schedule, but its fallback anomaly
  synthesis is not the DTD protocol unless anomaly-source images are supplied.
- `core_deep_svdd` and `vision_deep_svdd` implement the paper's one-class and
  soft-boundary objectives, bias-free final-linear encoder constraint, center
  initialization, and radius score. They use a generic feature MLP rather than
  the paper's dataset-specific LeNet CNNs and are therefore adaptations.
- `vision_devnet` follows the 2021 image paper's end-to-end ResNet-18, 1x1 patch
  scorer, two-scale top-10% MIL aggregation, Gaussian-reference loss, balanced
  batches, and optimizer defaults. It keeps an offline-safe weight default and
  does not expose the paper's smoothed input-gradient localization map.
- `vision_differnet` matches the paper's three AlexNet scales, eight two-sided
  affine coupling blocks, 2048-unit s/t networks, clamp, optimizer, and 4/64
  transform counts. It remains an adaptation because the paper's gradient-map
  localization path is not exposed and `pretrained=False` is the offline default.
- `vision_memae` uses the paper's CIFAR-10 RGB encoder/decoder, 500-slot memory,
  cosine addressing, hard shrinkage, and entropy objective. Its industrial-image
  detector contract is an adaptation of that CIFAR-10 experiment.
- `vision_fcdd` uses the paper's MVTec truncated VGG11-BN network, element-wise
  pseudo-Huber/HSC objective, confetti settings, SGD schedule, and receptive-field
  Gaussian transposed-convolution heatmap. Its offline default omits ImageNet
  weights and estimates category normalization bounds from the supplied normal
  images, so it remains a paper adaptation rather than a benchmark reproduction.
- `vision_ast` implements the paper's RGB MVTec AD path: frozen EfficientNet-B5
  layer-36 features, 32-channel positional conditioning, a four-block RealNVP
  teacher, a four-block residual convolutional student, two-stage normal-only
  training, and mean student-teacher distance scoring. It remains an adaptation
  because ImageNet weights are opt-in and the paper's 3D/foreground path is not
  part of this detector.
- `vision_riad` follows the paper's three-way disjoint region masking, five-level
  U-Net, assembled partial inpaintings, L2/SSIM/MSGMS objective, four region
  sizes, and MSGMS anomaly maps. It remains an adaptation because the authors
  did not publish reference code.
- `vision_efficientad` / `efficient_ad` implements the supplementary S/M PDN
  tables, 384/768-channel teacher/student path, 64-D autoencoder, three paper
  losses, 70,000-step Adam schedule, quantile-calibrated dual maps, and max
  image score. Strict fitting requires caller-supplied distilled teacher
  weights and ImageNet-style penalty data because the paper publishes neither
  an official checkpoint nor an author repository.
- `vision_panda` implements the paper's image-level PANDA-Early path:
  ImageNet-pretrained ResNet152, layer3/layer4 compactness fine-tuning for
  2,300 minibatches with the published SGD defaults, and summed squared-L2
  2-NN scoring. PANDA-EWC, PANDA-SES, and the separate SPADE segmentation path
  are not represented by this entry.
- `vision_realnet` implements the paper's WideResNet50-2 feature pyramid,
  one-time 64-batch AFS channel selection, four independent reconstruction
  U-Nets, max/mean RRS, joint reconstruction/segmentation losses, and published
  1,000-epoch defaults. Fitting strictly requires paired anomaly images and
  masks from the separate SDAS/SIA stage; it does not synthesize a different
  corruption or reproduce the authors' labeled validation checkpoint choice.
- `vision_promptad`, `vision_inctrl`, `vision_glad`,
  `vision_oneformore`, `vision_regad`, and `vision_winclip` are
  experimental proxies.
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
