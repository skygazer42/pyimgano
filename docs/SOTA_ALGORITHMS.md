# Neural Model Fidelity

PyImgAno does not treat a paper name as proof of a paper reproduction. Every
registered deep model exposes two audit fields in `model_info(...)`:

- `paper_fidelity`: its relationship to a paper.
- `implementation_status`: what the shipped code actually is.

The complete, generated inventory is in [MODEL_INDEX.md](MODEL_INDEX.md).

## Fidelity levels

| Value | Meaning |
|---|---|
| `core-aligned` | The native implementation contains the paper's defining data flow, objective, and score construction. It is not a claim of identical benchmark numbers. |
| `paper-adaptation` | The defining objective is retained but adapted to a different input or deployment setting. |
| `partial` | A meaningful subset or compact variant is implemented; use `related_paper`, not `paper`. |
| `inspired` | Experimental compatibility proxy only; it must not be presented as the paper method. |
| `external-backend` | PyImgAno delegates inference to an upstream/checkpoint-backed implementation. |
| `not-applicable` | Generic baseline or pipeline with no paper-reproduction claim. |

## Native core-aligned implementations

- `vision_patchcore`
- `vision_padim`
- `vision_stfpm`
- `vision_reverse_distillation` / `vision_reverse_dist`
- `vision_simplenet`
- `vision_spade`
- `vision_cutpaste` / `cutpaste`
- `core_deep_svdd`

These entries implement the defining algorithmic core. Inspect each model's
weight default: offline-safe `pretrained=False` produces an experimentally weak
model, while Reverse Distillation follows the paper with pretrained weights by
default. Reproducing paper metrics additionally requires the paper's data split
and evaluation protocol.

### Audited native scope (2026-07-15)

| Entry | Verified native scope | Remaining boundary |
|---|---|---|
| `vision_patchcore` | WRN50-2 layer2/layer3 padded 3x3 patches, 1024-to-1024 MeanMapper/Aggregator embedding, 10% approximate greedy coreset, 1-NN max score; resize 256 then crop 224 | Paper experiments require ImageNet weights; offline default is `pretrained=False` |
| `vision_padim` | fixed random channel subset, per-location Gaussian, Mahalanobis map, Gaussian smoothing; resize 256 then crop 224 at default size | Paper experiments require ImageNet weights; offline default is `pretrained=False` |
| `vision_spade` | global KNN retrieval, retrieved-image pyramid gallery, dense correspondence, sigma-4 smoothing; resize 256 then crop 224 | Paper experiments require ImageNet WRN50x2 weights; offline default is `pretrained=False` |
| `vision_stfpm` | frozen teacher/random student, first three ResNet-18 blocks, normalized feature loss, multiplicative map, 80/20 validation checkpoint selection | Exact paper path requires `pretrained_teacher=True` |
| `vision_reverse_distillation` | ImageNet WideResNet50-2 teacher stages 1--3, exact released OCBE and reverse-WRN block counts/channels, cosine loss, additive cosine maps, sigma-4 smoothing; 256px, Adam 0.005, batch 16, 200 epochs | Published metrics still require the MVTec category protocol and matching ImageNet weights |
| `vision_simplenet` | WRN50-2 layer2/layer3 padded 3x3 neighborhoods, 1536-d MeanMapper/Aggregator embedding, bias-free 1536-d adapter, Linear-BN-LeakyReLU-Linear discriminator, sigma-0.015 feature noise and truncated L1 objective; 256-to-224 input, 160 epochs, batch 4 | Set `pretrained=True` to use the paper's ImageNet feature extractor; local default remains offline-safe |
| `vision_cutpaste` | CutPaste and scar geometry, patch jitter, 3-way objective, 256px input, 65,536-update default schedule, cosine decay, Gaussian feature density | Paper does not publish global translation/jitter amplitudes; local values remain configurable; patch-localization branch is not implemented |
| `core_deep_svdd` | bias-free network option and one-class center-distance objective on supplied feature vectors | Generic feature-vector architecture; soft-boundary objective and paper dataset-specific CNNs are not included |

This table is a source/code conformance audit, not a numerical reproduction
certificate. The primary references are the
[PatchCore paper and author code](https://github.com/amazon-science/patchcore-inspection),
[PaDiM paper](https://arxiv.org/abs/2011.08785),
[SPADE paper](https://arxiv.org/abs/2005.02357),
[STFPM paper](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1273.pdf),
[Reverse Distillation paper and author code](https://github.com/hq-deng/RD4AD),
[SimpleNet paper and author code](https://github.com/DonaldRR/SimpleNet),
[CutPaste paper and supplement](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html),
and [Deep SVDD author code](https://github.com/lukasruff/Deep-SVDD-PyTorch).

## Adaptations and compact variants

- Adaptations: `vision_alad`, `vision_devnet`, `vision_deep_svdd`,
  `vision_differnet`, `vision_memae`, `vision_draem`.
- Partial variants: `vision_cflow`, `vision_fastflow`, `vision_dfm`,
  `vision_fcdd`, `vision_softpatch`.

The native DRAEM entry now matches the author's base-128 reconstructive network,
base-64 discriminative network, initialization, losses, and 700-epoch schedule,
but its fallback texture synthesis remains an adaptation unless DTD images are
provided. DRAEM must not be reported as an exact paper experiment unless the
DTD anomaly source and full data protocol are also used.

The native DevNet entry now follows the 2021 image paper rather than applying
the 2019 tabular MLP to pooled image features. It trains ResNet-18 end to end,
uses a 1x1 convolutional patch scorer, averages the largest signed 10% of patch
scores over two scales (448 and 224 pixels), and optimizes the margin-5
deviation loss against 5,000 freshly sampled standard-normal references. Its
paper defaults are 50 epochs, 20 balanced half-normal/half-anomaly batches per
epoch, batch size 48, and Adam with learning rate 0.001 and weight decay 0.01.
Equation 6 defines top-K by anomaly score, so the native implementation uses
signed scores; the released repository's `abs()` shortcut is not copied. The
offline default remains `pretrained=False`, and the paper's smoothed input-level
localization map is not exposed, so this entry remains `paper-adaptation`.

The native DifferNet detection path matches the paper and author code at its
defining boundaries: frozen AlexNet convolutional features at 448/224/112,
global pooling to 768 dimensions, eight fixed-permutation two-sided affine
coupling blocks, three 2048-unit hidden layers per s/t subnet, clamp 3, latent
energy scoring, 4 training / 64 evaluation rotations, and the released
Adam/192-epoch schedule. For deterministic repeated scoring it uses the fixed
angle option from the authors' evaluation helper rather than resampling random
test rotations. Its offline default does not download ImageNet weights,
gradient-based anomaly localization is not exposed, and the MTD-specific
brightness/contrast protocol is not built in. The entry therefore remains
`paper-adaptation` rather than claiming a complete reproduction.

The native MemAE entry follows the paper's CIFAR-10 RGB topology
(3-64-128-128-256 and its mirrored decoder), 500 by 256 memory, cosine-softmax
addressing, differentiable hard shrinkage, per-query entropy regularization,
and 0.0001 learning rate. The paper did not define an MVTec experiment; applying
that CIFAR network and score contract to industrial images is explicitly an
adaptation.

Primary references for these boundaries are the
[image DevNet paper](https://arxiv.org/abs/2108.00462) and
[author-endorsed image code](https://github.com/Choubo/deviation-network-image),
[DifferNet paper](https://arxiv.org/abs/2008.12577) and
[author code](https://github.com/marco-rudolph/differnet), plus the
[MemAE paper](https://arxiv.org/abs/1904.02639) and
[author code](https://github.com/donggong1/memae-anomaly-detection).

Use these for their stated local contract, not as drop-in sources of published
benchmark results.

## External paper paths

Model-specific `vision_*_anomalib` entries load anomalib checkpoints and are
marked `external-backend`. `vision_patchcore_inspection_checkpoint` delegates to
the PatchCore inspection runtime. `vision_bayesianpf` requires an official
Bayes-PFL backend and checkpoint instead of constructing a random local proxy.

External-backend status describes delegation, not independent validation of a
particular checkpoint. Keep the upstream version, configuration, weights hash,
and dataset protocol with benchmark artifacts.

## Experimental compatibility proxies

The following names remain available for compatibility, but the local classes
are **not paper reproductions**:

- `vision_ast`, `vision_glad`, `vision_inctrl`, `vision_oneformore`
- `vision_panda`, `vision_promptad`, `vision_realnet`, `vision_regad`
- `vision_riad`, `vision_winclip`, `winclip`
- `vision_anomalydino`, `vision_patchcore_lite_map`
- `vision_aaclip`, `vision_adaclip`, `vision_anogen_adapter`, `vision_filopp`
- `vision_logsad`, `vision_one_to_normal`, `vision_univad`, `vision_visionad`

`vision_dst`, `vision_favae`, and `vision_gcad` are generic baselines; their old
paper titles could not be verified and were removed. The directly importable
legacy modules `bgad`, `dsr`, `intra`, `pni`, and `rdplusplus` are unregistered
experimental code and declare the same limitation in-module. The unregistered
`csflow` module is only a partial variant; prefer `vision_csflow_anomalib`.

## Inspecting a model

```bash
pyimgano-benchmark --model-info vision_patchcore --json
pyimgano-benchmark --model-info vision_promptad --json
```

```python
from pyimgano.models import model_info

metadata = model_info("vision_promptad")["metadata"]
print(metadata["paper_fidelity"])
print(metadata["implementation_status"])
print(metadata.get("related_paper"))
```

Registry tests reject deep entries that omit fidelity/status, mark a proxy as a
paper implementation, or restore the unsupported `sota` tag.
