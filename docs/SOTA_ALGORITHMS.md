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
- `vision_softpatch`
- `vision_padim`
- `vision_stfpm`
- `vision_cflow`
- `vision_reverse_distillation` / `vision_reverse_dist`
- `vision_simplenet`
- `vision_spade`
- `vision_cutpaste` / `cutpaste`

These entries implement the defining algorithmic core. Inspect each model's
weight default: offline-safe `pretrained=False` produces an experimentally weak
model, while Reverse Distillation follows the paper with pretrained weights by
default. Reproducing paper metrics additionally requires the paper's data split
and evaluation protocol.

### Audited native scope (2026-07-15)

| Entry | Verified native scope | Remaining boundary |
|---|---|---|
| `vision_patchcore` | WRN50-2 layer2/layer3 padded 3x3 patches, 1024-to-1024 MeanMapper/Aggregator embedding, 10% approximate greedy coreset, 1-NN max score; resize 256 then crop 224 | Paper experiments require ImageNet weights; offline default is `pretrained=False` |
| `vision_softpatch` | PatchCore WRN50-2 layer2/layer3 embedding, position-wise LOF with k=6, top-15% patch removal, 10% greedy coreset with stored LOF weights, weighted 1-NN patch score, max image score, sigma-4 map smoothing | Paper experiments require ImageNet weights and the noisy MVTec/BTAD protocol; offline default is `pretrained=False` |
| `vision_padim` | R18 448→100 and WR50-2 1792→550 channel paths, blockwise cross-level correspondence, per-location Gaussian, Mahalanobis map, bicubic resize, sigma-4 smoothing | Paper experiments require ImageNet weights; offline default is `pretrained=False`; the paper does not link an author reference implementation |
| `vision_spade` | WRN50-2 ImageNet-V1 global mean squared-L2 KNN (`K=50`), block1--3 pyramid concatenation on the finest grid, squared-L2 correspondence (`kappa=1`), INTER_AREA 256px map, sigma-4 smoothing; resize 256 then crop 224 | Set `pretrained=True` for the paper weights; offline default is `False` |
| `vision_stfpm` | identical ResNet-18 teacher/student, conv2_x--conv4_x normalized feature loss, SGD 0.4 for 100 epochs, exact seeded 80/20 split, full-resolution bilinear map product, max image score | Exact paper path requires `pretrained_teacher=True`; the offline default keeps downloads opt-in |
| `vision_cflow` | frozen ResNet layer2--layer4 pyramid, 128-D sinusoidal position conditions, three independent eight-block conditional flows, normalized likelihood objective, summed multi-scale probability maps | Set `pretrained_backbone=True` and use the authors' category-specific input size and evaluation protocol for published MVTec experiments |
| `vision_reverse_distillation` | ImageNet WideResNet50-2 teacher stages 1--3, exact released OCBE and reverse-WRN block counts/channels, cosine loss, additive cosine maps, sigma-4 smoothing; 256px, Adam 0.005, batch 16, 200 epochs | Published metrics still require the MVTec category protocol and matching ImageNet weights |
| `vision_simplenet` | WRN50-2 layer2/layer3 padded 3x3 neighborhoods, 1536-d MeanMapper/Aggregator embedding, bias-free 1536-d adapter, Linear-BN-LeakyReLU-Linear discriminator, sigma-0.015 feature noise and truncated L1 objective; 256-to-224 input, 160 epochs, batch 4 | Set `pretrained=True` to use the paper's ImageNet feature extractor; local default remains offline-safe |
| `vision_cutpaste` | ResNet-18 pooled 512-d features, MLP CutPaste/scar 3-way objective, paper geometry and patch jitter, 256px input, 65,536 updates, SGD 0.03/0.9, weight decay 3e-5, cosine decay, Ledoit-Wolf Gaussian fit with the Eq. (2) negative log-density score | The paper does not publish MLP widths, input normalization, global translation/jitter amplitudes, covariance regularization, or an author implementation; EfficientNet-B4 transfer and patch-localization branches are not implemented |
| `vision_efficientad` / `efficient_ad` | EfficientAD-S/M PDNs, 384-channel teacher and 768-channel student, 64-D bottleneck autoencoder, exact 8.06M/20.74M parameter totals (paper: 8M/21M), channel normalization, hard-feature/ImageNet-penalty/AE losses, 70,000-step Adam schedule, 0.9/0.995 map calibration, equal map fusion, and max image score | The paper provides no official teacher checkpoint or repository; strict fitting requires an explicitly distilled teacher and ImageNet-style penalty directory; the local 10% normal holdout is used only when no validation set is supplied because the paper does not state a split fraction |

This table is a source/code conformance audit, not a numerical reproduction
certificate. The primary references are the
[PatchCore paper and author code](https://github.com/amazon-science/patchcore-inspection),
[SoftPatch paper and author code](https://github.com/TencentYoutuResearch/AnomalyDetection-SoftPatch),
[PaDiM paper](https://arxiv.org/abs/2011.08785),
[SPADE paper](https://arxiv.org/abs/2005.02357),
[STFPM paper](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1273.pdf) and [author code](https://github.com/gdwang08/STFPM),
[CFLOW-AD paper and author code](https://github.com/gudovskiy/cflow-ad),
[Reverse Distillation paper and author code](https://github.com/hq-deng/RD4AD),
[SimpleNet paper and author code](https://github.com/DonaldRR/SimpleNet),
[EfficientAD paper and supplement](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html),
[CutPaste paper and supplement](https://openaccess.thecvf.com/content/CVPR2021/html/Li_CutPaste_Self-Supervised_Learning_for_Anomaly_Detection_and_Localization_CVPR_2021_paper.html).
The SPADE paper does not link an author repository; its equations and stated
experiment parameters are therefore the canonical source for this audit.

## Adaptations

- Adaptations: `vision_alad`, `core_deep_svdd`, `vision_deep_svdd`, `vision_devnet`,
  `vision_dfm`, `vision_differnet`, `vision_memae`, `vision_draem`, `vision_fastflow`,
  `vision_fcdd`, `vision_efficientad` / `efficient_ad`.

The native EfficientAD entry implements both supplementary PDN tables, the
64-dimensional autoencoder, all three training losses, teacher-channel and
map-quantile calibration, the 70,000-step optimizer schedule, and the paper's
equal-map/max-score inference equations. The paper supplies neither an
official repository nor distilled teacher weights, so `paper_strict=True`
requires an explicit `teacher_checkpoint` and `imagenet_dir`; it never silently
substitutes a random ResNet or downloads assets. A caller may disable this gate
for diagnostics, but that run is not a paper reproduction.

The native DFM entry implements the Gaussian branch of the
[DFM paper](https://arxiv.org/abs/1909.11786): it extracts one independently
selected ResNet block, applies the paper's factor-4 average pooling and 99.5%
PCA variance retention, estimates the Gaussian by maximum likelihood, and
scores with the full negative log-likelihood including the log-determinant.
The default backbone/layer is ResNet50 `layer4`, one of the paper's three final
residual blocks. The repository API fits only the supplied normal class, uses a
contamination threshold, resizes industrial images to 224px, and keeps
pretrained weights opt-in, so this is a `paper-adaptation`, not a reproduction
of the paper's labeled CIFAR experiments.

The native CFLOW-AD entry implements the authors' released ResNet path: frozen
layer2--layer4 features, 128-D sinusoidal 2-D positional conditions, one
independent eight-block conditional FrEIA-equivalent decoder per scale, the
normalized likelihood objective, cosine schedule with warmup, and summed
multi-scale probability maps. The default WideResNet50-2 architecture and
decoder hyperparameters follow the author configuration. ImageNet downloads
remain opt-in, while the paper's published MVTec protocol also uses
category-specific input sizes; `core-aligned` therefore describes the defining
algorithm, not reproduction of the reported AUC.

The native FastFlow entry now implements the paper's CNN path: frozen
ResNet18 or WideResNet50-2 features from residual stages 1--3 at 256px, one
2-D flow per scale, ActNorm, fixed channel permutation, two-convolution affine
coupling subnets, eight flow steps, the paper's 3x3-only ResNet18 and alternating
3x3/1x1 WideResNet50-2 schedules, spatial likelihood training, and averaged
upsampled probability maps. The resulting WideResNet50-2 flow parameter counts
are 74.4M (3-3) and 41.3M (3-1), matching Table 5. Adam defaults are 0.001,
weight decay 0.00001, batch size 32, and 500 epochs. The paper does not release
an author implementation or fully specify affine stabilization, probability-map
normalization, image-score reduction, or rotation angles; the local default also
disables ImageNet downloads and category-specific augmentation. The entry is
therefore `paper-adaptation`, not a claim that its default run reproduces the
published AUC.

The native DRAEM entry now matches the author's base-128 reconstructive network,
base-64 discriminative network, initialization, losses, and 700-epoch schedule,
but its fallback texture synthesis remains an adaptation unless DTD images are
provided. DRAEM must not be reported as an exact paper experiment unless the
DTD anomaly source and full data protocol are also used.

The native FCDD entry now follows the paper's MVTec path: a 4,504,833-parameter
truncated VGG11-BN FCN with a 1x1 score head, element-wise pseudo-Huber map,
normal/anomalous HSC objective, 50% online confetti replacement, SGD/Nesterov
schedule, and receptive-field Gaussian transposed-convolution upsampling. Set
`pretrained=True` to load and freeze the paper's ImageNet feature slice. The
offline default trains the same network without those weights and learns the
category normalization bounds from supplied normal images, so the registry
classifies it as `paper-adaptation` rather than a numerical reproduction.

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

The Deep SVDD feature-space entries implement both objectives from the paper:
one-class mean squared center distance and soft-boundary radius loss with
signed `distance - radius^2` scores. The encoder has no bias terms, its final
projection is linear, the center uses the authors' epsilon adjustment, and the
soft radius is the `(1 - nu)` distance quantile. Defaults now use no dropout,
leaky ReLU slope 0.1, weight decay 1e-6, and the author's released CLI values
of learning rate 0.001, 50 epochs, and batch size 128. They remain adaptations:
the shipped network is a generic MLP over supplied features, not the paper's
MNIST/CIFAR-10 LeNet CNNs; DCAE initialization is opt-in, the paper's two-phase
benchmark schedule is not included in this generic path, and feature
standardization is a local deployment choice.

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

The native AST entry follows the MVTec AD RGB path: frozen EfficientNet-B5
layer-36 features at 768 px, a four-block position-conditioned RealNVP teacher,
a four-residual-block convolutional student, sequential likelihood/regression
training, and mean spatial student-teacher distance scoring. ImageNet weights
are opt-in for offline safety, and the MVTec 3D-AD depth/foreground path is not
implemented, so the entry is classified as `paper-adaptation`.

The native RIAD entry follows the paper's three-way disjoint `k x k` region
masking, five-level 64/128/256/512/512 U-Net, assembled partial inpaintings,
combined L2/SSIM/MSGMS loss, `{2, 4, 8, 16}` region-size ensemble, and MSGMS
maximum-map scoring. Since the authors did not release reference code, this is
classified as `paper-adaptation`, not a source-identical reproduction.

The native PANDA entry implements the fixed-iteration PANDA-Early image-level
path: ImageNet-pretrained ResNet152 pooled features, compactness adaptation of
blocks 3/4 with the published optimizer and clipping defaults, and summed
squared-L2 2-NN scoring. It does not claim the Fisher-dependent PANDA-EWC,
checkpoint-ensemble PANDA-SES, or the paper's separate SPADE segmentation path.

The native RealNet entry implements the paper's WideResNet50-2 four-level
features, 64-batch AFS channel selection, independent reconstruction U-Nets,
max/mean RRS decoder, joint reconstruction/segmentation objectives, and
published optimizer/schedule. SDAS remains the paper's separate offline data
generation stage: fitting requires paired SDAS/SIA-style anomaly images and
masks. The local unsupervised contract also does not use labeled validation
anomalies to choose a best checkpoint as the author evaluation script does.

Primary references for these boundaries are the
[image DevNet paper](https://arxiv.org/abs/2108.00462) and
[author-endorsed image code](https://github.com/Choubo/deviation-network-image),
[Deep SVDD paper](https://proceedings.mlr.press/v80/ruff18a.html) and
[author code](https://github.com/lukasruff/Deep-SVDD-PyTorch),
[DifferNet paper](https://arxiv.org/abs/2008.12577) and
[author code](https://github.com/marco-rudolph/differnet), plus the
[MemAE paper](https://arxiv.org/abs/1904.02639) and
[author code](https://github.com/donggong1/memae-anomaly-detection), the
[AST paper](https://openaccess.thecvf.com/content/WACV2023/html/Rudolph_Asymmetric_Student-Teacher_Networks_for_Industrial_Anomaly_Detection_WACV_2023_paper.html)
and [author code](https://github.com/marco-rudolph/AST), the
[RIAD paper](https://doi.org/10.1016/j.patcog.2020.107706), and the
[FastFlow paper](https://arxiv.org/abs/2111.07677), plus the
[PANDA paper](https://openaccess.thecvf.com/content/CVPR2021/html/Reiss_PANDA_Adapting_Pretrained_Features_for_Anomaly_Detection_and_Segmentation_CVPR_2021_paper.html)
and [RealNet paper](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_RealNet_A_Feature_Selection_Network_with_Realistic_Synthetic_Anomaly_for_CVPR_2024_paper.html),
plus the [RegAD paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840300.pdf)
and [author code](https://github.com/MediaBrain-SJTU/RegAD), plus the
[PromptAD paper](https://openaccess.thecvf.com/content/CVPR2024/html/Li_PromptAD_Learning_Prompts_with_only_Normal_Samples_for_Few-Shot_Anomaly_CVPR_2024_paper.html)
and [author code](https://github.com/FuNz-0/PromptAD).

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

- `vision_glad`, `vision_inctrl`, `vision_oneformore`
- `vision_winclip`, `winclip`
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
from pyimgano.models.registry import model_info

metadata = model_info("vision_promptad")["metadata"]
print(metadata["paper_fidelity"])
print(metadata["implementation_status"])
print(metadata.get("paper"))
```

Registry tests reject deep entries that omit fidelity/status, mark a proxy as a
paper implementation, or restore the unsupported `sota` tag.
