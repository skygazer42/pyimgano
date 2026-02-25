# Model Index

This file is auto-generated from `pyimgano/models/*` by `tools/generate_model_index.py`.

Total registered model names: **123**

## Discovering models

From the CLI:

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags numpy,pixel_map
pyimgano-benchmark --model-info vision_patchcore --json
```

From Python:

```python
from pyimgano.models import list_models
print(list_models(tags=["pixel_map"])[:10])
```

## Adding your own model

Register models via `@register_model(...)` and implement the detector contract (`fit`, `decision_function`).
See `docs/DEEP_LEARNING_MODELS.md` for a minimal template.

| Name | Tags | Year | Backend | Description | Module |
|---|---|---:|---|---|---|
| `ae_resnet_unet` | vision, deep, autoencoder |  |  | 基于 ResNet-UNet 的重建式异常检测器 | `pyimgano/models/ae.py` |
| `core_deep_svdd` | deep, torch, one-class |  |  | 核心 DeepSVDD 异常检测器 | `pyimgano/models/deep_svdd.py` |
| `core_kpca` | classical, kernel, projection |  |  | 核心 Kernel PCA 异常检测器 | `pyimgano/models/kpca.py` |
| `core_loda` | classical, projection, density |  |  | 核心 LODA 算法实现 | `pyimgano/models/loda.py` |
| `cutpaste` | vision, deep, cutpaste, self-supervised, cvpr2021 | 2021 |  | CutPaste (legacy alias) - self-supervised anomaly detection via synthetic cut/paste | `pyimgano/models/cutpaste.py` |
| `dbscan_anomaly` | vision, classical, clustering |  |  | 基于 DBSCAN 的图像异常检测器 | `pyimgano/models/dbscan.py` |
| `devnet` | vision, deep, devnet, weakly-supervised, kdd2019 | 2019 |  | DevNet (legacy alias) - weakly-supervised deviation loss anomaly detection | `pyimgano/models/devnet.py` |
| `differnet` | vision, deep, differnet, knn, wacv2023, pixel_map | 2023 |  | DifferNet (legacy alias) - learnable difference + kNN anomaly detection | `pyimgano/models/differnet.py` |
| `efficient_ad` | vision, deep, distillation |  |  | EfficientAD 快速异常检测器 | `pyimgano/models/efficientad.py` |
| `isolation_forest_struct` | vision, classical, ensemble |  |  | 结构特征 Isolation Forest 检测器 | `pyimgano/models/Isolationforest.py` |
| `kmeans_anomaly` | vision, classical, clustering |  |  | K-Means 图像异常检测器 | `pyimgano/models/k_means.py` |
| `lof_structure` | vision, classical, neighbors |  |  | 结构特征 LOF 异常检测器 | `pyimgano/models/lof.py` |
| `memseg` | vision, deep, memseg, memory, segmentation, pixel_map | 2022 |  | MemSeg (legacy alias) - memory-guided anomaly segmentation | `pyimgano/models/memseg.py` |
| `one_class_cnn` | vision, classical, svm |  |  | 基于多特征的一类 SVM 图像检测器 | `pyimgano/models/one_svm_cnn.py` |
| `padim` | vision, deep, patch, distribution, numpy, pixel_map | 2020 |  | PaDiM (legacy alias) - patch distribution modeling | `pyimgano/models/padim.py` |
| `riad` | vision, deep, riad, reconstruction, self-supervised, pixel_map | 2020 |  | RIAD (legacy alias) - reconstruction by adjacent image decomposition | `pyimgano/models/riad.py` |
| `spade` | vision, deep, spade, knn, numpy, pixel_map | 2020 |  | SPADE (legacy alias) - Deep pyramid k-NN localization | `pyimgano/models/spade.py` |
| `ssim_struct` | vision, classical, template |  |  | 结构化多模板弹窗检测器 | `pyimgano/models/ssim_struct.py` |
| `ssim_template` | vision, classical, template |  |  | 基于模板匹配的 SSIM 异常检测器 | `pyimgano/models/ssim.py` |
| `vae_conv` | vision, deep, variational |  |  | 卷积变分自编码器异常检测器 | `pyimgano/models/vae.py` |
| `vision_abod` | vision, classical |  |  | 基于 ABOD 的视觉异常检测器 | `pyimgano/models/abod.py` |
| `vision_ae1svm` | vision, deep, svm |  |  | 自编码器 + 一类 SVM 组合的视觉检测器 | `pyimgano/models/ae1svm.py` |
| `vision_alad` | vision, deep, gan |  |  | Adversarially Learned Anomaly Detection | `pyimgano/models/alad.py` |
| `vision_anogan` | vision, deep, gan, anogan, pyod |  |  | PyOD AnoGAN wrapper (feature-based; requires pandas/matplotlib) | `pyimgano/models/anogan.py` |
| `vision_anomalib_checkpoint` | vision, deep, backend, anomalib |  | anomalib | Generic anomalib checkpoint inferencer wrapper (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_anomalydino` | vision, deep, anomalydino, knn, dinov2, numpy, pixel_map |  |  | AnomalyDINO-style DINOv2 patch-kNN detector (few-shot friendly) | `pyimgano/models/anomalydino.py` |
| `vision_ast` | vision, deep, ast, student-teacher, anomaly-aware, sota | 2023 |  | AST - Anomaly-aware Student-Teacher with synthetic anomalies | `pyimgano/models/ast.py` |
| `vision_auto_encoder` | vision, deep, autoencoder, pyod |  |  | PyOD AutoEncoder wrapper (feature-based) | `pyimgano/models/auto_encoder.py` |
| `vision_bayesianpf` | vision, deep, bayesianpf, zero-shot, bayesian, cvpr2025, sota | 2025 |  | BayesianPF - Bayesian Prompt Flow for Zero-Shot AD (CVPR 2025) | `pyimgano/models/bayesianpf.py` |
| `vision_cblof` | vision, classical, clustering |  |  | 基于 CBLOF 的视觉异常检测器 | `pyimgano/models/cblof.py` |
| `vision_cd` | vision, classical, cd, pyod |  |  | Cook's Distance (CD) wrapper via PyOD | `pyimgano/models/cd.py` |
| `vision_cfa_anomalib` | vision, deep, backend, anomalib, cfa |  | anomalib | CFA via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cflow` | vision, deep, cflow, normalizing-flow, real-time | 2022 |  | CFlow-AD - Conditional normalizing flows (WACV 2022) | `pyimgano/models/cflow.py` |
| `vision_cflow_anomalib` | vision, deep, backend, anomalib, cflow |  | anomalib | CFlow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cof` | vision, classical, neighbors, cof | 2002 |  | COF - Connectivity-based outlier detector | `pyimgano/models/cof.py` |
| `vision_copod` | vision, classical, copod, parameter-free, high-performance | 2020 |  | COPOD - Copula-based outlier detector (ICDM 2020) | `pyimgano/models/copod.py` |
| `vision_crossmad` | vision, deep, crossmad, cross-modal, cvpr2025, sota | 2025 |  | CrossMAD - Cross-Modal Anomaly Detection (CVPR 2025) | `pyimgano/models/crossmad.py` |
| `vision_csflow_anomalib` | vision, deep, backend, anomalib, csflow |  | anomalib | CS-Flow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cutpaste` | vision, deep, cutpaste, self-supervised, cvpr2021 | 2021 |  | CutPaste - self-supervised anomaly detection via synthetic cut/paste (CVPR 2021) | `pyimgano/models/cutpaste.py` |
| `vision_deep_svdd` | vision, deep, torch |  |  | 基于 DeepSVDD 的视觉异常检测器 | `pyimgano/models/deep_svdd.py` |
| `vision_devnet` | vision, deep, devnet, weakly-supervised, kdd2019 | 2019 |  | DevNet - weakly-supervised deviation loss anomaly detection (KDD 2019) | `pyimgano/models/devnet.py` |
| `vision_dfkde_anomalib` | vision, deep, backend, anomalib, dfkde |  | anomalib | DFKDE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dfm` | vision, deep, dfm, fast, gaussian |  |  | DFM - Fast discriminative feature modeling | `pyimgano/models/dfm.py` |
| `vision_dfm_anomalib` | vision, deep, backend, anomalib, dfm |  | anomalib | DFM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dif` | vision, deep, dif, pyod |  |  | Deep Isolation Forest (DIF) wrapper via PyOD | `pyimgano/models/dif.py` |
| `vision_differnet` | vision, deep, differnet, knn, wacv2023, pixel_map | 2023 |  | DifferNet - learnable difference + kNN anomaly detection (WACV 2023) | `pyimgano/models/differnet.py` |
| `vision_dinomaly_anomalib` | vision, deep, backend, anomalib, dinomaly |  | anomalib | Dinomaly via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_draem` | vision, deep, draem, reconstruction, synthetic, numpy, pixel_map | 2021 |  | DRAEM - Discriminatively trained reconstruction (ICCV 2021) | `pyimgano/models/draem.py` |
| `vision_draem_anomalib` | vision, deep, backend, anomalib, draem |  | anomalib | DRAEM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dsr_anomalib` | vision, deep, backend, anomalib, dsr |  | anomalib | DSR via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dst` | vision, deep, dst, student-teacher, distillation, sota | 2023 |  | DST - Double Student-Teacher with complementary learning | `pyimgano/models/dst.py` |
| `vision_ecod` | vision, classical, ecod, parameter-free, high-performance | 2022 |  | ECOD - Empirical CDF-based outlier detector (TKDE 2022) | `pyimgano/models/ecod.py` |
| `vision_efficientad_anomalib` | vision, deep, backend, anomalib, efficientad |  | anomalib | EfficientAD via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_fastflow` | vision, deep, flow |  |  | FastFlow-based visual anomaly detector | `pyimgano/models/fastflow.py` |
| `vision_fastflow_anomalib` | vision, deep, backend, anomalib, fastflow |  | anomalib | FastFlow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_favae` | vision, deep, favae, vae, adaptive, sota | 2023 |  | Feature Adaptive VAE - Dynamic latent space adaptation | `pyimgano/models/favae.py` |
| `vision_feature_bagging` | vision, ensemble, feature_bagging, high-performance | 2005 |  | Feature Bagging - Ensemble outlier detector | `pyimgano/models/feature_bagging.py` |
| `vision_fre_anomalib` | vision, deep, backend, anomalib, fre |  | anomalib | FRE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_ganomaly_anomalib` | vision, deep, backend, anomalib, ganomaly |  | anomalib | GANomaly via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_gcad` | vision, deep, gcad, graph, sota | 2023 |  | Graph Convolutional Anomaly Detection - Uses GCN to model spatial relationships | `pyimgano/models/gcad.py` |
| `vision_glad` | vision, deep, glad, diffusion, adaptive, eccv2024, sota | 2024 |  | GLAD - Global-Local Adaptive Diffusion (ECCV 2024) | `pyimgano/models/glad.py` |
| `vision_gmm` | vision, classical, gmm, density, baseline |  |  | Gaussian Mixture Model via PyOD (density baseline) | `pyimgano/models/gmm.py` |
| `vision_hbos` | vision, classical |  |  | Vision wrapper for histogram-based outlier detector | `pyimgano/models/hbos.py` |
| `vision_iforest` | vision, classical, iforest, ensemble, baseline | 2008 |  | Isolation Forest via PyOD (baseline, robust general-purpose) | `pyimgano/models/iforest.py` |
| `vision_imdd` | vision, classical |  |  | Vision wrapper for IMDD deviation detector | `pyimgano/models/imdd.py` |
| `vision_inctrl` | vision, deep, inctrl, few-shot, generalist, cvpr2024, sota | 2024 |  | InCTRL - In-context Residual Learning for generalist AD (CVPR 2024) | `pyimgano/models/inctrl.py` |
| `vision_inne` | vision, classical, isolation, inne, fast | 2014 |  | INNE - Isolation using Nearest-Neighbor Ensembles | `pyimgano/models/inne.py` |
| `vision_kde` | vision, classical, kde, density, baseline |  |  | Kernel Density Estimation via PyOD (density baseline) | `pyimgano/models/kde.py` |
| `vision_knn` | vision, classical, neighbors, knn | 2000 |  | Vision wrapper for KNN outlier detector | `pyimgano/models/knn.py` |
| `vision_kpca` | vision, classical, kernel |  |  | 基于 Kernel PCA 的视觉异常检测器 | `pyimgano/models/kpca.py` |
| `vision_lmdd` | vision, classical, lmdd, baseline |  |  | LMDD via PyOD (baseline) | `pyimgano/models/lmdd.py` |
| `vision_loci` | vision, classical |  |  | Vision wrapper for LOCI outlier detector | `pyimgano/models/loci.py` |
| `vision_loda` | vision, classical |  |  | 基于 LODA 的视觉异常检测器 | `pyimgano/models/loda.py` |
| `vision_lscp` | vision, classical, ensemble |  |  | Vision wrapper for LSCP detector ensemble | `pyimgano/models/lscp.py` |
| `vision_lunar` | vision, deep, lunar, pyod |  |  | LUNAR wrapper via PyOD | `pyimgano/models/lunar.py` |
| `vision_mad` | vision, classical, mad, robust, baseline |  |  | Multivariate MAD robust baseline (median + MAD robust z-score) | `pyimgano/models/mad.py` |
| `vision_mambaad` | vision, deep, mambaad, mamba, ssm, numpy, pixel_map | 2024 |  | MambaAD-style patch embedding reconstruction with Mamba SSM (NeurIPS 2024) | `pyimgano/models/mambaad.py` |
| `vision_mcd` | vision, classical, statistical, mcd, robust | 1999 |  | MCD - Robust covariance-based outlier detector | `pyimgano/models/mcd.py` |
| `vision_memseg` | vision, deep, memseg, memory, segmentation, pixel_map | 2022 |  | MemSeg - memory-guided anomaly segmentation (ICCV 2022-style) | `pyimgano/models/memseg.py` |
| `vision_mo_gaal` | vision, deep, gan |  |  | Vision wrapper for MO-GAAL anomaly detector | `pyimgano/models/mo_gaal.py` |
| `vision_ocsvm` | vision, classical, svm |  |  | 模块化一类 SVM 视觉异常检测器 | `pyimgano/models/ocsvm.py` |
| `vision_oddoneout` | vision, deep, oddoneout, neighbors, cvpr2025, sota | 2025 |  | Odd-One-Out - Neighbor Comparison AD (CVPR 2025) | `pyimgano/models/oddoneout.py` |
| `vision_oneformore` | vision, deep, oneformore, continual, diffusion, cvpr2025, sota | 2025 |  | One-for-More - Continual Diffusion Model (CVPR 2025, #1 on MVTec/VisA) | `pyimgano/models/oneformore.py` |
| `vision_openclip_patchknn` | vision, deep, clip, openclip, backend, knn |  | openclip | OpenCLIP patch embedding + kNN detector (requires pyimgano[clip]) | `pyimgano/models/openclip_backend.py` |
| `vision_openclip_promptscore` | vision, deep, clip, openclip, backend, prompt |  | openclip | OpenCLIP prompt scoring detector (requires pyimgano[clip]) | `pyimgano/models/openclip_backend.py` |
| `vision_padim` | vision, deep, patch, distribution, numpy, pixel_map | 2020 |  | PaDiM - patch distribution modeling (ECCV 2020-style) | `pyimgano/models/padim.py` |
| `vision_padim_anomalib` | vision, deep, backend, anomalib, padim |  | anomalib | PaDiM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_panda` | vision, deep, panda, prototypical, metric, sota | 2023 |  | PANDA - Prototypical Anomaly Network with metric learning | `pyimgano/models/panda.py` |
| `vision_patchcore` | vision, deep, patchcore, sota, cvpr2022, numpy, pixel_map | 2022 |  | PatchCore - SOTA patch-level anomaly detection (CVPR 2022) | `pyimgano/models/patchcore.py` |
| `vision_patchcore_anomalib` | vision, deep, backend, anomalib, patchcore |  | anomalib | PatchCore via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_patchcore_inspection_checkpoint` | vision, deep, backend, patchcore_inspection, patchcore |  | patchcore_inspection | PatchCore (amazon-science/patchcore-inspection) checkpoint wrapper (optional backend) | `pyimgano/models/patchcore_inspection_backend.py` |
| `vision_pca` | vision, classical, linear, pca | 2003 |  | Vision wrapper for PCA-based outlier detector | `pyimgano/models/pca.py` |
| `vision_promptad` | vision, deep, promptad, few-shot, prompt, cvpr2024, sota | 2024 |  | PromptAD - Prompt learning with only normal samples (CVPR 2024) | `pyimgano/models/promptad.py` |
| `vision_qmcd` | vision, classical, qmcd, robust, baseline |  |  | QMCD via PyOD (robust covariance baseline) | `pyimgano/models/qmcd.py` |
| `vision_realnet` | vision, deep, realnet, feature-selection, cvpr2024, sota | 2024 |  | RealNet - Feature Selection with Realistic Synthetic Anomaly (CVPR 2024) | `pyimgano/models/realnet.py` |
| `vision_regad` | vision, deep, regad, registration, alignment, sota | 2023 |  | RegAD - Registration-based anomaly detection with STN | `pyimgano/models/regad.py` |
| `vision_reverse_dist` | vision, deep, distillation |  |  | Reverse distillation anomaly detector (alias) | `pyimgano/models/reverse_distillation.py` |
| `vision_reverse_distillation` | vision, deep, distillation |  |  | Reverse distillation anomaly detector | `pyimgano/models/reverse_distillation.py` |
| `vision_reverse_distillation_anomalib` | vision, deep, backend, anomalib, reverse_distillation |  | anomalib | Reverse Distillation via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_rgraph` | vision, classical, rgraph, pyod |  |  | RGraph wrapper via PyOD | `pyimgano/models/rgraph.py` |
| `vision_riad` | vision, deep, riad, reconstruction, self-supervised, pixel_map | 2020 |  | RIAD - reconstruction by adjacent image decomposition (2020-style) | `pyimgano/models/riad.py` |
| `vision_rkde_anomalib` | vision, deep, backend, anomalib, rkde |  | anomalib | R-KDE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_rod` | vision, classical, rod, baseline |  |  | Rotation-based Outlier Detection via PyOD (baseline) | `pyimgano/models/rod.py` |
| `vision_sampling` | vision, classical, sampling, pyod |  |  | Sampling wrapper via PyOD | `pyimgano/models/sampling.py` |
| `vision_score_ensemble` | vision, ensemble, score |  |  | Score-only ensemble wrapper (mean of rank-normalized scores by default) | `pyimgano/models/score_ensemble.py` |
| `vision_simplenet` | vision, deep, simplenet, fast, sota, cvpr2023 | 2023 |  | SimpleNet - Ultra-fast SOTA anomaly detection (CVPR 2023) | `pyimgano/models/simplenet.py` |
| `vision_so_gaal` | vision, deep, gan, so_gaal, pyod |  |  | SO-GAAL wrapper via PyOD | `pyimgano/models/so_gaal.py` |
| `vision_so_gaal_new` | vision, deep, gan, so_gaal, pyod |  |  | SO-GAAL (new) wrapper via PyOD | `pyimgano/models/so_gaal_new.py` |
| `vision_sod` | vision, classical, sod, subspace, baseline |  |  | Subspace Outlier Detection via PyOD (subspace baseline) | `pyimgano/models/sod.py` |
| `vision_softpatch` | vision, deep, softpatch, patchknn, robust, numpy, pixel_map |  |  | SoftPatch-inspired robust patch-memory detector (few-shot friendly) | `pyimgano/models/softpatch.py` |
| `vision_sos` | vision, classical, sos, probabilistic, baseline |  |  | Stochastic Outlier Selection via PyOD (probabilistic baseline) | `pyimgano/models/sos.py` |
| `vision_spade` | vision, deep, spade, knn, numpy, pixel_map | 2020 |  | SPADE - Deep pyramid k-NN localization (ECCV 2020) | `pyimgano/models/spade.py` |
| `vision_stfpm` | vision, deep, stfpm, student-teacher, pyramid, numpy, pixel_map | 2021 |  | STFPM - Student-Teacher Feature Pyramid Matching (BMVC 2021) | `pyimgano/models/stfpm.py` |
| `vision_stfpm_anomalib` | vision, deep, backend, anomalib, stfpm |  | anomalib | STFPM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_suod` | vision, classical, ensemble |  |  | Vision wrapper for SUOD ensemble detector | `pyimgano/models/suod.py` |
| `vision_superad` | vision, deep, superad, knn, dinov2, numpy, pixel_map |  |  | SuperAD-style DINOv2 patch-kNN detector using k-th NN distance per patch | `pyimgano/models/superad.py` |
| `vision_supersimplenet_anomalib` | vision, deep, backend, anomalib, supersimplenet |  | anomalib | SuperSimpleNet via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_uflow_anomalib` | vision, deep, backend, anomalib, uflow |  | anomalib | U-Flow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_vlmad_anomalib` | vision, deep, backend, anomalib, vlmad |  | anomalib | VLM-AD via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_winclip` | vision, deep, winclip, clip | 2023 |  | WinCLIP - Zero-/Few-shot CLIP-based anomaly detection (CVPR 2023) | `pyimgano/models/winclip.py` |
| `vision_winclip_anomalib` | vision, deep, backend, anomalib, winclip |  | anomalib | WinCLIP via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_xgbod` | vision, classical, supervised |  |  | Vision wrapper for XGBOD semi-supervised detector | `pyimgano/models/xgbod.py` |
| `winclip` |  |  |  |  | `pyimgano/models/winclip.py` |
