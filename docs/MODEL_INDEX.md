# Model Index

This file is auto-generated from `pyimgano/models/*` by `tools/generate_model_index.py`.

Total registered model names: **278**

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
| `ae_resnet_unet` | vision, deep, autoencoder, reconstruction |  |  | Reconstruction baseline (contract-aligned autoencoder; legacy name kept) | `pyimgano/models/ae.py` |
| `core_abod` | classical, core, features, abod, neighbors | 2008 |  | Core ABOD detector on feature matrices (native wrapper) | `pyimgano/models/abod.py` |
| `core_cblof` | classical, core, features, clustering, cblof |  |  | Core CBLOF detector on feature matrices (native implementation) | `pyimgano/models/cblof.py` |
| `core_cof` | classical, core, features, neighbors, cof, density | 2002 |  | Core COF detector on feature matrices (native wrapper) | `pyimgano/models/cof.py` |
| `core_cook_distance` | classical, core, features, pca, influence |  |  | Cook-distance-inspired influence score (PCA residual + leverage) | `pyimgano/models/cook_distance.py` |
| `core_copod` | classical, core, features, copod, parameter-free, interpretable | 2020 |  | Core COPOD on feature matrices (native wrapper) | `pyimgano/models/copod.py` |
| `core_cosine_mahalanobis` | classical, core, features, distance, gaussian, shrinkage, cosine |  |  | Mahalanobis distance on L2-normalized embeddings with Ledoit-Wolf covariance shrinkage | `pyimgano/models/core_cosine_mahalanobis.py` |
| `core_crossmad` | classical, core, features, crossmad, prototype, clustering | 2025 |  | Core CrossMAD-style prototype-distance detector on feature matrices (native) | `pyimgano/models/crossmad.py` |
| `core_dbscan` | classical, core, features, clustering, dbscan, density |  |  | Core DBSCAN-inspired distance-to-core-set detector on feature matrices | `pyimgano/models/dbscan.py` |
| `core_dcorr` | classical, core, features, projection, dependence |  |  | Distance-correlation influence (random projections; native) | `pyimgano/models/dcorr.py` |
| `core_deep_svdd` | deep, core, features, torch, one-class |  |  | 核心 DeepSVDD 异常检测器 | `pyimgano/models/deep_svdd.py` |
| `core_dtc` | classical, core, features, distance, baseline |  |  | Distance-to-centroid baseline (L2 to mean) | `pyimgano/models/dtc.py` |
| `core_ecod` | classical, core, features, ecod, parameter-free, interpretable | 2022 |  | Core ECOD on feature matrices (native wrapper) | `pyimgano/models/ecod.py` |
| `core_elliptic_envelope` | classical, core, features, gaussian, covariance |  |  | Robust covariance (MCD) Mahalanobis-distance outlier baseline | `pyimgano/models/elliptic_envelope.py` |
| `core_extra_trees_density` | classical, core, features, trees, density |  |  | RandomTreesEmbedding leaf-rarity density baseline (native) | `pyimgano/models/extra_trees_density.py` |
| `core_feature_bagging` | classical, core, features, ensemble, feature_bagging | 2005 |  | Core Feature Bagging ensemble on feature matrices (native wrapper) | `pyimgano/models/feature_bagging.py` |
| `core_feature_bagging_spec` | classical, core, features, ensemble, feature_bagging, spec | 2005 |  | Core Feature Bagging ensemble with JSON-friendly base-estimator spec | `pyimgano/models/feature_bagging.py` |
| `core_gmm` | classical, core, features, gmm, density, baseline |  |  | Core Gaussian Mixture Model detector on feature matrices (native wrapper) | `pyimgano/models/gmm.py` |
| `core_hbos` | classical, core, features, hbos, histogram, fast, baseline |  |  | Core HBOS on feature matrices (native wrapper) | `pyimgano/models/hbos.py` |
| `core_hst` | classical, core, features, tree, online |  |  | Half-Space Trees (leaf-mass scoring; native) | `pyimgano/models/hst.py` |
| `core_iforest` | classical, core, features, iforest, ensemble, baseline | 2008 |  | Core Isolation Forest on feature matrices (native wrapper) | `pyimgano/models/iforest.py` |
| `core_imdd` | classical, core, features, imdd, lmdd |  |  | IMDD/LMDD deviation detector for feature matrices (native wrapper) | `pyimgano/models/imdd.py` |
| `core_inne` | classical, core, features, isolation, inne, fast | 2014 |  | Core INNE on feature matrices (native wrapper) | `pyimgano/models/inne.py` |
| `core_kde` | classical, core, features, kde, density, baseline |  |  | Core Kernel Density Estimation detector on feature matrices (native wrapper) | `pyimgano/models/kde.py` |
| `core_kde_ratio` | classical, core, features, density, kde |  |  | Dual-bandwidth KDE density-contrast outlier score (native) | `pyimgano/models/kde_ratio.py` |
| `core_kmeans` | classical, core, features, clustering, kmeans |  |  | Core KMeans distance-to-centroid baseline on feature matrices | `pyimgano/models/k_means.py` |
| `core_knn` | classical, core, features, neighbors, knn |  |  | Core KNN outlier detector on feature matrices (native wrapper) | `pyimgano/models/knn.py` |
| `core_knn_cosine` | classical, core, features, neighbors, knn, cosine |  |  | Cosine kNN distance outlier detector (embedding-friendly) | `pyimgano/models/core_knn_cosine.py` |
| `core_knn_cosine_calibrated` | classical, core, features, neighbors, knn, cosine, calibration |  |  | Cosine kNN detector with unsupervised score standardization (rank/zscore/robust/minmax) | `pyimgano/models/core_knn_cosine_calibrated.py` |
| `core_knn_degree` | classical, core, features, neighbors, graph, density |  |  | kNN epsilon-graph degree (radius chosen from kNN distances) | `pyimgano/models/knn_degree.py` |
| `core_kpca` | classical, core, features, kernel, projection |  |  | 核心 Kernel PCA 异常检测器 | `pyimgano/models/kpca.py` |
| `core_ldof` | classical, core, features, neighbors, local |  |  | LDOF - Local Distance-based Outlier Factor (native) | `pyimgano/models/ldof.py` |
| `core_lid` | classical, core, features, neighbors, lid |  |  | Local Intrinsic Dimensionality (LID) kNN-distance outlier score | `pyimgano/models/lid.py` |
| `core_lmdd` | classical, core, features, lmdd, imdd |  |  | LMDD deviation detector for feature matrices (native wrapper) | `pyimgano/models/lmdd.py` |
| `core_loci` | classical, core, features, loci, density | 2003 |  | Core LOCI detector on feature matrices (native wrapper) | `pyimgano/models/loci.py` |
| `core_loda` | classical, core, features, projection, density |  |  | 核心 LODA 算法实现 | `pyimgano/models/loda.py` |
| `core_lof` | classical, core, features, lof, neighbors, density |  |  | Core Local Outlier Factor detector on feature matrices (sklearn backend) | `pyimgano/models/lof_core.py` |
| `core_loop` | classical, core, features, neighbors, probability |  |  | LoOP - Local Outlier Probability (native implementation) | `pyimgano/models/loop.py` |
| `core_lscp` | classical, core, features, ensemble, lscp | 2019 |  | Core LSCP ensemble on feature matrices (native wrapper) | `pyimgano/models/lscp.py` |
| `core_lscp_spec` | classical, core, features, ensemble, lscp | 2019 |  | Core LSCP ensemble with JSON-friendly base-detector specs | `pyimgano/models/lscp.py` |
| `core_mad` | classical, core, features, mad, robust, baseline |  |  | Core multivariate MAD baseline on feature matrices (native wrapper) | `pyimgano/models/mad.py` |
| `core_mahalanobis` | classical, core, features, distance, gaussian |  |  | Mahalanobis distance baseline (mean + covariance) | `pyimgano/models/mahalanobis.py` |
| `core_mahalanobis_shrinkage` | classical, core, features, distance, gaussian, shrinkage |  |  | Mahalanobis distance with Ledoit-Wolf covariance shrinkage (embedding-friendly) | `pyimgano/models/core_mahalanobis_shrinkage.py` |
| `core_mcd` | classical, core, features, statistical, mcd, robust | 1999 |  | Core MCD robust covariance outlier detector on feature matrices (native wrapper) | `pyimgano/models/mcd.py` |
| `core_mst_outlier` | classical, core, features, graph, mst |  |  | MST-based outlier baseline (max incident MST edge length) | `pyimgano/models/mst_outlier.py` |
| `core_neighborhood_entropy` | classical, core, features, neighbors, graph, entropy |  |  | Neighborhood entropy score over kNN distances (native baseline) | `pyimgano/models/neighborhood_entropy.py` |
| `core_ocsvm` | classical, core, features, svm, one-class, ocsvm |  |  | Core One-Class SVM detector on feature matrices (native wrapper) | `pyimgano/models/ocsvm.py` |
| `core_oddoneout` | classical, core, features, neighbors, oddoneout, cvpr2025 |  |  | Odd-One-Out (neighbor comparison) core detector on feature matrices | `pyimgano/models/core_oddoneout.py` |
| `core_odin` | classical, core, features, neighbors, graph |  |  | ODIN - indegree-based kNN graph outlier detector (native) | `pyimgano/models/odin.py` |
| `core_padim_lite` | classical, core, features, padim, gaussian |  |  | PaDiM-lite: Gaussian embedding baseline via robust covariance (Mahalanobis distance) | `pyimgano/models/padim_lite.py` |
| `core_patchcore_lite` | classical, core, features, neighbors, memory_bank, patchcore |  |  | PatchCore-lite: coreset memory bank + nearest-neighbor distance (image-level) | `pyimgano/models/patchcore_lite.py` |
| `core_patchcore_online` | classical, core, features, neighbors, memory_bank, patchcore, online |  |  | PatchCore-online: incremental memory bank + nearest-neighbor distance (image-level) | `pyimgano/models/patchcore_online.py` |
| `core_pca` | classical, core, features, linear, pca | 2003 |  | Core PCA reconstruction-error detector on feature matrices (native wrapper) | `pyimgano/models/pca.py` |
| `core_pca_md` | classical, core, features, pca, distance |  |  | PCA + Mahalanobis distance (subspace) | `pyimgano/models/pca_md.py` |
| `core_qmcd` | classical, core, features, qmcd, robust, baseline | 2001 |  | Core QMCD discrepancy detector on feature matrices (native wrapper) | `pyimgano/models/qmcd.py` |
| `core_random_projection_knn` | classical, core, features, neighbors, knn, projection |  |  | Random projection + kNN distance outlier score (native wrapper) | `pyimgano/models/random_projection_knn.py` |
| `core_rgraph` | classical, core, features, rgraph, graph |  |  | Core graph random-walk outlier detector on feature matrices (native wrapper) | `pyimgano/models/rgraph.py` |
| `core_rod` | classical, core, features, rod |  |  | ROD (Rotation-based Outlier Detection) for feature matrices (native wrapper) | `pyimgano/models/rod.py` |
| `core_rrcf` | classical, core, features, forest, random-cut |  |  | Random cut forest baseline (RRCF-style tree construction) | `pyimgano/models/rrcf.py` |
| `core_rzscore` | classical, core, features, robust, baseline |  |  | Robust z-score (median + MAD) | `pyimgano/models/rzscore.py` |
| `core_sampling` | classical, core, features, sampling, distance | 2013 |  | Core sampling-based distance outlier detector on feature matrices (native wrapper) | `pyimgano/models/sampling.py` |
| `core_score_ensemble` | classical, core, features, ensemble, score |  |  | Score-only ensemble wrapper for feature-matrix detectors (spec-friendly) | `pyimgano/models/score_ensemble.py` |
| `core_score_standardizer` | classical, core, features, wrapper, calibration |  |  | Wrap a core detector and standardize its scores (rank/zscore/robust/minmax) | `pyimgano/models/core_score_standardizer.py` |
| `core_sod` | classical, core, features, sod, subspace |  |  | SOD (Subspace Outlier Detection) for feature matrices (native wrapper) | `pyimgano/models/sod.py` |
| `core_sos` | classical, core, features, sos, probabilistic |  |  | SOS (Stochastic Outlier Selection) for feature matrices (native wrapper) | `pyimgano/models/sos.py` |
| `core_studentized_residual` | classical, core, features, pca, residual |  |  | PCA reconstruction residual standardized by median+MAD (studentized residual baseline) | `pyimgano/models/studentized_residual.py` |
| `core_suod` | classical, core, features, ensemble, suod | 2021 |  | Core SUOD-style score ensemble on feature matrices (native wrapper) | `pyimgano/models/suod.py` |
| `core_suod_spec` | classical, core, features, ensemble, suod | 2021 |  | Core SUOD-style ensemble with JSON-friendly base-estimator specs | `pyimgano/models/suod.py` |
| `core_torch_autoencoder` | deep, core, features, torch, autoencoder, reconstruction |  |  | Core torch MLP autoencoder on feature matrices (reconstruction error) | `pyimgano/models/torch_autoencoder.py` |
| `cutpaste` | vision, deep, cutpaste, self-supervised, cvpr2021 | 2021 |  | CutPaste (legacy alias) - self-supervised anomaly detection via synthetic cut/paste | `pyimgano/models/cutpaste.py` |
| `dbscan_anomaly` | vision, classical, clustering, dbscan, structural |  |  | Structural-features DBSCAN-inspired anomaly baseline (modernized) | `pyimgano/models/dbscan.py` |
| `devnet` | vision, deep, devnet, weakly-supervised, kdd2019 | 2019 |  | DevNet (legacy alias) - weakly-supervised deviation loss anomaly detection | `pyimgano/models/devnet.py` |
| `differnet` | vision, deep, differnet, knn, wacv2023, pixel_map | 2023 |  | DifferNet (legacy alias) - learnable difference + kNN anomaly detection | `pyimgano/models/differnet.py` |
| `efficient_ad` | vision, deep, distillation |  |  | EfficientAD-lite: teacher/student embedding distillation (contract-aligned) | `pyimgano/models/efficientad.py` |
| `isolation_forest_struct` | vision, classical, iforest, ensemble, structural |  |  | Structural-features Isolation Forest baseline (modernized; native base classes) | `pyimgano/models/Isolationforest.py` |
| `kmeans_anomaly` | vision, classical, clustering, kmeans, structural |  |  | Structural-features KMeans anomaly baseline (modernized) | `pyimgano/models/k_means.py` |
| `lof_structure` | vision, classical, lof, neighbors, structural |  |  | Structural-features LOF anomaly detector (modernized; native base classes) | `pyimgano/models/lof.py` |
| `memseg` | vision, deep, memseg, memory, segmentation, pixel_map | 2022 |  | MemSeg (legacy alias) - memory-guided anomaly segmentation | `pyimgano/models/memseg.py` |
| `one_class_cnn` | vision, classical, svm |  |  | 基于多特征的一类 SVM 图像检测器 | `pyimgano/models/one_svm_cnn.py` |
| `padim` | vision, deep, patch, distribution, numpy, pixel_map | 2020 |  | PaDiM (legacy alias) - patch distribution modeling | `pyimgano/models/padim.py` |
| `riad` | vision, deep, riad, reconstruction, self-supervised, pixel_map | 2020 |  | RIAD (legacy alias) - reconstruction by adjacent image decomposition | `pyimgano/models/riad.py` |
| `spade` | vision, deep, spade, knn, numpy, pixel_map | 2020 |  | SPADE (legacy alias) - Deep pyramid k-NN localization | `pyimgano/models/spade.py` |
| `ssim_struct` | vision, classical, template, ssim, structural |  |  | Structural SSIM template-match baseline (edges; modernized) | `pyimgano/models/ssim_struct.py` |
| `ssim_struct_map` | vision, classical, template, ssim, structural, pixel_map |  |  | Structural SSIM (edges) with pixel anomaly maps | `pyimgano/models/ssim_map.py` |
| `ssim_template` | vision, classical, template, ssim |  |  | SSIM template-match baseline (modernized; native BaseDetector contract) | `pyimgano/models/ssim.py` |
| `ssim_template_map` | vision, classical, template, ssim, pixel_map |  |  | SSIM template detector with pixel anomaly maps (1 - SSIM map) | `pyimgano/models/ssim_map.py` |
| `vae_conv` | vision, deep, vae, reconstruction |  |  | Convolutional VAE reconstruction baseline (contract-aligned) | `pyimgano/models/vae.py` |
| `vision_abod` | vision, classical, abod |  |  | 基于 ABOD 的视觉异常检测器 (native) | `pyimgano/models/abod.py` |
| `vision_ae1svm` | vision, deep, svm |  |  | 自编码器 + 一类 SVM 组合的视觉检测器 | `pyimgano/models/ae1svm.py` |
| `vision_alad` | vision, deep, gan |  |  | Adversarially Learned Anomaly Detection | `pyimgano/models/alad.py` |
| `vision_anomalib_checkpoint` | vision, deep, backend, anomalib |  | anomalib | Generic anomalib checkpoint inferencer wrapper (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_anomalydino` | vision, deep, anomalydino, knn, dinov2, numpy, pixel_map |  |  | AnomalyDINO-style DINOv2 patch-kNN detector (few-shot friendly) | `pyimgano/models/anomalydino.py` |
| `vision_ast` | vision, deep, ast, student-teacher, anomaly-aware, sota | 2023 |  | AST - Anomaly-aware Student-Teacher with synthetic anomalies | `pyimgano/models/ast.py` |
| `vision_bayesianpf` | vision, deep, bayesianpf, zero-shot, bayesian, cvpr2025, sota | 2025 |  | BayesianPF - Bayesian Prompt Flow for Zero-Shot AD (CVPR 2025) | `pyimgano/models/bayesianpf.py` |
| `vision_cblof` | vision, classical, clustering |  |  | 基于 CBLOF 的视觉异常检测器 | `pyimgano/models/cblof.py` |
| `vision_cfa_anomalib` | vision, deep, backend, anomalib, cfa |  | anomalib | CFA via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cflow` | vision, deep, cflow, normalizing-flow, real-time | 2022 |  | CFlow-AD - Conditional normalizing flows (WACV 2022) | `pyimgano/models/cflow.py` |
| `vision_cflow_anomalib` | vision, deep, backend, anomalib, cflow |  | anomalib | CFlow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cof` | vision, classical, neighbors, cof | 2002 |  | COF - Connectivity-based outlier detector (native) | `pyimgano/models/cof.py` |
| `vision_cook_distance` | vision, classical, pca, influence |  |  | Vision wrapper for Cook-distance-inspired influence score | `pyimgano/models/cook_distance.py` |
| `vision_copod` | vision, classical, copod, parameter-free, high-performance | 2020 |  | COPOD - Copula-based outlier detector (ICDM 2020) | `pyimgano/models/copod.py` |
| `vision_crossmad` | vision, classical, crossmad, prototype, embeddings, cvpr2025 | 2025 |  | Vision CrossMAD-style prototype-distance detector (embeddings + core_crossmad) | `pyimgano/models/crossmad.py` |
| `vision_csflow_anomalib` | vision, deep, backend, anomalib, csflow |  | anomalib | CS-Flow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_cutpaste` | vision, deep, cutpaste, self-supervised, cvpr2021 | 2021 |  | CutPaste - self-supervised anomaly detection via synthetic cut/paste (CVPR 2021) | `pyimgano/models/cutpaste.py` |
| `vision_dbscan` | vision, classical, clustering, dbscan, density |  |  | Vision wrapper for DBSCAN-inspired distance-to-core-set baseline | `pyimgano/models/dbscan.py` |
| `vision_dcorr` | vision, classical, projection, dependence |  |  | Vision distance-correlation influence detector | `pyimgano/models/dcorr.py` |
| `vision_deep_svdd` | vision, deep, torch |  |  | 基于 DeepSVDD 的视觉异常检测器 | `pyimgano/models/deep_svdd.py` |
| `vision_devnet` | vision, deep, devnet, weakly-supervised, kdd2019 | 2019 |  | DevNet - weakly-supervised deviation loss anomaly detection (KDD 2019) | `pyimgano/models/devnet.py` |
| `vision_dfkde_anomalib` | vision, deep, backend, anomalib, dfkde |  | anomalib | DFKDE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dfm` | vision, deep, dfm, fast, gaussian |  |  | DFM - Fast discriminative feature modeling | `pyimgano/models/dfm.py` |
| `vision_dfm_anomalib` | vision, deep, backend, anomalib, dfm |  | anomalib | DFM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_differnet` | vision, deep, differnet, knn, wacv2023, pixel_map | 2023 |  | DifferNet - learnable difference + kNN anomaly detection (WACV 2023) | `pyimgano/models/differnet.py` |
| `vision_dinomaly_anomalib` | vision, deep, backend, anomalib, dinomaly |  | anomalib | Dinomaly via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_draem` | vision, deep, draem, reconstruction, synthetic, numpy, pixel_map | 2021 |  | DRAEM - Discriminatively trained reconstruction (ICCV 2021) | `pyimgano/models/draem.py` |
| `vision_draem_anomalib` | vision, deep, backend, anomalib, draem |  | anomalib | DRAEM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dsr_anomalib` | vision, deep, backend, anomalib, dsr |  | anomalib | DSR via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_dst` | vision, deep, dst, student-teacher, distillation, sota | 2023 |  | DST - Double Student-Teacher with complementary learning | `pyimgano/models/dst.py` |
| `vision_dtc` | vision, classical, distance, baseline |  |  | Vision distance-to-centroid baseline | `pyimgano/models/dtc.py` |
| `vision_ecod` | vision, classical, ecod, parameter-free, high-performance | 2022 |  | ECOD - Empirical CDF-based outlier detector (TKDE 2022) | `pyimgano/models/ecod.py` |
| `vision_efficientad_anomalib` | vision, deep, backend, anomalib, efficientad |  | anomalib | EfficientAD via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_elliptic_envelope` | vision, classical, gaussian, covariance |  |  | Vision wrapper for robust covariance / Mahalanobis-distance baseline | `pyimgano/models/elliptic_envelope.py` |
| `vision_embedding_core` | vision, classical, pipeline, embeddings |  |  | Embedding extractor + core detector pipeline (industrial baseline) | `pyimgano/models/vision_embedding_core.py` |
| `vision_embedding_torch_autoencoder` | vision, deep, torch, autoencoder, embeddings, pipeline |  |  | Industrial preset: deep embeddings (torchvision_backbone) -> torch MLP autoencoder -> recon error | `pyimgano/models/torch_autoencoder.py` |
| `vision_extra_trees_density` | vision, classical, trees, density |  |  | Vision wrapper for random-trees leaf-rarity density baseline | `pyimgano/models/extra_trees_density.py` |
| `vision_fastflow` | vision, deep, flow |  |  | FastFlow-based visual anomaly detector | `pyimgano/models/fastflow.py` |
| `vision_fastflow_anomalib` | vision, deep, backend, anomalib, fastflow |  | anomalib | FastFlow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_favae` | vision, deep, favae, vae, adaptive, sota | 2023 |  | Feature Adaptive VAE - Dynamic latent space adaptation | `pyimgano/models/favae.py` |
| `vision_feature_bagging` | vision, ensemble, feature_bagging | 2005 |  | Feature Bagging - random feature-subspace ensemble (native) | `pyimgano/models/feature_bagging.py` |
| `vision_feature_bagging_spec` | vision, ensemble, feature_bagging, spec | 2005 |  | Feature Bagging ensemble with JSON-friendly base-estimator spec | `pyimgano/models/feature_bagging.py` |
| `vision_feature_pipeline` | vision, classical, pipeline |  |  | Feature extractor + core detector pipeline (dynamic vision wrapper). | `pyimgano/models/feature_pipeline.py` |
| `vision_fre_anomalib` | vision, deep, backend, anomalib, fre |  | anomalib | FRE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_ganomaly_anomalib` | vision, deep, backend, anomalib, ganomaly |  | anomalib | GANomaly via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_gcad` | vision, deep, gcad, graph, sota | 2023 |  | Graph Convolutional Anomaly Detection - Uses GCN to model spatial relationships | `pyimgano/models/gcad.py` |
| `vision_glad` | vision, deep, glad, diffusion, adaptive, eccv2024, sota | 2024 |  | GLAD - Global-Local Adaptive Diffusion (ECCV 2024) | `pyimgano/models/glad.py` |
| `vision_gmm` | vision, classical, gmm, density, baseline |  |  | Gaussian Mixture Model detector (density baseline) | `pyimgano/models/gmm.py` |
| `vision_hbos` | vision, classical, hbos, histogram, fast |  |  | HBOS - Histogram-based Outlier Score (fast, interpretable baseline) | `pyimgano/models/hbos.py` |
| `vision_hst` | vision, classical, tree, online |  |  | Vision Half-Space Trees (leaf-mass scoring) | `pyimgano/models/hst.py` |
| `vision_iforest` | vision, classical, iforest, ensemble, baseline | 2008 |  | Isolation Forest detector (baseline, robust general-purpose) | `pyimgano/models/iforest.py` |
| `vision_imdd` | vision, classical |  |  | Vision wrapper for IMDD deviation detector | `pyimgano/models/imdd.py` |
| `vision_inctrl` | vision, deep, inctrl, few-shot, generalist, cvpr2024, sota | 2024 |  | InCTRL - In-context Residual Learning for generalist AD (CVPR 2024) | `pyimgano/models/inctrl.py` |
| `vision_inne` | vision, classical, isolation, inne, fast | 2014 |  | INNE - Isolation using Nearest-Neighbor Ensembles (ICDMW 2014) | `pyimgano/models/inne.py` |
| `vision_kde` | vision, classical, kde, density, baseline |  |  | Kernel Density Estimation detector (density baseline) | `pyimgano/models/kde.py` |
| `vision_kde_ratio` | vision, classical, density, kde |  |  | Vision wrapper for dual-bandwidth KDE density-contrast baseline | `pyimgano/models/kde_ratio.py` |
| `vision_kmeans` | vision, classical, clustering, kmeans |  |  | Vision wrapper for KMeans distance-to-centroid baseline | `pyimgano/models/k_means.py` |
| `vision_knn` | vision, classical, neighbors, knn | 2000 |  | Vision wrapper for KNN outlier detector | `pyimgano/models/knn.py` |
| `vision_knn_degree` | vision, classical, neighbors, graph, density |  |  | Vision kNN epsilon-graph degree | `pyimgano/models/knn_degree.py` |
| `vision_kpca` | vision, classical, kernel |  |  | 基于 Kernel PCA 的视觉异常检测器 | `pyimgano/models/kpca.py` |
| `vision_ldof` | vision, classical, neighbors, local |  |  | Vision LDOF - Local Distance-based Outlier Factor | `pyimgano/models/ldof.py` |
| `vision_lid` | vision, classical, neighbors, lid |  |  | Vision wrapper for LID kNN-distance outlier score | `pyimgano/models/lid.py` |
| `vision_lmdd` | vision, classical, lmdd, baseline |  |  | LMDD deviation detector (native implementation) | `pyimgano/models/lmdd.py` |
| `vision_loci` | vision, classical, loci |  |  | Vision wrapper for LOCI outlier detector (native) | `pyimgano/models/loci.py` |
| `vision_loda` | vision, classical |  |  | 基于 LODA 的视觉异常检测器 | `pyimgano/models/loda.py` |
| `vision_lof` | vision, classical, lof, neighbors, density |  |  | Vision wrapper for LOF (Local Outlier Factor, novelty mode) | `pyimgano/models/lof_native.py` |
| `vision_loop` | vision, classical, neighbors, probability |  |  | Vision LoOP - Local Outlier Probability | `pyimgano/models/loop.py` |
| `vision_lscp` | vision, classical, ensemble, lscp |  |  | LSCP - locally selective combination ensemble (native) | `pyimgano/models/lscp.py` |
| `vision_lscp_spec` | vision, classical, ensemble, lscp |  |  | LSCP ensemble with JSON-friendly base-detector specs | `pyimgano/models/lscp.py` |
| `vision_mad` | vision, classical, mad, robust, baseline |  |  | Multivariate MAD robust baseline (median + MAD robust z-score) | `pyimgano/models/mad.py` |
| `vision_mahalanobis` | vision, classical, distance, gaussian |  |  | Vision Mahalanobis baseline (mean + covariance) | `pyimgano/models/mahalanobis.py` |
| `vision_mcd` | vision, classical, statistical, mcd, robust | 1999 |  | MCD - Robust covariance-based outlier detector (MinCovDet backend) | `pyimgano/models/mcd.py` |
| `vision_memseg` | vision, deep, memseg, memory, segmentation, pixel_map | 2022 |  | MemSeg - memory-guided anomaly segmentation (ICCV 2022-style) | `pyimgano/models/memseg.py` |
| `vision_mst_outlier` | vision, classical, graph, mst |  |  | Vision wrapper for MST-based outlier detector | `pyimgano/models/mst_outlier.py` |
| `vision_neighborhood_entropy` | vision, classical, neighbors, graph, entropy |  |  | Vision wrapper for neighborhood entropy score | `pyimgano/models/neighborhood_entropy.py` |
| `vision_ocsvm` | vision, classical, svm, one-class, ocsvm |  |  | One-Class SVM outlier detector (sklearn backend) | `pyimgano/models/ocsvm.py` |
| `vision_oddoneout` | vision, classical, neighbors, oddoneout, cvpr2025, embeddings |  |  | Odd-One-Out neighbor comparison (CVPR 2025-inspired) on deep embeddings | `pyimgano/models/oddoneout.py` |
| `vision_odin` | vision, classical, neighbors, graph |  |  | Vision ODIN - indegree-based kNN graph detector | `pyimgano/models/odin.py` |
| `vision_oneformore` | vision, deep, oneformore, continual, diffusion, cvpr2025, sota | 2025 |  | One-for-More - Continual Diffusion Model (CVPR 2025, #1 on MVTec/VisA) | `pyimgano/models/oneformore.py` |
| `vision_onnx_copod` | vision, classical, pipeline, industrial, embeddings, onnx, fast, parameter-free |  |  | Industrial baseline: ONNX Runtime embeddings + core_copod | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_cosine_mahalanobis` | vision, classical, pipeline, industrial, embeddings, onnx, distance, gaussian, cosine |  |  | Industrial baseline: ONNX Runtime embeddings + core_cosine_mahalanobis | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_ecod` | vision, classical, pipeline, industrial, embeddings, onnx, fast |  |  | Industrial baseline: ONNX Runtime embeddings + core_ecod | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_extra_trees_density` | vision, classical, pipeline, industrial, embeddings, onnx, trees, density |  |  | Industrial baseline: ONNX Runtime embeddings + core_extra_trees_density | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_iforest` | vision, classical, pipeline, industrial, embeddings, onnx, baseline |  |  | Industrial baseline: ONNX Runtime embeddings + core_iforest | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_knn_cosine` | vision, classical, pipeline, industrial, embeddings, onnx, neighbors |  |  | Industrial baseline: ONNX Runtime embeddings + core_knn_cosine | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_knn_cosine_calibrated` | vision, classical, pipeline, industrial, embeddings, onnx, neighbors, calibration |  |  | Industrial baseline: ONNX Runtime embeddings + core_knn_cosine_calibrated | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_lid` | vision, classical, pipeline, industrial, embeddings, onnx, neighbors, lid |  |  | Industrial baseline: ONNX Runtime embeddings + core_lid | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_lof` | vision, classical, pipeline, industrial, embeddings, onnx, neighbors, density, lof |  |  | Industrial baseline: ONNX Runtime embeddings + core_lof | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_mcd` | vision, classical, pipeline, industrial, embeddings, onnx, gaussian, robust, mcd |  |  | Industrial baseline: ONNX Runtime embeddings + core_mcd | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_mst_outlier` | vision, classical, pipeline, industrial, embeddings, onnx, graph, mst |  |  | Industrial baseline: ONNX Runtime embeddings + core_mst_outlier | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_oddoneout` | vision, classical, pipeline, industrial, embeddings, onnx, neighbors, oddoneout |  |  | Industrial baseline: ONNX Runtime embeddings + core_oddoneout | `pyimgano/models/industrial_wrappers.py` |
| `vision_onnx_pca_md` | vision, classical, pipeline, industrial, embeddings, onnx, pca, distance |  |  | Industrial baseline: ONNX Runtime embeddings + core_pca_md | `pyimgano/models/industrial_wrappers.py` |
| `vision_openclip_patch_map` | vision, deep, clip, openclip, backend, pixel_map |  | openclip | OpenCLIP patch template distance anomaly map (requires pyimgano[clip]) | `pyimgano/models/openclip_patch_map.py` |
| `vision_openclip_patchknn` | vision, deep, clip, openclip, backend, knn |  | openclip | OpenCLIP patch embedding + kNN detector (requires pyimgano[clip]) | `pyimgano/models/openclip_backend.py` |
| `vision_openclip_promptscore` | vision, deep, clip, openclip, backend, prompt |  | openclip | OpenCLIP prompt scoring detector (requires pyimgano[clip]) | `pyimgano/models/openclip_backend.py` |
| `vision_padim` | vision, deep, patch, distribution, numpy, pixel_map | 2020 |  | PaDiM - patch distribution modeling (ECCV 2020-style) | `pyimgano/models/padim.py` |
| `vision_padim_anomalib` | vision, deep, backend, anomalib, padim |  | anomalib | PaDiM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_padim_lite` | vision, classical, padim, gaussian |  |  | PaDiM-lite: embedding extractor + robust covariance baseline (image-level) | `pyimgano/models/padim_lite.py` |
| `vision_panda` | vision, deep, panda, prototypical, metric, sota | 2023 |  | PANDA - Prototypical Anomaly Network with metric learning | `pyimgano/models/panda.py` |
| `vision_patch_embedding_core_map` | vision, classical, pipeline, embeddings, patch, pixel_map |  |  | Patch embeddings + core detector anomaly map (generic industrial baseline) | `pyimgano/models/patch_embedding_core_map.py` |
| `vision_patchcore` | vision, deep, patchcore, sota, cvpr2022, numpy, pixel_map | 2022 |  | PatchCore - SOTA patch-level anomaly detection (CVPR 2022) | `pyimgano/models/patchcore.py` |
| `vision_patchcore_anomalib` | vision, deep, backend, anomalib, patchcore |  | anomalib | PatchCore via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_patchcore_inspection_checkpoint` | vision, deep, backend, patchcore_inspection, patchcore |  | patchcore_inspection | PatchCore (amazon-science/patchcore-inspection) checkpoint wrapper (optional backend) | `pyimgano/models/patchcore_inspection_backend.py` |
| `vision_patchcore_lite` | vision, classical, neighbors, memory_bank, patchcore |  |  | PatchCore-lite: embedding extractor + coreset memory bank + NN distance (image-level) | `pyimgano/models/patchcore_lite.py` |
| `vision_patchcore_lite_map` | vision, deep, patchcore, lite, patchknn, numpy, pixel_map | 2022 |  | PatchCore-lite anomaly map: conv patch embeddings + memory bank kNN distance | `pyimgano/models/patchcore_lite_map.py` |
| `vision_patchcore_online` | vision, classical, neighbors, memory_bank, patchcore, online |  |  | PatchCore-online: feature extractor + incremental memory bank + NN distance (image-level) | `pyimgano/models/patchcore_online.py` |
| `vision_pca` | vision, classical, linear, pca | 2003 |  | Vision wrapper for PCA-based outlier detector | `pyimgano/models/pca.py` |
| `vision_pca_md` | vision, classical, pca, distance |  |  | Vision PCA + Mahalanobis distance (subspace) | `pyimgano/models/pca_md.py` |
| `vision_phase_correlation_map` | vision, classical, template, phase_corr, pixel_map |  |  | Template baseline using phase correlation alignment + abs-diff anomaly map | `pyimgano/models/phase_corr_map.py` |
| `vision_pixel_gaussian_map` | vision, classical, template, pixel_stats, gaussian, numpy, pixel_map |  |  | Per-pixel Gaussian baseline (mean+std) anomaly map via z-score | `pyimgano/models/pixel_stats_map.py` |
| `vision_pixel_mad_map` | vision, classical, template, pixel_stats, robust, numpy, pixel_map |  |  | Per-pixel robust MAD baseline anomaly map (median + MAD z-score) | `pyimgano/models/pixel_stats_map.py` |
| `vision_pixel_mean_absdiff_map` | vision, classical, template, pixel_stats, numpy, pixel_map |  |  | Per-pixel mean template abs-diff anomaly map (fast aligned baseline) | `pyimgano/models/pixel_stats_map.py` |
| `vision_promptad` | vision, deep, promptad, few-shot, prompt, cvpr2024, sota | 2024 |  | PromptAD - Prompt learning with only normal samples (CVPR 2024) | `pyimgano/models/promptad.py` |
| `vision_qmcd` | vision, classical, qmcd, robust, baseline |  |  | QMCD wrap-around discrepancy detector (robust-statistical baseline) | `pyimgano/models/qmcd.py` |
| `vision_random_projection_knn` | vision, classical, neighbors, knn, projection |  |  | Vision wrapper for random projection + kNN detector | `pyimgano/models/random_projection_knn.py` |
| `vision_realnet` | vision, deep, realnet, feature-selection, cvpr2024, sota | 2024 |  | RealNet - Feature Selection with Realistic Synthetic Anomaly (CVPR 2024) | `pyimgano/models/realnet.py` |
| `vision_ref_patch_distance_map` | vision, deep, torch, torchvision, pixel_map, reference |  |  | Reference-based patch distance anomaly map (torchvision feature map) | `pyimgano/models/ref_patch_distance.py` |
| `vision_regad` | vision, deep, regad, registration, alignment, sota | 2023 |  | RegAD - Registration-based anomaly detection with STN | `pyimgano/models/regad.py` |
| `vision_resnet18_copod` | vision, classical, pipeline, industrial, embeddings, fast, parameter-free |  |  | Industrial baseline: resnet18 embeddings (safe) + core_copod | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_cosine_mahalanobis` | vision, classical, pipeline, industrial, embeddings, distance, gaussian, cosine |  |  | Industrial baseline: resnet18 embeddings (safe) + core_cosine_mahalanobis | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_ecod` | vision, classical, pipeline, industrial, embeddings, fast |  |  | Industrial baseline: resnet18 embeddings (safe) + core_ecod | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_extra_trees_density` | vision, classical, pipeline, industrial, embeddings, trees, density |  |  | Industrial baseline: resnet18 embeddings (safe) + core_extra_trees_density | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_iforest` | vision, classical, pipeline, industrial, embeddings, baseline |  |  | Industrial baseline: resnet18 embeddings (safe) + core_iforest | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_knn` | vision, classical, pipeline, industrial, embeddings, neighbors |  |  | Industrial baseline: resnet18 embeddings (safe) + core_knn | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_knn_cosine` | vision, classical, pipeline, industrial, embeddings, neighbors, cosine |  |  | Industrial baseline: resnet18 embeddings (safe) + core_knn_cosine | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_knn_cosine_calibrated` | vision, classical, pipeline, industrial, embeddings, neighbors, cosine, calibration |  |  | Industrial baseline: resnet18 embeddings (safe) + core_knn_cosine_calibrated | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_lid` | vision, classical, pipeline, industrial, embeddings, neighbors, lid |  |  | Industrial baseline: resnet18 embeddings (safe) + core_lid | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_lof` | vision, classical, pipeline, industrial, embeddings, neighbors, density, lof |  |  | Industrial baseline: resnet18 embeddings (safe) + core_lof | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_mahalanobis_shrinkage` | vision, classical, pipeline, industrial, embeddings, distance, gaussian, shrinkage |  |  | Industrial baseline: resnet18 embeddings (safe) + core_mahalanobis_shrinkage | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_mcd` | vision, classical, pipeline, industrial, embeddings, gaussian, robust, mcd |  |  | Industrial baseline: resnet18 embeddings (safe) + core_mcd (robust covariance) | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_mst_outlier` | vision, classical, pipeline, industrial, embeddings, graph, mst |  |  | Industrial baseline: resnet18 embeddings (safe) + core_mst_outlier | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_oddoneout` | vision, classical, pipeline, industrial, embeddings, neighbors, oddoneout |  |  | Industrial baseline: resnet18 embeddings (safe) + core_oddoneout | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_pca_md` | vision, classical, pipeline, industrial, embeddings, pca, distance |  |  | Industrial baseline: resnet18 embeddings (safe) + core_pca_md | `pyimgano/models/industrial_wrappers.py` |
| `vision_resnet18_torch_ae` | vision, deep, pipeline, industrial, embeddings, reconstruction |  |  | Industrial baseline: resnet18 embeddings (safe) + core_torch_autoencoder | `pyimgano/models/industrial_wrappers.py` |
| `vision_reverse_dist` | vision, deep, distillation |  |  | Reverse distillation anomaly detector (alias) | `pyimgano/models/reverse_distillation.py` |
| `vision_reverse_distillation` | vision, deep, distillation |  |  | Reverse distillation anomaly detector | `pyimgano/models/reverse_distillation.py` |
| `vision_reverse_distillation_anomalib` | vision, deep, backend, anomalib, reverse_distillation |  | anomalib | Reverse Distillation via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_rgraph` | vision, classical, rgraph, graph |  |  | Graph random-walk outlier detector (native, simplified) | `pyimgano/models/rgraph.py` |
| `vision_riad` | vision, deep, riad, reconstruction, self-supervised, pixel_map | 2020 |  | RIAD - reconstruction by adjacent image decomposition (2020-style) | `pyimgano/models/riad.py` |
| `vision_rkde_anomalib` | vision, deep, backend, anomalib, rkde |  | anomalib | R-KDE via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_rod` | vision, classical, rod, baseline |  |  | Rotation-based Outlier Detection (baseline) | `pyimgano/models/rod.py` |
| `vision_rrcf` | vision, classical, forest, random-cut |  |  | Vision random cut forest baseline (RRCF-style) | `pyimgano/models/rrcf.py` |
| `vision_rzscore` | vision, classical, robust, baseline |  |  | Vision robust z-score (median + MAD) | `pyimgano/models/rzscore.py` |
| `vision_sampling` | vision, classical, sampling, distance |  |  | Sampling-based distance outlier detector (native) | `pyimgano/models/sampling.py` |
| `vision_score_ensemble` | vision, ensemble, score |  |  | Score-only ensemble wrapper (mean of rank-normalized scores by default) | `pyimgano/models/score_ensemble.py` |
| `vision_score_standardizer` | vision, wrapper, calibration, score |  |  | Wrap a vision detector and standardize its scores (rank/zscore/robust/minmax) | `pyimgano/models/vision_score_standardizer.py` |
| `vision_simplenet` | vision, deep, simplenet, fast, sota, cvpr2023 | 2023 |  | SimpleNet - Ultra-fast SOTA anomaly detection (CVPR 2023) | `pyimgano/models/simplenet.py` |
| `vision_sod` | vision, classical, sod, subspace, baseline |  |  | Subspace Outlier Detection (subspace baseline) | `pyimgano/models/sod.py` |
| `vision_softpatch` | vision, deep, softpatch, patchknn, robust, numpy, pixel_map |  |  | SoftPatch-inspired robust patch-memory detector (few-shot friendly) | `pyimgano/models/softpatch.py` |
| `vision_sos` | vision, classical, sos, probabilistic, baseline |  |  | Stochastic Outlier Selection (probabilistic baseline) | `pyimgano/models/sos.py` |
| `vision_spade` | vision, deep, spade, knn, numpy, pixel_map | 2020 |  | SPADE - Deep pyramid k-NN localization (ECCV 2020) | `pyimgano/models/spade.py` |
| `vision_stfpm` | vision, deep, stfpm, student-teacher, pyramid, numpy, pixel_map | 2021 |  | STFPM - Student-Teacher Feature Pyramid Matching (BMVC 2021) | `pyimgano/models/stfpm.py` |
| `vision_stfpm_anomalib` | vision, deep, backend, anomalib, stfpm |  | anomalib | STFPM via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_structural_copod` | vision, classical, pipeline, industrial, fast, parameter-free |  |  | Industrial baseline: structural features + core_copod | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_ecod` | vision, classical, pipeline, industrial, fast |  |  | Industrial baseline: structural features + core_ecod | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_extra_trees_density` | vision, classical, pipeline, industrial, trees, density |  |  | Industrial baseline: structural features + core_extra_trees_density | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_iforest` | vision, classical, pipeline, industrial, baseline |  |  | Industrial baseline: structural features + core_iforest | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_knn` | vision, classical, pipeline, industrial, neighbors |  |  | Industrial baseline: structural features + core_knn | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_lid` | vision, classical, pipeline, industrial, neighbors |  |  | Structural features + core_lid (kNN distance statistic) | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_lof` | vision, classical, pipeline, industrial, neighbors, density |  |  | Industrial baseline: structural features + core_lof | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_mcd` | vision, classical, pipeline, industrial, gaussian, robust |  |  | Industrial baseline: structural features + core_mcd (robust covariance) | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_mst_outlier` | vision, classical, pipeline, industrial, graph |  |  | Structural features + core_mst_outlier (MST baseline) | `pyimgano/models/industrial_wrappers.py` |
| `vision_structural_pca_md` | vision, classical, pipeline, industrial, pca, distance |  |  | Industrial baseline: structural features + core_pca_md (subspace MD) | `pyimgano/models/industrial_wrappers.py` |
| `vision_student_teacher_lite` | vision, classical, embeddings, student_teacher |  |  | Student-Teacher lite: linear map residual between two embedding extractors | `pyimgano/models/student_teacher_lite.py` |
| `vision_studentized_residual` | vision, classical, pca, residual |  |  | Vision wrapper for studentized PCA residual baseline | `pyimgano/models/studentized_residual.py` |
| `vision_suod` | vision, classical, ensemble, suod |  |  | SUOD-style score ensemble (native, simplified) | `pyimgano/models/suod.py` |
| `vision_suod_spec` | vision, classical, ensemble, suod |  |  | SUOD-style ensemble with JSON-friendly base-estimator specs | `pyimgano/models/suod.py` |
| `vision_superad` | vision, deep, superad, knn, dinov2, numpy, pixel_map |  |  | SuperAD-style DINOv2 patch-kNN detector using k-th NN distance per patch | `pyimgano/models/superad.py` |
| `vision_supersimplenet_anomalib` | vision, deep, backend, anomalib, supersimplenet |  | anomalib | SuperSimpleNet via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_template_ncc_map` | vision, classical, template, ncc, pixel_map |  |  | Template baseline using local NCC similarity → pixel anomaly map | `pyimgano/models/template_ncc_map.py` |
| `vision_torch_autoencoder` | vision, deep, torch, autoencoder, reconstruction |  |  | Vision wrapper for core_torch_autoencoder (feature extractor + torch AE core) | `pyimgano/models/torch_autoencoder.py` |
| `vision_torchscript_copod` | vision, classical, pipeline, industrial, embeddings, torchscript, fast, parameter-free |  |  | Industrial baseline: TorchScript embeddings + core_copod | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_cosine_mahalanobis` | vision, classical, pipeline, industrial, embeddings, torchscript, distance, gaussian, cosine |  |  | Industrial baseline: TorchScript embeddings + core_cosine_mahalanobis | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_ecod` | vision, classical, pipeline, industrial, embeddings, torchscript, fast |  |  | Industrial baseline: TorchScript embeddings + core_ecod | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_extra_trees_density` | vision, classical, pipeline, industrial, embeddings, torchscript, trees, density |  |  | Industrial baseline: TorchScript embeddings + core_extra_trees_density | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_iforest` | vision, classical, pipeline, industrial, embeddings, torchscript, baseline |  |  | Industrial baseline: TorchScript embeddings + core_iforest | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_knn_cosine` | vision, classical, pipeline, industrial, embeddings, torchscript, neighbors |  |  | Industrial baseline: TorchScript embeddings + core_knn_cosine | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_knn_cosine_calibrated` | vision, classical, pipeline, industrial, embeddings, torchscript, neighbors, calibration |  |  | Industrial baseline: TorchScript embeddings + core_knn_cosine_calibrated | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_lid` | vision, classical, pipeline, industrial, embeddings, torchscript, neighbors, lid |  |  | Industrial baseline: TorchScript embeddings + core_lid | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_lof` | vision, classical, pipeline, industrial, embeddings, torchscript, neighbors, density, lof |  |  | Industrial baseline: TorchScript embeddings + core_lof | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_mcd` | vision, classical, pipeline, industrial, embeddings, torchscript, gaussian, robust, mcd |  |  | Industrial baseline: TorchScript embeddings + core_mcd (robust covariance) | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_mst_outlier` | vision, classical, pipeline, industrial, embeddings, torchscript, graph, mst |  |  | Industrial baseline: TorchScript embeddings + core_mst_outlier | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_oddoneout` | vision, classical, pipeline, industrial, embeddings, torchscript, neighbors, oddoneout |  |  | Industrial baseline: TorchScript embeddings + core_oddoneout | `pyimgano/models/industrial_wrappers.py` |
| `vision_torchscript_pca_md` | vision, classical, pipeline, industrial, embeddings, torchscript, pca, distance |  |  | Industrial baseline: TorchScript embeddings + core_pca_md | `pyimgano/models/industrial_wrappers.py` |
| `vision_uflow_anomalib` | vision, deep, backend, anomalib, uflow |  | anomalib | U-Flow via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_vlmad_anomalib` | vision, deep, backend, anomalib, vlmad |  | anomalib | VLM-AD via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vision_winclip` | vision, deep, winclip, clip, pixel_map | 2023 |  | WinCLIP - Zero-/Few-shot CLIP-based anomaly detection (CVPR 2023) | `pyimgano/models/winclip.py` |
| `vision_winclip_anomalib` | vision, deep, backend, anomalib, winclip |  | anomalib | WinCLIP via anomalib backend (requires pyimgano[anomalib]) | `pyimgano/models/anomalib_backend.py` |
| `vqvae_conv` | vision, deep, vqvae, reconstruction |  |  | Convolutional VQ-VAE reconstruction baseline (tiny-capable) | `pyimgano/models/vqvae.py` |
| `winclip` | vision, deep, winclip, clip, pixel_map | 2023 |  | WinCLIP (legacy name) - Zero-/Few-shot CLIP-based anomaly detection | `pyimgano/models/winclip.py` |
