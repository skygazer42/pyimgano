# Industrial Reference Projects (Research Index)

This document tracks **external industrial anomaly detection / inspection projects** that are useful
to study for architecture ideas, evaluation patterns, preprocessing pipelines, and augmentation /
synthetic anomaly generation.

Policy for `pyimgano`:
- We may **clone** reference repos locally to learn patterns, then re-implement ideas using
  `pyimgano` base classes (`BaseDetector`, `BaseVisionDetector`) and our feature/preprocessing APIs.
- Copying code is allowed **only when the upstream license is compatible** and we preserve the
  required notices. Prefer re-implementation unless copying clearly reduces risk/bugs.
- Any optional integration must be guarded behind optional imports and should **not** add new
  heavy runtime dependencies.

---

## What We Look For

- Dataset + manifest conventions (industrial data is messy; labels incomplete)
- Pipelines for “paths → decode → preprocess → embeddings → detector → postprocess → artifacts”
- Robustness evaluation: corruptions, lighting shifts, JPEG, blur, sensor noise
- Synthetic anomaly generation (CutPaste family, Perlin masks, scratches/stains/pits)
- Efficient nearest-neighbor search / memory-bank compression (PatchCore-like patterns)

---

## Reference Repos (To Clone For Study)

The list below is intentionally mixed: frameworks, single-method repos, and augmentation toolkits.

Datasets / benchmarks (read the papers + check licenses before redistribution):
- MVTec AD 2 dataset page (CC BY-NC-SA 4.0): https://www.mvtec.com/company/research/datasets/mvtec-ad-2
  - paper: https://arxiv.org/abs/2503.21622
- Real-IAD (multi-view industrial IAD; access-gated): https://realiad4ad.github.io/Real-IAD/
  - D3 paper: https://arxiv.org/abs/2504.14221
- RAD (robotic multi-view; viewpoint/lighting shifts): https://rad-iad.github.io/
  - paper: https://arxiv.org/abs/2411.12179
- ReinAD (reinforced inspection): https://reinad.ai/
- VAND @ CVPR 2025 (Visual Anomaly & Novelty Detection workshop; evaluation conventions, baselines):
  - schedule: https://cvpr2025.thecvf.com/virtual/2025/workshop/36626
  - openaccess: https://openaccess.thecvf.com/CVPR2025_workshops/VAND.html
- MMAD (ICLR 2025 multimodal AD benchmark): https://github.com/jam-cc/MMAD

Framework-style (pipelines + config):
- `openvinotoolkit/anomalib` (Apache-2.0)
- `DoMaLi94/industrial-image-anomaly-detection` (MIT)

Method implementations (conceptual reference only):
- PatchCore: `amazon-science/patchcore-inspection` (Apache-2.0; **can be large to clone**)
- PatchCore (unofficial, lightweight): `tiskw/patchcore-ad` (MIT)
- SPADE (no training; deep pyramid correspondences): `byungjae89/SPADE-pytorch` (Apache-2.0)
- MahalanobisAD (Gaussian deep features): `byungjae89/MahalanobisAD-pytorch` (Apache-2.0)
- PaDiM (unofficial): `xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master` (Apache-2.0)
- SimpleNet: `DonaldRR/SimpleNet` (MIT)
- DRAEM: `VitjanZ/DRAEM` (MIT)
- CutPaste: `LilitYolyan/CutPaste` (MIT)
- Dinomaly: `guojiajeremy/Dinomaly` (Apache-2.0)
- MuSc: `xrli-U/MuSc` (MIT)
- RealNet: `cnulab/RealNet` (MIT)
- RD++ (Reverse Distillation++): `tientrandinh/Revisiting-Reverse-Distillation` (MIT)
- PUAD (Prompt-based): `LeapMind/PUAD` (Apache-2.0)
- STFPM: `gdwang08/STFPM` (**GPL-3.0**; study-only, do not copy code into MIT project)
- RAD (robotic multi-view dataset + baselines): `kaichen-z/RAD` (check upstream license; study-only clone)
- ReinAD (reinforced inspection): `FangshuoX/ReinAD` (check upstream license; study-only clone)
- AnomalyAny (CVPR 2025; diffusion-style anomaly generation ideas): `EPFL-IMOS/AnomalyAny` (check upstream license; study-only clone)

Augmentation/synthesis toolkits (conceptual reference):
- `albumentations-team/albumentations` (MIT)
- `kornia/kornia` (Apache-2.0)
- WE-PaDiM (wavelet enhanced): `BioHPC/WE-PaDiM` (MIT; method reference)

Classical outlier detection (API/contract reference only):
- `yzhao062/pyod` (BSD-2-Clause; API patterns; *not* used as a dependency)

Indexes / survey repositories (good for keeping up-to-date; do not treat as authoritative implementations):
- Awesome IAD list: `M-3LAB/awesome-industrial-anomaly-detection`
- IAD-Survey (paper+code index): `IHPCRits/IAD-Survey`

---

## “Borrowed Concepts” Mapping

- **Registry-driven models:** keep a stable “model name → constructor” API surface.
- **Feature extractor registry:** treat feature extraction as first-class (embeddings vs handcrafted).
- **Industrial presets:** make “works out of the box” configs for common factory settings.
- **Synthesis pipelines:** generate controllable anomaly masks and blend modes; store masks in manifests.
- **Cache layers:** caching decoded images and/or cached feature vectors is a huge speed win in practice.

---

## Local Clone Helper

Use `tools/clone_reference_repos.sh` to clone shallow copies into a local cache directory.

Example:

```bash
bash tools/clone_reference_repos.sh --dir .cache/pyimgano_refs --jobs 4
```
