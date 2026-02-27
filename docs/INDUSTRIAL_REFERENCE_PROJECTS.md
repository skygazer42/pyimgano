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

Framework-style (pipelines + config):
- `openvinotoolkit/anomalib` (Apache-2.0)
- `DoMaLi94/industrial-image-anomaly-detection` (MIT)

Method implementations (conceptual reference only):
- PatchCore: `amazon-science/patchcore-inspection` (Apache-2.0)
- SimpleNet: `DonaldRR/SimpleNet` (MIT)
- DRAEM: `VitjanZ/DRAEM` (MIT)
- CutPaste: `LilitYolyan/CutPaste` (MIT)
- Dinomaly: `guojiajeremy/Dinomaly` (Apache-2.0)
- MuSc: `xrli-U/MuSc` (MIT)

Augmentation/synthesis toolkits (conceptual reference):
- `albumentations-team/albumentations` (MIT)
- `kornia/kornia` (Apache-2.0)

Classical outlier detection (API/contract reference only):
- `yzhao062/pyod` (BSD-2-Clause; API patterns; *not* used as a dependency)

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
