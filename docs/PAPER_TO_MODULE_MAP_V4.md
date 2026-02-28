# Paper → Module Map (v4)

This document maps recent industrial anomaly detection datasets/methods to the **specific modules,
contracts, and CLI surfaces** in this repository.

Principles:
- Keep `BaseDetector` semantics consistent (higher score ⇒ more anomalous; thresholding via contamination / calibration).
- Keep `core_*` detectors **feature-matrix first** (`np.ndarray` / torch tensors convertible to NumPy).
- Keep dataset interchange stable via **JSONL manifests** (`image_path`, `mask_path`, `category`, `split`, `label`, `meta`).
- Avoid new required dependencies and avoid implicit weight downloads.

---

## Datasets / Benchmarks

### MVTec AD 2 (dataset + paper)
- Source:
  - Dataset page: https://www.mvtec.com/company/research/datasets/mvtec-ad-2
  - Paper: https://arxiv.org/abs/2503.21622
- What it implies for us:
  - Public splits include `test_public` with `good` / `bad` and optional ground-truth masks.
  - Real deployments often want *paths-first* evaluation plus manifest interchange.
- PyImgAno mapping:
  - Paths-first dataset loader: `pyimgano.utils.datasets.MVTecAD2Dataset` (benchmark loader surface)
  - Manifest converter (paths-first): `pyimgano/datasets/mvtec_ad2.py`
  - Converter entrypoint surface: `pyimgano-manifest --dataset mvtec_ad2`
  - Notes / license reminders: `docs/DATASET_MVTEC_AD2_NOTES.md`

### Real-IAD / Real-IAD D3 (multi-view industrial)
- Source:
  - Project page: https://realiad4ad.github.io/Real-IAD/
  - D3 paper: https://arxiv.org/abs/2504.14221
- What it implies for us:
  - Multi-view / multi-condition inspection is common in factories (viewpoint, lighting, jig).
  - Data splits must be **group-aware** to avoid leakage across views/conditions.
- PyImgAno mapping:
  - Manifest converter with best-effort layout recognition: `pyimgano/datasets/real_iad.py`
  - Manifest schema support:
    - `meta.view_id` (view/camera index)
    - `meta.condition` (lighting / station / configuration label)
    - grouping (`meta.group_id` or `group_id`) for leakage-safe splitting
  - Loader behavior: `pyimgano.datasets.manifest.load_manifest_benchmark_split`
  - Notes: `docs/DATASET_REAL_WORLD_IAD_NOTES.md`

### RAD (robotic multi-view)
- Source:
  - Project page: https://rad-iad.github.io/
  - Paper: https://arxiv.org/abs/2411.12179
- What it implies for us:
  - Viewpoint changes + illumination changes should be represented in manifest metadata.
  - Evaluation should be configurable to split by grouped captures.
- PyImgAno mapping:
  - Manifest converter: `pyimgano/datasets/rad.py`
  - Manifest schema: `meta.view_id`, `meta.condition`, and group-aware splitting.

### VAND workshop framing (pixel metrics conventions)
- Source:
  - Workshop page: https://cvpr2025.thecvf.com/virtual/2025/workshop/36626
  - Openaccess listing: https://openaccess.thecvf.com/CVPR2025_workshops/VAND.html
- What it implies for us:
  - Single-threshold pixel metrics like SegF1 are common in industrial evaluations.
  - When pixel masks are absent, pipelines should still run image-level metrics and clearly report why pixel metrics are disabled.
- PyImgAno mapping:
  - Pixel metrics: `pyimgano.evaluation` (pixel AUROC/AP/AUPRO/SegF1)
  - Pixel-threshold strategies: `pyimgano.calibration.pixel_threshold`
  - Manifest behavior and provenance reporting: `pyimgano.datasets.manifest`

---

## Methods / Pipelines (study targets → contract-aligned implementations)

### PatchCore-style nearest-neighbor inspection
- Source: `amazon-science/patchcore-inspection` (large) and lightweight forks
- What it implies for us:
  - Industrial baseline: patch embeddings + memory bank + kNN distances.
  - Tiling and map blending are essential for high-res images.
- PyImgAno mapping:
  - Existing lightweight paths-first variants under `pyimgano.models` (PatchCore-lite and related)
  - Embedding extractors: `pyimgano.features.torchvision_*`
  - Classical core on embeddings route: `vision_embedding_core` + `core_*`

### CutPaste / synthetic anomaly generation
- Source: `LilitYolyan/CutPaste` (and follow-ups)
- What it implies for us:
  - Controlled anomaly masks + blending modes are practical for data-scarce factories.
  - Synthetic generation should emit manifest records with reproducible seeds and mask paths.
- PyImgAno mapping:
  - Synthesis primitives and presets: `pyimgano.synthesis`
  - CLI dataset generation: `pyimgano-synthesize` / `pyimgano.synthesize_cli`

### AnomalyAny (diffusion-based generation ideas)
- Source: https://github.com/EPFL-IMOS/AnomalyAny
- What it implies for us:
  - Interesting augmentation ideas, but must remain optional (no implicit downloads).
- PyImgAno mapping:
  - Conceptual reference for synthesis; keep any future integration behind guarded optional imports.

### MMAD (multimodal benchmark)
- Source: https://github.com/jam-cc/MMAD
- What it implies for us:
  - Useful taxonomy for evaluation and reporting, even if we stay vision-only for the core library.
- PyImgAno mapping:
  - Reporting / benchmark structure references: `pyimgano.reporting`, `pyimgano.cli`

---

## Reference Clone Policy (for study-only repos)

- Use `tools/clone_reference_repos.sh` for shallow, no-checkout clones into `.cache/pyimgano_refs/`.
- Do not commit reference clones; enforce via local audits.
- Copying code into this repo requires license compatibility + `UPSTREAM:` markers + `third_party/NOTICE.md` entries.

