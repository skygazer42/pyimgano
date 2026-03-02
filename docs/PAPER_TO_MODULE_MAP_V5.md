# Paper → Module Map (v5)

This document maps **recent papers / industrial directions** to concrete `pyimgano` modules and
contracts, with an emphasis on **offline-by-default** behavior (no implicit downloads) and
compatibility with:

- `BaseDetector` semantics (higher score ⇒ more anomalous)
- manifest JSONL datasets (paths-first)
- pixel anomaly maps → defects export

This is a planning + onboarding aid, not an exhaustive bibliography.

---

## Patch-level kNN (foundation embeddings → classical core)

### AnomalyDINO (WACV 2025; DINOv2 patch-kNN)

**Concept:**
- Extract DINOv2 patch tokens
- Build a memory bank on normal patches
- Score each patch by NN distance
- Aggregate to image score + upsample to pixel anomaly map

**pyimgano mapping:**
- Model: `pyimgano/models/anomalydino.py` (`vision_anomalydino`)
- Patch aggregation helpers: `pyimgano/models/patchknn_core.py`
- kNN index abstraction: `pyimgano/models/knn_index.py`

**Industrial constraints:**
- Default behavior must be offline. Provide patch embedder via `embedder=...`
  or explicitly opt-in via `pretrained=True` (may use `torch.hub`).

### SuperAD (DINOv2 patch-kNN; k-th neighbor distance)

**Concept:**
- Same embedder + memory bank, but use k-th NN distance for robustness

**pyimgano mapping:**
- Model: `pyimgano/models/superad.py` (`vision_superad`)
- Shares embedder + patch helpers with AnomalyDINO modules above.

---

## Patch/Pixel map baselines (template/reference inspection)

### Odd-One-Out neighbor comparison (CVPR 2025)

**Concept:**
- Robust neighbor comparison under nuisance variation

**pyimgano mapping (v5 direction):**
- Study-only reference for additional `core_*` and `vision_*` “neighbor-compare” variants.
- Existing starting point: `pyimgano/models/oddoneout.py` (registry: `oddoneout` family).

---

## LVLM / agent-style pipelines (study-only in v5)

### IAD-GPT (ICLR 2025 workshop) + VELM (arXiv)

**Concept:**
- Use a LVLM to produce anomaly descriptions / rationales / query-time prompts
- Often depends on large foundation weights and external services

**pyimgano mapping:**
- Keep as docs-only until a dependency/weights policy is defined.
- Study note: `docs/STUDY_LVLM_ANOMALY_DETECTION.md`

---

## Notes for implementers

- Prefer **injectable embedders** and **explicit checkpoints** over hidden downloads.
- Keep unit tests hermetic: tests may monkeypatch `torch.hub.*` to hard-fail.
- When a model cannot work offline without weights, make that requirement explicit:
  raise a clear error early (constructor) or mark `requires_checkpoint` in registry metadata
  and enforce via CLI.

