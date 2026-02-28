# MVTec AD 2 Notes (License, Layout, Evaluation)

This page captures practical notes for working with **MVTec AD 2** in a production-style pipeline.

Sources:
- Dataset page: https://www.mvtec.com/company/research/datasets/mvtec-ad-2
- Paper: https://arxiv.org/abs/2503.21622

---

## License / redistribution reminder

MVTec AD 2 is published under **CC BY‑NC‑SA 4.0** (per the dataset page).
Before distributing any derived artifacts (cropped images, exports, manifests that embed pixel data, etc.),
confirm your intended use is compliant (especially “NC” and “SA” clauses).

For this repository:
- Dataset converters focus on **paths + metadata** and do not bundle any images.
- Unit tests use tiny synthetic images, not the dataset.

---

## Expected on-disk layout (common public split)

The dataset contains multiple categories; a typical layout per category is:

```
<root>/<category>/
  train/good/*.png
  validation/good/*.png
  test_public/good/*.png
  test_public/bad/*.png
  test_public/ground_truth/bad/*.png   (optional; mask naming varies)
```

Notes:
- “private” test splits may not include GT masks.
- Mask naming can vary (`<stem>_mask.png`, same filename, etc.). Our loaders/converters use a
  best-effort set of candidates and fall back to “all zeros” if GT is absent.

---

## Evaluation gotchas

- **Pixel metrics require masks.** If the split you have does not include ground-truth masks,
  you can still compute image-level metrics but pixel AUROC/AP/AUPRO/SegF1 will be unavailable.
- **Split leakage:** if the dataset contains multiple captures per object/view, keep splitting
  group-aware when exporting to manifests.
- **Thresholding:** industrial workflows often fix a single threshold per product/station and
  track drift; the workbench and CLI support calibrating thresholds from normal samples.

---

## Converting to a JSONL manifest (paths-first)

Use the dataset converter CLI to emit a stable manifest for downstream pipelines:

```bash
pyimgano-manifest \
  --dataset mvtec_ad2 \
  --root /path/to/mvtec_ad2_root \
  --category bottle \
  --out /tmp/mvtec_ad2_bottle.jsonl \
  --absolute-paths
```

The manifest format is documented in `docs/MANIFEST_DATASET.md`.

