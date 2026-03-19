# Robustness Benchmark (Clean + Industrial Drift Corruptions)

This guide describes the **deploy-style robustness benchmark** introduced in `pyimgano`.

It evaluates an anomaly detector on:
- a **clean** test set, and
- the same test set under a set of **deterministic corruptions** (industrial drift),

while keeping a **single fixed pixel threshold** for the entire run (calibrated once from
normal training images).

---

## What it measures

### Image-level metrics
You always get standard image-level metrics (AUROC/AP/F1/etc.) from `evaluate_detector(...)`.

### Pixel SegF1 + background FPR (VAND-style)
If you enable pixel SegF1 (`--pixel-segf1`, default in the robustness CLI):

- A single pixel threshold `t` is calibrated from normal pixels using a quantile:
  - `t = quantile(scores_normal_pixels, q)`
- The same `t` is used for:
  - clean evaluation
  - every corruption × severity
- Reported pixel metrics include:
  - `pixel_segf1` (pixel-level Segmentation F1)
  - `bg_fpr` (background false-positive rate)

This matches a production constraint: **you don't get to retune the threshold per condition**.

### Latency
The report includes a best-effort `latency_ms_per_image` per condition (measured around
`decision_function()` + optional anomaly map extraction).

---

## Quickstart: CLI (recommended)

Run a robustness benchmark on MVTec AD:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --preset industrial-balanced \
  --pretrained \
  --device cuda \
  --pixel-normal-quantile 0.999 \
  --pixel-calibration-fraction 0.2 \
  --corruptions lighting,jpeg,blur,glare,geo_jitter \
  --severities 1 2 3 4 5 \
  --save-run \
  --output-dir /tmp/pyimgano_robust_run \
  --output runs/robust_mvtec_bottle_patchcore.json
```

For a quick smoke run:

```bash
pyimgano-robust-benchmark \
  --dataset mvtec \
  --root /path/to/mvtec_ad \
  --category bottle \
  --model vision_patchcore \
  --device cpu \
  --no-pretrained \
  --corruptions lighting \
  --severities 1 \
  --limit-train 32 \
  --limit-test 64
```

Notes:
- Datasets without segmentation masks (e.g. some exports of BTAD) should run with:
  - `--no-pixel-segf1`
- Corruptions require `--input-mode numpy` (default), which feeds detectors **numpy RGB** images.
  Use vision models that accept `RGB/u8/HWC` numpy images (e.g. PatchCore / AnomalyDINO / SuperAD).
- For detectors that only accept file paths (many classical baselines), use `--input-mode paths`
  for clean-only evaluation (corruptions are skipped).
- `--save-run` persists `report.json`, `config.json`, and `environment.json` so robustness
  runs can be indexed with `pyimgano-runs`.
- `--output-dir` lets you choose the persisted run directory explicitly. If omitted, a run
  directory is created only when `--save-run` is enabled.
- Saved robustness runs also export:
  - `artifacts/robustness_conditions.csv`
  - `artifacts/robustness_summary.json`
  - `robustness_conditions.csv` includes per-condition `drop_*` columns relative to the clean baseline,
    so publication tables and regression gates do not need to recompute clean deltas offline.
  - `robustness_summary.json` carries a `trust_summary` plus audit refs/digests for
    `robustness_conditions.csv`, while the saved `report.json` payload carries matching
    `robustness_trust.audit_refs|audit_digests` entries for both robustness artifacts.
  - `pyimgano-runs list` / `latest` revalidate those saved artifact digests on read, so a later
    edit to either robustness artifact degrades the reported `robustness_trust` status.

You can then inspect saved robustness runs with:

```bash
pyimgano-runs list --root runs --kind robustness --json
pyimgano-runs list --root runs --kind robustness --same-robustness-protocol-as runs/robust_a --json
pyimgano-runs latest --root runs --kind robustness --same-robustness-protocol-as runs/robust_a --json
pyimgano-runs compare runs/robust_a runs/robust_b --baseline runs/robust_a --require-same-robustness-protocol --json
```

The compare JSON payload now includes `robustness_protocol_comparison`, which checks that
baseline and candidate runs used the same corruption mode, corruption set, severity schedule,
input mode, and resize before you trust a regression decision.
For discovery workflows, `--same-robustness-protocol-as` applies the same protocol signature to
`pyimgano-runs list` and `pyimgano-runs latest`, so you can shortlist only directly comparable
robustness runs before opening the full comparison payload.

---

## Corruptions included

The default corruption set is:

- `lighting`: exposure / contrast / gamma + mild per-channel gain drift
- `jpeg`: JPEG encode/decode artifacts (blocking/ringing)
- `blur`: Gaussian blur
- `glare`: synthetic specular/glare blobs
- `geo_jitter`: small affine warp (image + mask are warped consistently)

Each corruption is deterministic for a given `--seed`, name, and severity.

---

## Output schema

The CLI prints a JSON object (or saves via `--output`) with a structure like:

```json
{
  "dataset": "...",
  "category": "...",
  "model": "...",
  "robustness_summary": {
    "clean_auroc": 0.99,
    "mean_corruption_auroc": 0.94,
    "worst_corruption_auroc": 0.88,
    "mean_corruption_drop_auroc": 0.05,
    "worst_corruption_drop_auroc": 0.11,
    "clean_latency_ms_per_image": 12.3,
    "mean_corruption_latency_ms_per_image": 14.1,
    "worst_corruption_latency_ms_per_image": 16.8,
    "mean_corruption_latency_ratio": 1.146341463415,
    "worst_corruption_latency_ratio": 1.365853658537
  },
  "robustness": {
    "pixel_threshold_strategy": "normal_pixel_quantile",
    "pixel_normal_quantile": 0.999,
    "clean": {
      "latency_ms_per_image": 12.3,
      "results": { "...": "evaluate_detector() payload" }
    },
    "corruptions": {
      "lighting": {
        "severity_1": { "...": "same schema as clean" },
        "severity_2": { "...": "..." }
      }
    }
  }
}
```

When `--save-run` is enabled, `artifacts/robustness_conditions.csv` flattens the clean and
corrupted conditions into a publication-friendly table and includes explicit `drop_*` columns
relative to the clean baseline, while `artifacts/robustness_summary.json` stores the compact
aggregate summary. That summary also records audit refs/digests for the exported conditions CSV,
so downstream tooling can verify the flattened robustness table without recomputing hashes ad hoc.

The summary is designed to support direct gating in `pyimgano-runs compare`, including:
- accuracy retention (`mean_corruption_*`, `worst_corruption_*`)
- clean-to-corruption degradation (`mean_corruption_drop_*`, `worst_corruption_drop_*`)
- latency stability (`*_latency_ms_per_image`, `*_latency_ratio`)

For saved robustness runs, `pyimgano-runs compare` also defaults the comparison contract to the
robustness-aware metric surface, so the primary ranking metric becomes `worst_corruption_auroc`
when available. Use `--require-same-robustness-protocol` to fail fast when the candidate changed
corruptions, severities, input mode, or resize and therefore is not directly comparable.

---

## Tuning tips

- `--pixel-normal-quantile` controls the tradeoff:
  - higher quantile → fewer background false positives (lower `bg_fpr`), but may miss small defects
  - lower quantile → more sensitive, but more false positives
- If your production normals are noisy, consider evaluating with `vision_softpatch` and the
  `industrial-balanced` preset first.
