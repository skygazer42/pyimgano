# PyImgAno Manifest Dataset (JSONL) — Design

**Date:** 2026-02-23  
**Status:** Draft (approved sections + one open decision)  

## Context

`pyimgano` currently supports benchmark datasets (MVTec/VisA/BTAD/...) and a
directory-based `custom` dataset layout. For industrial projects, data often
exists as:

- Multiple categories in a single storage root (or across multiple roots)
- Mixed sources with partial labels/masks
- Strong “group” semantics (part/lot/video clip) where leakage is easy
- Need for a single canonical dataset interface for workbench + training + inference

This design adds a **Manifest dataset**: a single JSONL file describing samples
across categories, with explicit metadata for splits/labels/masks when present,
and a safe, reproducible auto-split policy when not.

## Goals

- A single JSONL manifest can describe **multiple categories**.
- Compatible with the industrial workbench:
  - produces the same artifacts (`report.json`, per-image JSONL, optional maps)
  - supports `dataset.category="all"` (run across categories)
- Split semantics:
  - `split` is **optional** in the manifest
  - if `split` is missing, `split_policy` generates deterministic splits
  - optional `group_id` enables **group-aware** splitting to avoid leakage
- Label/mask semantics:
  - train defaults to normal when `label` missing
  - test must have labels to compute meaningful image-level metrics
  - masks are optional; pixel metrics are computed only when safe
- Path resolution that works in real industrial repos (portable manifests).

## Non-goals

- Replacing existing benchmark loaders or changing their semantics.
- Supporting non-path inputs (e.g., embedding raw image bytes) in v1.
- Trying to guess labels/masks from filenames (explicit is better).

## Recommended Approach

**Option A (recommended):** introduce a new `pyimgano.datasets.manifest` module
that:

1) Parses JSONL records
2) Resolves paths
3) Applies `split_policy`
4) Emits a `BenchmarkSplit`-compatible payload:
   - `train_paths: list[str]`
   - `test_paths: list[str]`
   - `test_labels: np.ndarray`
   - `test_masks: np.ndarray | None`

This keeps `workbench.runner` small and lets future CLIs/pipelines reuse the
same loader.

## Manifest JSONL Schema (v1)

Each line is a JSON object representing a single sample.

### Required fields

- `image_path: str`
- `category: str` (single JSONL supports multi-category)

### Optional fields

- `split: "train" | "val" | "test"` (optional; if absent, auto-split applies)
- `label: 0 | 1`
  - if `split=="train"` and `label` missing → treated as `0` (normal)
  - if `split=="test"` and `label` missing → **invalid** (must provide label)
- `mask_path: str` (optional; enables pixel metrics when present)
- `group_id: str` (optional; group-aware split)
- `meta: object` (optional; user-defined metadata, treated as opaque)

### Example

```jsonl
{"image_path":"cats/bottle/img_0001.png","category":"bottle","split":"train","group_id":"lotA"}
{"image_path":"cats/bottle/img_0101.png","category":"bottle","split":"test","label":0,"group_id":"lotB"}
{"image_path":"cats/bottle/img_0201.png","category":"bottle","split":"test","label":1,"mask_path":"masks/bottle/img_0201.png","group_id":"lotB","meta":{"line":"L1"}}
{"image_path":"/abs/path/cats/cable/img_0005.png","category":"cable","label":1,"mask_path":"/abs/path/masks/cable/img_0005.png"}
```

## Path Resolution Rules

For `image_path` and `mask_path`:

- If absolute: use as-is.
- If relative:
  1) resolve relative to the manifest file directory
  2) if not found, fallback to `dataset.root` (compat / migration support)

## Split Policy (Auto Split)

### When split is explicit

- If a record provides `split`, it is respected.
- Validation:
  - `split=="test"` requires `label` to be present.
  - `split=="train"` with `label==1` is rejected (prevents leakage).

### When split is missing

`split_policy` assigns records deterministically with a seed:

**Recommended default behavior (`mode="benchmark"`):**

- For each category (recommended default scope):
  - If `label==1` (anomaly): assign to `test`.
  - Else (normal or label missing): split into `train` and `test` based on
    `test_normal_fraction` (default e.g. `0.2`).

**Group-aware splitting (when `group_id` present):**

- Groups are treated as indivisible units.
- If any record in a group is anomalous (`label==1`), the whole group is assigned to `test`.
- Remaining all-normal groups are split to satisfy `test_normal_fraction`.

### Open decision (needs confirmation)

When `dataset.category=="all"` and auto-splitting is needed, the policy scope
can be:

- `scope="category"` (recommended): split within each category separately.
- `scope="global"`: split across the entire manifest regardless of category.

Default recommendation: `scope="category"`.

## Evaluation Semantics

### Image-level metrics

- Image-level metrics require `test_labels` (0/1).
- If auto-splitting produces a test set without any anomalies, AUROC/AP may be `NaN` (expected).

### Pixel-level metrics (masks)

- Pixel metrics are computed only when:
  - at least one test record provides `mask_path`, AND
  - for every `label==1` test record, a corresponding mask is available.
- If any anomaly test record is missing a mask:
  - **skip pixel metrics** and write an explicit reason in the run report payload.

## Workbench Config Additions (Proposed)

Extend `dataset` config to support:

```json
{
  "dataset": {
    "name": "manifest",
    "root": "/fallback/root",
    "manifest_path": "/path/to/manifest.jsonl",
    "category": "all",
    "resize": [256, 256],
    "input_mode": "paths",
    "split_policy": {
      "mode": "benchmark",
      "scope": "category",
      "seed": 123,
      "test_normal_fraction": 0.2
    }
  }
}
```

Notes:

- `input_mode="paths"` only in v1.
- `split_policy.seed` defaults to top-level `seed` when omitted.

## Testing Strategy

- Unit tests for:
  - JSONL parsing + validation (required fields, label rules)
  - path resolution (manifest-dir first, then root fallback)
  - deterministic splitting (seeded, stable ordering)
  - group-aware split (no group leakage)
  - pixel-metric gating behavior (skip on missing anomaly masks)
- Workbench integration smoke:
  - manifest → train (recipe) → infer-from-run on a tiny synthetic dataset (paths only)

## Acceptance Criteria

1) `pyimgano-train` can run with `dataset.name="manifest"` and produce standard run artifacts.
2) `dataset.category="all"` works for multi-category manifests (per-category reports).
3) Auto-split is deterministic and group-aware when `group_id` exists.
4) Pixel metrics are computed only when safe; otherwise explicitly skipped with a reason.

