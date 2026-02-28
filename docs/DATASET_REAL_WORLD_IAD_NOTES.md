# Real-World Industrial AD Notes (Real-IAD / RAD / ReinAD)

This page captures practical settings and schema choices for **real-world industrial anomaly detection**
datasets and deployments, especially when data is **multi-view** or **multi-condition**.

It is intentionally “paths + metadata” focused: this repository avoids bundling datasets, weights,
or large assets.

---

## What changes in real deployments

Compared to single-view benchmarks, factories often have:
- multiple cameras or viewpoints per item (`view_id`)
- multiple lighting/station configurations (`condition`)
- repeated captures of the same physical item (grouped samples)
- unstable backgrounds / fixtures that should be excluded via ROI

These properties matter for:
- leakage-safe splitting (avoid train/test overlap across the same physical item)
- stable calibration (choose thresholds per view/condition when needed)
- debugging (false positives correlate strongly with view + lighting)

---

## Manifest schema (recommended)

Use a JSONL manifest with:
- `image_path` (required)
- `mask_path` (optional; required only for pixel metrics)
- `category` (required)
- `split` / `label` (optional; can be auto-assigned for benchmark-style runs)
- `meta` (optional object/dict), recommended keys:
  - `meta.view_id`: camera/view identifier (string or int-like)
  - `meta.condition`: lighting/station/config label (string)
  - `meta.group_id`: stable group ID for leakage-safe splitting (string)

Splitting:
- Prefer grouping by `meta.group_id` (or a top-level `group_id` field when present).
- Groups must not mix explicit splits.

The core loader is `pyimgano.datasets.manifest.load_manifest_benchmark_split`.

---

## Real-IAD

Source: https://realiad4ad.github.io/Real-IAD/

Practical notes:
- Access is gated; layouts can evolve.
- For conversion, use best-effort layout recognition and emit a manifest that preserves view/condition.

Repository mapping:
- Manifest converter: `pyimgano/datasets/real_iad.py`

---

## RAD (robotic multi-view)

Source: https://rad-iad.github.io/

Practical notes:
- Multi-view and lighting shifts are central; `meta.view_id` and `meta.condition` are first-class.
- Group-aware splitting is recommended to avoid leakage across views of the same item.

Repository mapping:
- Manifest converter: `pyimgano/datasets/rad.py`

---

## ReinAD

Source: https://reinad.ai/

Practical notes:
- Reinforcement-style inspection suggests interactive loops and station-level policies.
- For this repository, the near-term focus is on stable primitives:
  manifests, pixel maps, defects export, and embedding+core routes.

---

## Recommended “minimum viable” reporting

For industrial debugging, include in evaluation outputs:
- per-image score + label
- view/condition metadata (if present)
- defects export artifacts when pixel maps are available (mask + regions + provenance)

