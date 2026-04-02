# Industrial Workbench

`pyimgano` includes a **recipe-driven workbench** layer for industrial anomaly
detection. The goal is to standardize the “industrial loop” and make runs
comparable via artifacts (reports, per-image JSONL, optional maps/checkpoints).

Primary entrypoint:

```bash
pyimgano-train --config cfg.json
```

Recipe discovery is also available from the umbrella CLI:

```bash
pyimgano train --list-recipes
pyimgano train --recipe-info industrial-adapt --json
```

See also:

- `docs/RECIPES.md` (config schema + builtin recipes)
- `docs/CLI_REFERENCE.md` (CLI flags)
- `docs/MANIFEST_DATASET.md` (JSONL manifest dataset)

---

## Quickstart (end-to-end)

1) Start from an example config:

```
examples/configs/industrial_adapt_fast.json
examples/configs/industrial_adapt_preprocessing_illumination.json
examples/configs/industrial_adapt_maps_tiling.json
examples/configs/industrial_adapt_defects_roi.json
examples/configs/manifest_industrial_workflow_balanced.json
```

2) Edit at least:

- `dataset.root`
- (manifest only) `dataset.manifest_path`
- `output.output_dir` (optional; omit to write under `runs/`)
- `model.device` (`cpu` or `cuda`)

3) Validate the config without running:

```bash
pyimgano-train --config cfg.json --dry-run
```

4) Run the recipe and export an inference helper artifact:

```bash
pyimgano-train --config cfg.json --export-infer-config
```

This writes a run directory containing:

- `report.json` / `config.json` / `environment.json`
- `categories/<cat>/per_image.jsonl`
- `categories/<cat>/report.json` with dataset summary, threshold provenance, split fingerprint,
  and an `evaluation_contract` block for metric semantics
- `artifacts/infer_config.json` (model + adaptation + preprocessing + defects + optional `prediction` defaults + `threshold` + `threshold_provenance` + optional `checkpoint`)
- `artifacts/calibration_card.json` (image-threshold audit payload derived from `threshold_provenance`)
- optional: `artifacts/maps/*.npy`
- optional: `checkpoints/<cat>/...`

`artifacts/infer_config.json` is stamped with `schema_version=1`. Legacy infer-configs without a
schema version are still accepted by `pyimgano-infer` and `pyimgano-validate-infer-config`; they are
normalized to schema version `1` with a warning for backwards compatibility.

Optionally, export a deploy-friendly bundle directory (infer-config + checkpoints + metadata):

```bash
pyimgano-train --config cfg.json --export-deploy-bundle
pyimgano-validate-infer-config /path/to/run_dir/deploy_bundle/infer_config.json
```

The deploy bundle now also includes `bundle_manifest.json`, which records the
bundled files, their relative paths, sizes, and SHA256 digests for auditability.
This manifest is intended to be machine-verifiable as well as human-readable.
It also includes `handoff_report.json`, a compact operator-facing summary of the
bundle contents, threshold context, and expected handoff refs.
When available, `deploy_bundle/` also carries `calibration_card.json` so the
image-threshold context travels with the exported inference bundle.
If you also place `model_card.json` or `weights_manifest.json` in the bundle
root, the bundle manifest and quality gates will validate them as part of the
handoff contract.

5) Reuse the run for inference:

```bash
pyimgano-infer --from-run /path/to/run_dir --input /path/to/images --save-jsonl out.jsonl
```

If the run contains multiple categories, select one:

```bash
pyimgano-infer --from-run /path/to/run_dir --from-run-category bottle --input /path/to/images
```

6) (Optional) Defects export (mask + regions) using the exported infer-config:

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

Notes:

- When the infer-config contains `defects.pixel_threshold`, `pyimgano-infer` uses it for defects export
  (so you can omit `--pixel-threshold`).
- When running with `--infer-config` (or `--from-run`), other defects extraction knobs can also be
  exported in the `defects` block (ROI, morphology, min-area, mask format, max regions, etc.).
  `pyimgano-infer` uses these values as defaults, but explicit CLI flags always override.
- Workbench configs can also export a top-level `prediction` block with
  `reject_confidence_below` / `reject_label`, so low-confidence rejection policy can be reproduced
  in deploy-style inference without retyping CLI flags.

---

## Notes

- `pyimgano-infer --from-run` is **best-effort**: it loads model settings, applies `threshold_`,
  and loads checkpoints when the detector supports it.
- For production shipping, prefer `artifacts/infer_config.json` (`pyimgano-infer --infer-config ...`):
  it’s a minimal “what inference needs” payload and includes `threshold_provenance` for auditing.
- `artifacts/calibration_card.json` is the compact threshold-audit companion artifact for review,
  release notes, and deploy-bundle handoff.
- `deploy_bundle/handoff_report.json` is the short machine-readable delivery note for operators and
  release checklists.
- See `docs/CALIBRATION_AUDIT.md` for the review checklist and expected quality states.
- Use `pyimgano-runs list` / `pyimgano-runs compare` to inspect and compare saved run directories
  without opening `report.json` by hand.
- The same workflow is available from the umbrella CLI:
  `pyimgano train ...`, `pyimgano validate-infer-config ...`, and
  `pyimgano runs quality ...`.
- `pyimgano bundle validate ...`, `pyimgano runs quality ...`, and `pyimgano runs acceptance ...`
  now expose `handoff_report_status` and `next_action` in JSON output so automation can distinguish
  “missing handoff note” from “invalid handoff note” and suggest the next operator step.
- Future infer-config schema versions are rejected explicitly when the installed pyimgano build does
  not support them, rather than being silently accepted.
- For high-resolution inference, you can still use `pyimgano-infer` tiling flags (see
  `docs/INDUSTRIAL_INFERENCE.md`).
- When using `preprocessing.illumination_contrast`, the detector must support numpy inputs (tag: `numpy`).
  Otherwise, preflight/CLI will emit `PREPROCESSING_REQUIRES_NUMPY_MODEL`.

## Weights And Cache Policy

`pyimgano` does **not** ship model weights inside the wheel.

The workbench run directory is meant to stay reviewable and portable:

- `checkpoints/<cat>/...` is for small run-local checkpoints that a recipe writes explicitly.
- `pyimgano-infer --from-run` can reuse those run-local checkpoints when the detector supports checkpoint restore.
- large pretrained weights still live in the cache locations used by the underlying runtime libraries.

For models that fetch upstream weights, cache placement is controlled by the usual environment variables:

- `TORCH_HOME`
- `HF_HOME` / `TRANSFORMERS_CACHE`
- `XDG_CACHE_HOME`

This split keeps deploy bundles and saved runs focused on reproducible artifacts, while heavyweight upstream
weights remain under the standard cache policy for the environment.

## Manifest datasets (JSONL)

For real industrial repos (multi-category, mixed sources, group-aware splits),
prefer the manifest dataset:

- `dataset.name="manifest"`
- `dataset.manifest_path="/path/to/manifest.jsonl"`
- optional: `dataset.split_policy` and `group_id` fields inside the manifest

See: `docs/MANIFEST_DATASET.md`
