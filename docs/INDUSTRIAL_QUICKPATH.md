# Industrial Quick Path

This is the shortest reliable path from a manifest dataset to deploy-style
inference artifacts.

Use this flow when you want:

- a quick data-health check before training
- a reproducible workbench run with exported `infer_config.json`
- a deploy-friendly inference command that does not need the full training config

## 1. Start from a manifest-first config

Use the example config as a template:

```bash
cp examples/configs/manifest_industrial_workflow_balanced.json ./cfg_manifest.json
```

Edit at least:

- `dataset.root`
- `dataset.manifest_path`
- `model.device`
- `output.output_dir` if you do not want the default under `runs/`

## 2. Run preflight before training

Preflight validates the dataset contract and exits non-zero when there are
blocking issues such as missing files or invalid train/test grouping.

```bash
pyimgano-train --config ./cfg_manifest.json --preflight
```

What to look for:

- `summary.counts` for train/test coverage
- `summary.mask_coverage` when pixel metrics matter
- `issues` with severities `error`, `warning`, or `info`

## 3. Train and export the deploy artifact

```bash
pyimgano-train --config ./cfg_manifest.json --export-infer-config
```

This writes a run directory containing:

- `report.json`
- `categories/<category>/per_image.jsonl`
- `artifacts/infer_config.json`
- optional `checkpoints/<category>/...`

If you prefer a self-contained handoff directory, export a deploy bundle:

```bash
pyimgano-train --config ./cfg_manifest.json --export-deploy-bundle
```

## 4. Validate the exported infer-config

Validate the file before shipping it to another machine or pipeline:

```bash
pyimgano-validate-infer-config runs/<run_dir>/artifacts/infer_config.json
```

For a deploy bundle:

```bash
pyimgano-validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json
```

## 5. Run deploy-style inference

Use the exported config instead of the full training config:

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

If the infer-config contains multiple categories, add:

```bash
--infer-category bottle
```

## 6. Optional: defects export

When the exported config contains a `defects` block, `pyimgano-infer` can reuse
those defaults for mask and region generation:

```bash
pyimgano-infer \
  --infer-config runs/<run_dir>/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-jsonl /tmp/pyimgano_results.jsonl \
  --save-masks /tmp/pyimgano_masks
```

Notes:

- CLI flags still override values coming from `infer_config.json`.
- Relative checkpoint paths inside the exported infer-config remain portable as
  long as the config stays near the run or deploy bundle layout.
- For production handoff, prefer the deploy bundle over sharing the entire run
  directory.

## 7. Optional: benchmark the same manifest dataset

Use the manifest directly with the benchmark CLI when you want a comparable
algorithm-selection run:

```bash
pyimgano-benchmark \
  --dataset manifest \
  --root /path/to/dataset_root \
  --manifest-path /path/to/manifest.jsonl \
  --category bottle \
  --suite industrial-v4 \
  --device cpu \
  --no-pretrained
```

See also:

- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/MANIFEST_DATASET.md`
