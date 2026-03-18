# Industrial Fast-Path

This guide is the shortest audited route from a workbench config to a
deployable inference bundle.

## Goal

Use one config, produce one run directory, and leave behind an artifact set
that is easy to inspect, compare, and hand off:

- `report.json`
- `config.json`
- `environment.json`
- `artifacts/infer_config.json`
- `artifacts/calibration_card.json`
- `deploy_bundle/bundle_manifest.json`

The recommended starting point is:

```text
examples/configs/industrial_adapt_audited.json
```

That example is tuned for the audited export path:

- `output.save_run=true` so the run is persisted
- `training.enabled=true` so a checkpoint can be exported
- `adaptation.save_maps=true` so map artifacts can be reviewed when needed
- `defects.pixel_threshold` is fixed so `--export-deploy-bundle` is self-contained

## Recommended Flow

1. Validate the config:

```bash
pyimgano-train --config examples/configs/industrial_adapt_audited.json --dry-run
```

2. Run training and export the audited inference payload:

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_audited.json \
  --export-infer-config
```

3. Export the deploy bundle:

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_audited.json \
  --export-deploy-bundle
```

4. Inspect artifact completeness:

```bash
pyimgano-runs quality /path/to/run_dir --json
```

## What To Review

`artifacts/infer_config.json`

- Inference-facing payload for `pyimgano-infer`
- Includes threshold and threshold provenance
- Carries split metadata when available

`artifacts/calibration_card.json`

- Compact threshold-audit card
- Records the calibration context behind the exported threshold
- Must be valid, not just present, for `pyimgano-runs quality` to report an `audited` run

`deploy_bundle/bundle_manifest.json`

- Lists every bundled file
- Captures relative paths, sizes, and SHA256 digests
- Intended for machine verification during release handoff

## Verification Loop

After export, the minimal audited check is:

```bash
pyimgano-runs quality /path/to/run_dir --require-status audited --json
pyimgano-validate-infer-config /path/to/run_dir/deploy_bundle/infer_config.json
```

If `pyimgano-runs quality` reports `deployable`, the run has the full audited
artifact set including `infer_config.json`, `calibration_card.json`, and
`bundle_manifest.json`. If the bundle also carries `model_card.json` or
`weights_manifest.json`, those files must validate as well for the run to stay
at `deployable`.

For a field-by-field review checklist, see `docs/CALIBRATION_AUDIT.md`.
