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
- `deploy_bundle/handoff_report.json`

The recommended starting point is:

```text
examples/configs/industrial_adapt_audited.json
```

If you want to inspect the available train recipes before running the audited path,
use the umbrella CLI first:

```bash
pyimgano train --list-recipes
pyimgano train --recipe-info industrial-adapt --json
pyimgano train --dry-run --config examples/configs/industrial_adapt_audited.json
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

`deploy_bundle/handoff_report.json`

- Compact operator-facing summary of the bundle handoff
- Mirrors the expected key refs for deploy validation
- Helps downstream automation distinguish missing vs invalid handoff metadata

## Verification Loop

After export, the minimal audited check is:

```bash
pyimgano-runs quality /path/to/run_dir --require-status audited --json
pyimgano-validate-infer-config /path/to/run_dir/deploy_bundle/infer_config.json
pyimgano-runs acceptance /path/to/run_dir --require-status audited --check-bundle-hashes --json
```

If the deploy bundle also carries `model_card.json` and `weights_manifest.json`,
gate that handoff with:

```bash
pyimgano-weights audit-bundle /path/to/run_dir/deploy_bundle --check-hashes --json
```

If the next step is a suite export handoff instead of a single-run delivery,
the same acceptance entrypoint also works on the publication bundle:

```bash
pyimgano-runs acceptance /path/to/suite_export --json
```

## Offline Bundle Execution

Once the bundle is exported and accepted, the same artifact can be validated and
executed as a fixed offline QC package:

```bash
pyimgano bundle validate /path/to/run_dir/deploy_bundle --json

pyimgano bundle run /path/to/run_dir/deploy_bundle \
  --image-dir /path/to/lot_images \
  --output-dir ./bundle_run \
  --max-anomaly-rate 0.05 \
  --max-reject-rate 0.02 \
  --max-error-rate 0.00 \
  --min-processed 100 \
  --json
```

The bundle run contract is intentionally narrow:

- `results.jsonl` is always written to `<output_dir>/results.jsonl`
- `run_report.json` is always written to `<output_dir>/run_report.json`
- `run_report.json` carries `batch_verdict`, `batch_gate_summary`,
  `batch_gate_reason_codes`, and output digests for downstream automation
- pixel outputs stay gated behind the bundle contract and only write to the fixed
  locations `<output_dir>/masks`, `<output_dir>/overlays`, and
  `<output_dir>/defects_regions.jsonl`

If `pyimgano-runs quality` reports `deployable`, the run has the full audited
artifact set including `infer_config.json`, `calibration_card.json`, and
`bundle_manifest.json`. If the bundle also carries `model_card.json` or
`weights_manifest.json`, those files must validate as well for the run to stay
at `deployable`.

For a field-by-field review checklist, see `docs/CALIBRATION_AUDIT.md`.
