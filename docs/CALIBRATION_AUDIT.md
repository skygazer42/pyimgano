# Calibration Audit

This guide explains how to review and trust the threshold-audit artifacts
produced by the industrial workbench flow.

## Why This Exists

`pyimgano` exports image-level thresholds into deploy artifacts, so the
threshold itself needs an audit trail. The compact audit artifact is:

- `artifacts/calibration_card.json`

That card is intended to answer three questions:

1. What threshold is being deployed?
2. How was that threshold derived?
3. Which dataset split or category context did it come from?

## Minimal Flow

Generate the audit artifact from a workbench run:

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_audited.json \
  --export-infer-config
```

Then inspect the run quality:

```bash
pyimgano-runs quality /path/to/run_dir --require-status audited --json
```

For deploy handoff, also validate the infer-config:

```bash
pyimgano-validate-infer-config /path/to/run_dir/deploy_bundle/infer_config.json
```

## What To Review In `calibration_card.json`

Top-level fields:

- `schema_version`: current calibration-card schema version.
- `run_dir`, `dataset`, `category`, `model`, `recipe`: source run metadata.
- `split_fingerprint`: optional split identity for comparability-sensitive runs.

Threshold payload:

- `image_threshold.threshold`: the exported image-level decision threshold.
- `image_threshold.provenance.method`: how the threshold was chosen, such as `quantile`.
- `image_threshold.provenance.source`: where the calibration rule came from, such as `contamination`.
- `image_threshold.provenance.quantile`: the quantile used when applicable.
- `image_threshold.provenance.score_summary`: compact distribution summary of calibration scores.

Multi-category runs may export `per_category` instead of a single
`image_threshold`. Each category entry should carry the same threshold and
provenance structure.

## Quality Expectations

`pyimgano-runs quality` now distinguishes between:

- `reproducible`: core run artifacts are present, but audited calibration artifacts are incomplete or invalid.
- `audited`: `infer_config.json` and a valid `calibration_card.json` are both present.
- `deployable`: audited artifacts are present, the deploy bundle manifest is valid, and any bundled
  `model_card.json` / `weights_manifest.json` files also validate cleanly.

This matters because a merely present `calibration_card.json` is not enough.
If the card is malformed, the run should not be treated as fully audited.

If you want a machine-enforced gate, use:

```bash
pyimgano-runs quality /path/to/run_dir --require-status audited --json
```

or, for a full handoff bundle:

```bash
pyimgano-runs quality /path/to/run_dir --require-status deployable --json
```

## Common Review Questions

Check these before handing off a deploy bundle:

- Does the threshold value in `calibration_card.json` match the threshold exported in `infer_config.json`?
- Does `provenance.source` match your intended calibration route, such as contamination-based or fixed?
- Is the `score_summary.count` large enough to justify the calibration?
- If split comparability matters, is `split_fingerprint.sha256` present and stable across the runs you compare?
- If the bundle includes `model_card.json` or `weights_manifest.json`, do they validate and point at the same checkpoint asset?

## Related Docs

- `docs/WORKBENCH.md`
- `docs/INDUSTRIAL_FASTPATH.md`
- `docs/CLI_REFERENCE.md`
