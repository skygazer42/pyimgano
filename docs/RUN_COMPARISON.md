# Run Comparison

`pyimgano-runs` is a lightweight helper for inspecting and comparing saved
benchmark and workbench runs without opening JSON by hand.

## List Runs

Scan the default `runs/` directory:

```bash
pyimgano-runs list
```

Scan a custom root and emit JSON:

```bash
pyimgano-runs list --root /path/to/runs --json
```

Filter by run kind or dataset:

```bash
pyimgano-runs list --root runs --kind suite --dataset mvtec
pyimgano-runs list --root runs --query patchcore --json
pyimgano-runs list --root runs --min-quality reproducible --json
pyimgano-runs list --root runs --same-environment-as runs/run_a --json
pyimgano-runs list --root runs --same-target-as runs/run_a --json
pyimgano-runs list --root runs --kind robustness --same-robustness-protocol-as runs/robust_a --json
```

The tool summarizes top-level run directories only. Nested category reports and
suite baseline sub-runs are ignored so the listing stays stable.
Plain-text discovery output includes `quality=...`, `trust=...`,
`primary_metric=...`, and a first `reason=...`, while JSON output keeps the full
nested `artifact_quality` payload plus the complete metric table.

Pick the latest matching run:

```bash
pyimgano-runs latest --root runs --kind suite --dataset mvtec --json
pyimgano-runs latest --root runs --min-quality audited --json
pyimgano-runs latest --root runs --same-environment-as runs/run_a --json
pyimgano-runs latest --root runs --same-target-as runs/run_a --json
pyimgano-runs latest --root runs --kind robustness --same-robustness-protocol-as runs/robust_a --json
```

Use `--min-quality` to hide partial runs before discovery, `--same-environment-as`
when you only trust apples-to-apples comparisons from the same dependency /
hardware / runtime fingerprint, and `--same-target-as` when you want discovery
results restricted to the same checked `dataset` / `category` target before you
even reach the `compare` gate.

## Compare Runs

Compare two or more saved runs:

```bash
pyimgano-runs compare runs/20260317_101010_mvtec_suite_industrial-v4_bottle \
  runs/20260317_111010_mvtec_vision_patchcore_bottle
```

JSON output is useful for dashboards and scripts:

```bash
pyimgano-runs compare runs/run_a runs/run_b --json
```

The comparison payload includes:

- run summaries (`dataset`, `category`, `model_or_suite`, `timestamp_utc`)
- run-level `dataset_readiness`, `dataset_readiness_status`, and `dataset_issue_codes`
- best-effort environment fingerprint from `environment.json`
- aggregated metric ranges (`min` / `max`) across the selected runs
- an `evaluation_contract` block describing primary metric, directionality, and comparability hints

Use a saved run as a baseline and fail automation on regressions:

```bash
pyimgano-runs compare runs/run_a runs/run_b \
  --baseline runs/run_a \
  --metric auroc \
  --fail-on-regression \
  --json
```

Require split comparability before accepting a candidate:

```bash
pyimgano-runs compare runs/run_a runs/run_b \
  --baseline runs/run_a \
  --require-same-split \
  --json
```

Require dataset/category consistency too:

```bash
pyimgano-runs compare runs/run_a runs/run_b \
  --baseline runs/run_a \
  --require-same-target \
  --json
```

Require environment compatibility too:

```bash
pyimgano-runs compare runs/run_a runs/run_b \
  --baseline runs/run_a \
  --require-same-environment \
  --json
```

Require robustness protocol compatibility too:

```bash
pyimgano-runs compare runs/robust_a runs/robust_b \
  --baseline runs/robust_a \
  --require-same-robustness-protocol \
  --json
```

When `--baseline` is set, the payload also includes:

- metric direction metadata (currently `higher_is_better` for benchmark metrics)
- `evaluation_contract.primary_metric` / `ranking_metric` so downstream automation knows which
  metric the comparison is centered around
- per-run `delta_vs_baseline`
- per-run status (`baseline`, `improved`, `unchanged`, `regressed`, `missing`)
- `summary.total_regressions` plus machine-readable `regression_gate`,
  `comparability_gates`, `blocking_flags`, `verdict`, `primary_metric*`,
  `primary_metric_statuses`, `primary_metric_deltas`, `trust_*`,
  `candidate_verdicts`, `candidate_blocking_reasons`, and
  `candidate_comparability_gates`, `baseline_dataset_readiness`, and
  `candidate_dataset_readiness` fields for CI or release gating
  - when no baseline is provided, `baseline_checked=false`,
    `regression_gate=unchecked`, and `verdict=informational`
  - candidate `candidate_incompatibility_digest` entries also include
    `dataset_readiness_status` / `dataset_issue_codes` when those signals are present
- split compatibility metadata:
  - baseline split fingerprint
  - per-run split status (`baseline`, `matched`, `mismatched`, `missing`, `unchecked`)
  - summary counts for matched / mismatched / missing runs
- target compatibility metadata:
  - baseline `dataset` / `category`
  - per-run target status (`baseline`, `matched`, `mismatched`, `missing`, `unchecked`)
  - field-level dataset/category status plus summary counts
- environment compatibility metadata:
  - baseline environment fingerprint
  - per-run environment status (`baseline`, `matched`, `mismatched`, `missing`, `unchecked`)
  - summary counts for matched / mismatched / missing runs
- baseline trust normalization metadata:
  - `trust_comparison`
  - normalized `gate`, `status`, `reason`, `degraded_by`, and `audit_refs`
  - lets CI consume baseline trust posture without reconstructing it from nested run quality payloads
- robustness protocol compatibility metadata:
  - `robustness_protocol_comparison`
  - baseline protocol signature (`corruption_mode`, `conditions`, `severities`, `input_mode`, `resize`)
  - per-run protocol status (`baseline`, `matched`, `mismatched`, `missing`, `unchecked`)
  - `mismatch_fields` for protocol drifts that make robustness runs non-comparable

Plain-text compare output mirrors the operator-critical parts of that contract:
it prints the chosen `comparison_primary_metric`, its direction and baseline
value, baseline run `quality` / `trust`, then per-candidate primary-metric
status / delta, structured per-run briefs at the top, plus a single
`comparability_gates: ...` summary, `comparison_blocking_flags=...`, and an
overall `comparison_verdict=...`, plus per-candidate
`candidate_verdict.<run>=...`, `candidate_blocking_reasons.<run>=...`, and
`candidate_comparability_gates.<run>=...`, plus
`baseline_dataset_readiness_status=...`, `baseline_dataset_issue_codes=...`,
`candidate_dataset_readiness_status.<run>=...`, and
`candidate_dataset_issue_codes.<run>=...`, plus `comparison_trust_gate=...`,
`comparison_trust_status=...`, `comparison_trust_reason=...`,
`comparison_trust_degraded_by=...`, and `comparison_trust_ref.*=...` so you can
distinguish “comparable enough to evaluate” from “baseline trust is still only
partial” before the detailed compatibility sections.

`--require-same-split` returns exit code `1` when the baseline has no split
fingerprint or any non-baseline run is split-incompatible. This is the safest
mode for leaderboard, release, and CI comparisons.

`--require-same-target` returns exit code `1` when the baseline has no checked
target fields or any non-baseline run disagrees on `dataset` / `category`.

`--require-same-environment` returns exit code `1` when the baseline has no
environment fingerprint or any non-baseline run disagrees on it.

`--require-same-robustness-protocol` returns exit code `1` when the baseline has
no checked robustness protocol or any non-baseline run changes corruption mode,
conditions, severities, input mode, or resize.

## Recommended Workflow

1. Run a benchmark suite or workbench recipe with `--save-run`
2. Pick the current reference with `pyimgano-runs latest --json`
3. Compare candidate runs with `pyimgano-runs compare ... --baseline ... --require-same-split --require-same-target --require-same-environment`
4. Use `pyimgano-runs compare ... --json` when preparing benchmark updates,
   release notes, or internal dashboards
