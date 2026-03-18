# Benchmark Publication Checklist

Use this checklist before publishing or citing a `pyimgano` benchmark result.

## Reproducibility

- Use an official benchmark preset from `benchmarks/configs/official_*.json` when possible.
- Discover the current built-in presets with `pyimgano-benchmark --list-official-configs`.
- Inspect one preset before publishing with
  `pyimgano-benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json`.
- Keep the executed config JSON under version control.
- Record the exact dataset root, category, and split policy used.
- Keep `report.json`, `config.json`, and `environment.json` together.
- Export `leaderboard.*` and `leaderboard_metadata.json` for suite runs.
- Prefer official benchmark presets so `leaderboard_metadata.json` also carries the
  benchmark citation payload for the exported run.
- Review `leaderboard_metadata.json` before publication:
  - `artifact_quality`
  - `evaluation_contract`
  - `publication_ready`
  - `exported_files`
- If `exported_files` declares `model_card_json` or `weights_manifest_json`, make sure those
  artifacts validate too; `pyimgano-runs publication` now checks them.
- Prefer a machine gate before publishing:

```bash
pyimgano-benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json
pyimgano-runs publication /path/to/suite_export --json
```

Repository maintainers should also keep the synthetic publication contract audit green:

```bash
python tools/audit_publication_contract.py
```

## Auditability

- Confirm `environment.json` contains a stable environment fingerprint.
- Confirm exported benchmark metadata declares an `evaluation_contract` so dashboards and
  downstream publication tools can interpret metric directionality and ranking semantics
  without hardcoding them.
- If using a deployable checkpoint, keep the corresponding model card and weights manifest.
- If you attach those files to a suite export, prefer a model card with `weights.manifest_entry`
  so the package can be checked against the manifest deterministically.
- Prefer offline-safe runs (`--no-pretrained`) unless pretrained weights are explicitly required.
- Note any optional extras that were unavailable and caused baselines to be skipped.

## Publication Hygiene

- Include the exact command or benchmark config used.
- State whether results came from a single model run or a curated suite.
- Attach the generated `leaderboard_metadata.json` for suite publications.
- Fail the release check when `pyimgano-runs publication ...` is not `status=ready`.
- Use `pyimgano-runs compare --baseline ... --require-same-split --require-same-target --require-same-environment`
  before release-grade comparisons or benchmark update claims.
- Link to `docs/FIRST_TIER_ROADMAP.md` when discussing ongoing maturity work.
