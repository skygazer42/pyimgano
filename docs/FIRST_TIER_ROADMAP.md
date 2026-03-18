# First-Tier Roadmap

This roadmap tracks the highest-value work needed to make `pyimgano` feel like
a first-tier anomaly detection package, not just a capable codebase.

## 1. Benchmark authority

- Maintain official, versioned benchmark presets under `benchmarks/configs/`
- Require run artifacts (`report.json`, `environment.json`, `leaderboard.*`) for published claims
- Keep reproducibility issues on a dedicated template so regressions are actionable

## 2. Deployment-grade auditability

- Treat `infer_config.json` and `deploy_bundle/` as first-class release artifacts
- Require stable metadata, schema versions, and provenance in exported bundles
- Keep offline-safe defaults and explicit weight provenance

## 3. Assetized model operations

- Standardize local weight manifests and model cards
- Make source, license, and runtime requirements explicit
- Keep model assets reproducible and reviewable without network access

## 4. Trust signals

- Enforce repository link hygiene in CI
- Keep contributing, publishing, and benchmark docs aligned with the current repo
- Prefer a small number of precise workflows over sprawling docs that drift

## 5. Community feedback loops

- Separate generic bugs from benchmark reproducibility issues
- Push users toward Discussions for support and design questions
- Make external contributions easier with current commands and release guidance
