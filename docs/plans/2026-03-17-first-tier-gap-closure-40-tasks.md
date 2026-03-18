# First-Tier Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the highest-leverage gaps between `pyimgano` and first-tier anomaly detection packages by improving benchmark reproducibility, run comparability, deploy artifact auditability, model asset validation, and external trust signals.

**Architecture:** Build on existing `workbench`, `reporting`, and `weights` primitives instead of introducing a new framework layer. Keep all additions lightweight, offline-safe, JSON-friendly, and compatible with the current CLI-first workflow. Prefer additive metadata, small service helpers, and focused tooling that improves auditability without forcing users into a new stack.

**Tech Stack:** Python 3.9+, argparse, JSON artifacts, existing `pyimgano.reporting`, `pyimgano.services`, `pyimgano.weights`, GitHub Actions.

---

## Stream A: Benchmark Provenance And Reproducibility

### Task 1: Add benchmark config validation helper
- Files:
  - Create: `pyimgano/reporting/benchmark_config.py`
  - Test: `tests/test_benchmark_config.py`
- Verify:
  - `pytest --no-cov tests/test_benchmark_config.py -q`

### Task 2: Record official benchmark config path in benchmark reports
- Files:
  - Modify: `pyimgano/cli.py`
  - Modify: `pyimgano/pipelines/run_suite.py`
  - Test: `tests/test_cli_baseline_suites_v16.py`
- Verify:
  - `pytest --no-cov tests/test_cli_baseline_suites_v16.py -q`

### Task 3: Export benchmark metadata JSON next to leaderboard tables
- Files:
  - Modify: `pyimgano/reporting/suite_export.py`
  - Test: `tests/test_suite_export_metadata.py`
- Verify:
  - `pytest --no-cov tests/test_suite_export_metadata.py -q`

### Task 4: Include environment fingerprint in suite export metadata
- Files:
  - Modify: `pyimgano/reporting/suite_export.py`
  - Test: `tests/test_suite_export_metadata.py`
- Verify:
  - `pytest --no-cov tests/test_suite_export_metadata.py -q`

### Task 5: Include benchmark citation payload for exported official runs
- Files:
  - Modify: `pyimgano/reporting/suite_export.py`
  - Test: `tests/test_suite_export_metadata.py`
- Verify:
  - `pytest --no-cov tests/test_suite_export_metadata.py -q`

### Task 6: Validate official benchmark presets in test suite
- Files:
  - Create: `tests/test_benchmark_configs_official.py`
- Verify:
  - `pytest --no-cov tests/test_benchmark_configs_official.py -q`

### Task 7: Add benchmark publication checklist doc
- Files:
  - Create: `docs/BENCHMARK_PUBLICATION.md`
- Verify:
  - `python -m py_compile pyimgano/reporting/benchmark_config.py`

### Task 8: Add CI audit step for official benchmark presets
- Files:
  - Modify: `.github/workflows/ci.yml`
  - Create: `tools/audit_benchmark_configs.py`
  - Test: `tests/test_tools_audit_benchmark_configs.py`
- Verify:
  - `pytest --no-cov tests/test_tools_audit_benchmark_configs.py -q`

## Stream B: Run Discovery And Comparability

### Task 9: Add run summary loader for workbench/suite runs
- Files:
  - Create: `pyimgano/reporting/run_index.py`
  - Test: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 10: Normalize report/environment payload extraction
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Test: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 11: Add run comparison payload builder
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Test: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 12: Add sorting/filtering helpers for run listings
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Test: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 13: Add `pyimgano-runs list`
- Files:
  - Create: `pyimgano/runs_cli.py`
  - Modify: `pyproject.toml`
  - Test: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 14: Add `pyimgano-runs compare --json`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Test: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 15: Add human-readable run comparison output
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Test: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 16: Document run indexing/comparison workflow
- Files:
  - Create: `docs/RUN_COMPARISON.md`
  - Modify: `README.md`
  - Modify: `docs/WORKBENCH.md`
- Verify:
  - `python -m py_compile pyimgano/runs_cli.py`

## Stream C: Deploy Bundle Auditability

### Task 17: Add deploy bundle manifest schema helper
- Files:
  - Create: `pyimgano/reporting/deploy_bundle.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 18: Record copied file entries with relative destination paths
- Files:
  - Modify: `pyimgano/services/train_service.py`
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 19: Record sha256 and file size in deploy bundle manifest
- Files:
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 20: Record source run metadata and environment fingerprint in deploy bundle manifest
- Files:
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Modify: `pyimgano/services/train_service.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 21: Persist bundle manifest into `deploy_bundle/`
- Files:
  - Modify: `pyimgano/services/train_service.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 22: Add validator for deploy bundle manifest
- Files:
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Test: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py -q`

### Task 23: Cover `train_cli --export-deploy-bundle` manifest behavior end-to-end
- Files:
  - Modify: `tests/test_train_cli_smoke.py`
  - Modify: `tests/test_integration_workbench_train_then_infer.py`
- Verify:
  - `pytest --no-cov tests/test_train_cli_smoke.py tests/test_integration_workbench_train_then_infer.py -q`

### Task 24: Document deploy bundle manifest contract
- Files:
  - Modify: `docs/WORKBENCH.md`
  - Modify: `docs/CLI_REFERENCE.md`
  - Modify: `docs/WEIGHTS.md`
- Verify:
  - `python -m py_compile pyimgano/services/train_service.py`

## Stream D: Weights And Model Card Assetization

### Task 25: Add model card validation module
- Files:
  - Create: `pyimgano/weights/model_card.py`
  - Test: `tests/test_model_card_validation.py`
- Verify:
  - `pytest --no-cov tests/test_model_card_validation.py -q`

### Task 26: Define a JSON-friendly model card schema
- Files:
  - Modify: `pyimgano/weights/model_card.py`
  - Test: `tests/test_model_card_validation.py`
- Verify:
  - `pytest --no-cov tests/test_model_card_validation.py -q`

### Task 27: Strengthen weights manifest recommendations for source/license/runtime metadata
- Files:
  - Modify: `pyimgano/weights/manifest.py`
  - Test: `tests/test_weights_manifest_v1.py`
- Verify:
  - `pytest --no-cov tests/test_weights_manifest_v1.py -q`

### Task 28: Add `pyimgano-weights template manifest`
- Files:
  - Modify: `pyimgano/weights_cli.py`
  - Test: `tests/test_weights_cli.py`
- Verify:
  - `pytest --no-cov tests/test_weights_cli.py -q`

### Task 29: Add `pyimgano-weights template model-card`
- Files:
  - Modify: `pyimgano/weights_cli.py`
  - Modify: `pyimgano/weights/model_card.py`
  - Test: `tests/test_weights_cli.py`
- Verify:
  - `pytest --no-cov tests/test_weights_cli.py -q`

### Task 30: Add `pyimgano-weights validate-model-card`
- Files:
  - Modify: `pyimgano/weights_cli.py`
  - Modify: `pyimgano/weights/model_card.py`
  - Test: `tests/test_weights_cli.py`
- Verify:
  - `pytest --no-cov tests/test_weights_cli.py -q`

### Task 31: Add example model card asset
- Files:
  - Create: `examples/configs/example_model_card.json`
  - Modify: `docs/MODEL_CARDS.md`
- Verify:
  - `python -m py_compile pyimgano/weights/model_card.py`

### Task 32: Document weight/model card workflow in README and docs
- Files:
  - Modify: `README.md`
  - Modify: `docs/WEIGHTS.md`
  - Modify: `docs/MODEL_CARDS.md`
- Verify:
  - `python -m py_compile pyimgano/weights_cli.py`

## Stream E: Trust Signals, Docs, And Community Hygiene

### Task 33: Add stale repo link audit tool
- Files:
  - Create: `tools/audit_repo_links.py`
  - Test: `tests/test_tools_audit_repo_links.py`
- Verify:
  - `pytest --no-cov tests/test_tools_audit_repo_links.py -q`

### Task 34: Fix stale repo links and citations in docs
- Files:
  - Modify: `docs/QUICKSTART.md`
  - Modify: `docs/source/index.rst`
  - Modify: `docs/DEEP_LEARNING_MODELS.md`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 35: Refresh `CONTRIBUTING.md` with current commands and release guidance
- Files:
  - Modify: `CONTRIBUTING.md`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 36: Add benchmark reproducibility issue template
- Files:
  - Create: `.github/ISSUE_TEMPLATE/benchmark_repro.yml`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 37: Add docs for benchmark publication and maintainer checklist
- Files:
  - Modify: `docs/PUBLISHING.md`
  - Modify: `README.md`
  - Modify: `benchmarks/README.md`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 38: Add link-audit and docs-trust checks to CI
- Files:
  - Modify: `.github/workflows/ci.yml`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 39: Add repository trust references to docs landing pages
- Files:
  - Modify: `docs/source/index.rst`
  - Modify: `docs/source/contributing.rst`
- Verify:
  - `python tools/audit_repo_links.py`

### Task 40: Add a concise top-level maturity roadmap for first-tier gap closure
- Files:
  - Create: `docs/FIRST_TIER_ROADMAP.md`
  - Modify: `docs/CAPABILITY_ASSESSMENT.md`
- Verify:
  - `python tools/audit_repo_links.py`

---

## Execution Order

1. Stream C (`deploy bundle`) because it is high-value, scoped, and easy to verify end-to-end.
2. Stream D (`weights/model cards`) because it builds directly on existing CLI and manifest primitives.
3. Stream B (`run discovery`) because it leverages existing run artifacts and improves operability.
4. Stream A (`benchmark provenance`) because it depends on stable run/export metadata.
5. Stream E (`trust signals`) because it should be applied after new docs/tools land.

## Verification Bundle

Run the focused checks after implementation:

```bash
pytest --no-cov \
  tests/test_benchmark_config.py \
  tests/test_suite_export_metadata.py \
  tests/test_benchmark_configs_official.py \
  tests/test_tools_audit_benchmark_configs.py \
  tests/test_run_index.py \
  tests/test_runs_cli.py \
  tests/test_deploy_bundle_manifest.py \
  tests/test_train_cli_smoke.py \
  tests/test_integration_workbench_train_then_infer.py \
  tests/test_model_card_validation.py \
  tests/test_weights_manifest_v1.py \
  tests/test_weights_cli.py \
  tests/test_tools_audit_repo_links.py -q

python tools/audit_benchmark_configs.py
python tools/audit_repo_links.py
ruff check pyimgano tests tools
python -m py_compile \
  pyimgano/reporting/benchmark_config.py \
  pyimgano/reporting/run_index.py \
  pyimgano/reporting/deploy_bundle.py \
  pyimgano/weights/model_card.py \
  pyimgano/runs_cli.py \
  pyimgano/weights_cli.py \
  tools/audit_benchmark_configs.py \
  tools/audit_repo_links.py
```
