# Second-Tier Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the next layer of gaps between `pyimgano` and top-tier anomaly detection packages by improving prediction semantics, regression-aware run comparison, calibration auditability, split comparability, and artifact completeness.

**Architecture:** Build on the existing `BaseDetector`, `workbench`, `reporting`, `runs_cli`, and `infer_config` contracts instead of introducing a new framework. Prefer additive JSON artifacts, small helpers, CLI-first workflows, and offline-safe behavior that can be verified in CI and reused in industrial deployments.

**Tech Stack:** Python 3.9+, NumPy, argparse, existing `pyimgano.reporting`, `pyimgano.services`, `pyimgano.workbench`, JSON artifacts, pytest.

---

## Research Snapshot (2026-03-17)

- `anomalib` currently emphasizes benchmark jobs, inferencers, and export/deployment utilities in its official docs.
- `PyOD` currently exposes richer prediction semantics such as probability conversion, confidence estimation, and rejection-aware prediction flows in its official docs/examples.
- `pyimgano` is already strong on industrial IO, run artifacts, deploy bundles, and model breadth; the next leverage is making those artifacts easier to trust, compare, and gate automatically.

---

## Stream A: Prediction Semantics (Confidence + Rejection)

### Task 1: Add failing tests for native confidence APIs
- Files:
  - Create: `tests/test_base_detector_confidence.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py -q`

### Task 2: Add failing tests for sklearn adapter confidence/rejection surface
- Files:
  - Modify: `tests/test_sklearn_adapter.py`
- Verify:
  - `pytest --no-cov tests/test_sklearn_adapter.py -q`

### Task 3: Add base helper to derive normalized outlier rank/confidence from train and test scores
- Files:
  - Modify: `pyimgano/models/base_detector.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py -q`

### Task 4: Implement `predict(..., return_confidence=True)` in native `BaseDetector`
- Files:
  - Modify: `pyimgano/models/base_detector.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py -q`

### Task 5: Add `predict_confidence(...)` convenience API
- Files:
  - Modify: `pyimgano/models/base_detector.py`
  - Modify: `pyimgano/sklearn_adapter.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py tests/test_sklearn_adapter.py -q`

### Task 6: Add `predict_with_rejection(...)` with explicit reject label semantics
- Files:
  - Modify: `pyimgano/models/base_detector.py`
  - Modify: `pyimgano/sklearn_adapter.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py tests/test_sklearn_adapter.py -q`

### Task 7: Implement `predict_proba(..., return_confidence=True)` parity for native detectors
- Files:
  - Modify: `pyimgano/models/base_detector.py`
  - Modify: `tests/test_base_detector_predict_proba.py`
- Verify:
  - `pytest --no-cov tests/test_base_detector_predict_proba.py tests/test_base_detector_confidence.py -q`

### Task 8: Document confidence/rejection behavior and constraints
- Files:
  - Modify: `README.md`
  - Modify: `docs/COMPARISON.md`
  - Modify: `docs/CORE_MODELS.md`
- Verify:
  - `python -m py_compile pyimgano/models/base_detector.py pyimgano/sklearn_adapter.py`

## Stream B: Regression-Aware Run Comparison

### Task 9: Add failing tests for metric delta extraction vs baseline run
- Files:
  - Modify: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 10: Add failing tests for CLI regression gating flags
- Files:
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 11: Add metric direction metadata helper for comparable run metrics
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 12: Compute deltas, regressions, and missing-metric status in run comparisons
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 13: Add `pyimgano-runs compare --baseline`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Modify: `pyimgano/reporting/run_index.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py tests/test_run_index.py -q`

### Task 14: Add `--metric`, `--max-regressions`, and `--fail-on-regression`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 15: Add `pyimgano-runs latest`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 16: Document regression-aware comparison workflow
- Files:
  - Modify: `docs/RUN_COMPARISON.md`
  - Modify: `docs/CLI_REFERENCE.md`
  - Modify: `.github/workflows/ci.yml`
- Verify:
  - `python -m py_compile pyimgano/runs_cli.py pyimgano/reporting/run_index.py`

## Stream C: Calibration Auditability

### Task 17: Add failing tests for calibration card builder
- Files:
  - Create: `tests/test_calibration_card.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py -q`

### Task 18: Add calibration card schema/helper module
- Files:
  - Create: `pyimgano/reporting/calibration_card.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py -q`

### Task 19: Record score distribution summaries and threshold context in calibration cards
- Files:
  - Modify: `pyimgano/reporting/calibration_card.py`
  - Modify: `pyimgano/services/workbench_service.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py -q`

### Task 20: Persist `calibration_card.json` in workbench run artifacts
- Files:
  - Modify: `pyimgano/services/train_service.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_workbench_export_infer_config.py tests/test_calibration_card.py -q`

### Task 21: Reference calibration artifacts from deploy bundles
- Files:
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Modify: `pyimgano/services/train_service.py`
  - Modify: `tests/test_deploy_bundle_manifest.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_calibration_card.py -q`

### Task 22: Carry calibration summary into exported infer-config metadata
- Files:
  - Modify: `pyimgano/services/workbench_service.py`
  - Modify: `pyimgano/inference/config.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_workbench_export_infer_config.py -q`

### Task 23: Add validation helper/tests for calibration card payloads
- Files:
  - Modify: `pyimgano/reporting/calibration_card.py`
  - Modify: `tests/test_calibration_card.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py -q`

### Task 24: Document calibration audit workflow
- Files:
  - Modify: `docs/WORKBENCH.md`
  - Modify: `docs/CLI_REFERENCE.md`
  - Create: `docs/CALIBRATION_AUDIT.md`
- Verify:
  - `python -m py_compile pyimgano/reporting/calibration_card.py`

## Stream D: Split Fingerprints And Comparability

### Task 25: Add failing tests for dataset split fingerprint helper
- Files:
  - Create: `tests/test_split_fingerprint.py`
- Verify:
  - `pytest --no-cov tests/test_split_fingerprint.py -q`

### Task 26: Add split fingerprint helper for workbench/manifest style runs
- Files:
  - Create: `pyimgano/reporting/split_fingerprint.py`
- Verify:
  - `pytest --no-cov tests/test_split_fingerprint.py -q`

### Task 27: Record split fingerprint in workbench reports
- Files:
  - Modify: `pyimgano/workbench/category_report.py`
  - Modify: `tests/test_workbench_report_dataset_summary.py`
- Verify:
  - `pytest --no-cov tests/test_workbench_report_dataset_summary.py tests/test_split_fingerprint.py -q`

### Task 28: Record split fingerprint in suite export metadata
- Files:
  - Modify: `pyimgano/reporting/suite_export.py`
  - Modify: `tests/test_suite_export_metadata.py`
- Verify:
  - `pytest --no-cov tests/test_suite_export_metadata.py tests/test_split_fingerprint.py -q`

### Task 29: Extend run index summaries with split fingerprint
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Modify: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py -q`

### Task 30: Add `pyimgano-runs list --same-split-as`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py -q`

### Task 31: Add docs for split-aware comparison and publication guidance
- Files:
  - Modify: `docs/BENCHMARK_PUBLICATION.md`
  - Modify: `docs/RUN_COMPARISON.md`
  - Modify: `docs/MANIFEST_DATASET.md`
- Verify:
  - `python -m py_compile pyimgano/reporting/split_fingerprint.py`

### Task 32: Add audit/test coverage for split comparability metadata
- Files:
  - Modify: `tests/test_cli_baseline_suites_v16.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_cli_baseline_suites_v16.py tests/test_workbench_export_infer_config.py -q`

## Stream E: Artifact Completeness And Fast-Path Trust

### Task 33: Add failing tests for run artifact completeness checker
- Files:
  - Create: `tests/test_run_quality.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py -q`

### Task 34: Add helper to evaluate run artifact completeness/quality
- Files:
  - Create: `pyimgano/reporting/run_quality.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py -q`

### Task 35: Surface artifact completeness in `pyimgano-runs list --json`
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Modify: `tests/test_run_index.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py tests/test_run_quality.py -q`

### Task 36: Add `pyimgano-runs quality`
- Files:
  - Modify: `pyimgano/runs_cli.py`
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_runs_cli.py tests/test_run_quality.py -q`

### Task 37: Add a short audited fast-path guide
- Files:
  - Create: `docs/INDUSTRIAL_FASTPATH.md`
  - Modify: `README.md`
  - Modify: `docs/QUICKSTART.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/INDUSTRIAL_FASTPATH.md').read_text(encoding='utf-8')"`

### Task 38: Add example config geared for audited deploy/export flow
- Files:
  - Create: `examples/configs/industrial_adapt_audited.json`
  - Modify: `tests/test_examples_configs_load.py`
- Verify:
  - `pytest --no-cov tests/test_examples_configs_load.py -q`

### Task 39: Add repo-level audit for docs/examples mentioning audited artifact set
- Files:
  - Create: `tools/audit_audited_fastpath_docs.py`
  - Create: `tests/test_tools_audit_audited_fastpath_docs.py`
  - Modify: `.github/workflows/ci.yml`
- Verify:
  - `pytest --no-cov tests/test_tools_audit_audited_fastpath_docs.py -q`

### Task 40: Final verification for the second-tier batch
- Verify:
  - `pytest --no-cov tests/test_base_detector_confidence.py tests/test_base_detector_predict_proba.py tests/test_sklearn_adapter.py tests/test_run_index.py tests/test_runs_cli.py tests/test_calibration_card.py tests/test_split_fingerprint.py tests/test_workbench_report_dataset_summary.py tests/test_suite_export_metadata.py tests/test_workbench_export_infer_config.py tests/test_deploy_bundle_manifest.py tests/test_run_quality.py tests/test_examples_configs_load.py tests/test_tools_audit_audited_fastpath_docs.py -q`
  - `python tools/audit_benchmark_configs.py`
  - `python tools/audit_repo_links.py`
  - `git diff --check`
