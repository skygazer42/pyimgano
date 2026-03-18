# Evaluation / Calibration / Postprocess / Deploy Gap Closure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the next major gap between `pyimgano` and top-tier anomaly detection packages by making evaluation outputs more authoritative, calibration more auditable, postprocessing more production-oriented, deployment artifacts more trustworthy, and the overall package easier to operate and trust in industrial settings.

**Architecture:** Build on the existing `workbench`, `reporting`, `services`, `runs_cli`, `infer_cli`, and JSON artifact surfaces instead of adding a new framework. Prefer additive contracts, strict validators, exportable artifacts, and CLI-first workflows that remain offline-safe and easy to verify in CI.

**Tech Stack:** Python 3.9+, NumPy, argparse, pytest, JSON artifacts, existing `pyimgano.reporting`, `pyimgano.services`, `pyimgano.workbench`, and deployment/export utilities.

---

## Stream A: Evaluation Authority And Metric Contracts

### Task 1: Add failing tests for stricter evaluation summary metadata
- Files:
  - Modify: `tests/test_run_index.py`
  - Modify: `tests/test_workbench_report_dataset_summary.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py tests/test_workbench_report_dataset_summary.py -q`

### Task 2: Record metric directionality and comparability hints in run summaries
- Files:
  - Modify: `pyimgano/reporting/run_index.py`
  - Modify: `pyimgano/workbench/category_report.py`
- Verify:
  - `pytest --no-cov tests/test_run_index.py tests/test_workbench_report_dataset_summary.py -q`

### Task 3: Add failing tests for benchmark/export evaluation contracts
- Files:
  - Modify: `tests/test_benchmark_config.py`
  - Modify: `tests/test_suite_export_metadata.py`
- Verify:
  - `pytest --no-cov tests/test_benchmark_config.py tests/test_suite_export_metadata.py -q`

### Task 4: Export evaluation contract metadata into suite and benchmark outputs
- Files:
  - Modify: `pyimgano/reporting/benchmark_config.py`
  - Modify: `pyimgano/reporting/suite_export.py`
- Verify:
  - `pytest --no-cov tests/test_benchmark_config.py tests/test_suite_export_metadata.py -q`

### Task 5: Document evaluation authority workflow
- Files:
  - Modify: `docs/BENCHMARK_PUBLICATION.md`
  - Modify: `docs/RUN_COMPARISON.md`
  - Modify: `docs/WORKBENCH.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/BENCHMARK_PUBLICATION.md').read_text(encoding='utf-8')"`

## Stream B: Calibration Audit And Threshold Governance

### Task 6: Add failing tests for richer calibration card threshold context
- Files:
  - Modify: `tests/test_calibration_card.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py tests/test_workbench_export_infer_config.py -q`

### Task 7: Add threshold context, rejection policy, and score distribution metadata to calibration cards
- Files:
  - Modify: `pyimgano/reporting/calibration_card.py`
  - Modify: `pyimgano/services/workbench_service.py`
- Verify:
  - `pytest --no-cov tests/test_calibration_card.py tests/test_workbench_export_infer_config.py -q`

### Task 8: Add failing tests for calibration governance checks in run quality
- Files:
  - Modify: `tests/test_run_quality.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py -q`

### Task 9: Surface calibration completeness and governance warnings in run quality reports
- Files:
  - Modify: `pyimgano/reporting/run_quality.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py tests/test_calibration_card.py -q`

### Task 10: Document calibration audit and review policy
- Files:
  - Modify: `docs/CALIBRATION_AUDIT.md`
  - Modify: `docs/CLI_REFERENCE.md`
  - Modify: `docs/INDUSTRIAL_FASTPATH.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/CALIBRATION_AUDIT.md').read_text(encoding='utf-8')"`

## Stream C: Postprocess And Defect Businessization

### Task 11: Add failing tests for richer defect postprocess metadata
- Files:
  - Modify: `tests/test_inference_api.py`
  - Modify: `tests/test_infer_context_service.py`
- Verify:
  - `pytest --no-cov tests/test_inference_api.py tests/test_infer_context_service.py -q`

### Task 12: Export stable defect filtering and morphology metadata
- Files:
  - Modify: `pyimgano/inference/api.py`
  - Modify: `pyimgano/services/infer_context_service.py`
- Verify:
  - `pytest --no-cov tests/test_inference_api.py tests/test_infer_context_service.py -q`

### Task 13: Add failing tests for defect rejection/triage summaries
- Files:
  - Modify: `tests/test_infer_continue_service.py`
  - Modify: `tests/test_inference_service.py`
- Verify:
  - `pytest --no-cov tests/test_infer_continue_service.py tests/test_inference_service.py -q`

### Task 14: Add reject/triage summary payloads to inference outputs
- Files:
  - Modify: `pyimgano/services/infer_continue_service.py`
  - Modify: `pyimgano/services/inference_service.py`
- Verify:
  - `pytest --no-cov tests/test_infer_continue_service.py tests/test_inference_service.py -q`

### Task 15: Document production defect postprocess policy
- Files:
  - Modify: `docs/INDUSTRIAL_INFERENCE.md`
  - Modify: `docs/CLI_REFERENCE.md`
  - Modify: `README.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/INDUSTRIAL_INFERENCE.md').read_text(encoding='utf-8')"`

## Stream D: Deploy And Infer Artifact Completeness

### Task 16: Add failing tests for deploy bundle audit references
- Files:
  - Modify: `tests/test_deploy_bundle_manifest.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_workbench_export_infer_config.py -q`

### Task 17: Extend deploy bundle manifest with audit references and artifact roles
- Files:
  - Modify: `pyimgano/reporting/deploy_bundle.py`
  - Modify: `pyimgano/services/train_service.py`
- Verify:
  - `pytest --no-cov tests/test_deploy_bundle_manifest.py tests/test_workbench_export_infer_config.py -q`

### Task 18: Add failing tests for infer-config deployment metadata completeness
- Files:
  - Modify: `tests/test_validate_infer_config_cli.py`
  - Modify: `tests/test_workbench_export_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_validate_infer_config_cli.py tests/test_workbench_export_infer_config.py -q`

### Task 19: Carry deploy/audit completeness metadata into infer configs
- Files:
  - Modify: `pyimgano/inference/config.py`
  - Modify: `pyimgano/services/workbench_service.py`
  - Modify: `pyimgano/inference/validate_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_validate_infer_config_cli.py tests/test_workbench_export_infer_config.py -q`

### Task 20: Document deployable artifact contract
- Files:
  - Modify: `docs/INDUSTRIAL_FASTPATH.md`
  - Modify: `docs/MODEL_CARDS.md`
  - Modify: `docs/WEIGHTS.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/INDUSTRIAL_FASTPATH.md').read_text(encoding='utf-8')"`

## Stream E: Run Quality, Audit, And Trust Contracts

### Task 21: Add failing tests for missing audit references and degraded trust states
- Files:
  - Modify: `tests/test_run_quality.py`
  - Modify: `tests/test_runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py tests/test_runs_cli.py -q`

### Task 22: Expand run quality levels with warnings, references, and trust reasons
- Files:
  - Modify: `pyimgano/reporting/run_quality.py`
  - Modify: `pyimgano/runs_cli.py`
- Verify:
  - `pytest --no-cov tests/test_run_quality.py tests/test_runs_cli.py -q`

### Task 23: Add failing tests for publication/repo audit contracts
- Files:
  - Modify: `tests/test_publication_quality.py`
  - Modify: `tests/test_tools_audit_publication_contract.py`
- Verify:
  - `pytest --no-cov tests/test_publication_quality.py tests/test_tools_audit_publication_contract.py -q`

### Task 24: Tighten publication quality and repo audit helpers
- Files:
  - Modify: `pyimgano/reporting/publication_quality.py`
  - Modify: `tools/audit_publication_contract.py`
  - Modify: `tools/audit_repo_links.py`
- Verify:
  - `pytest --no-cov tests/test_publication_quality.py tests/test_tools_audit_publication_contract.py tests/test_tools_audit_repo_links.py -q`

### Task 25: Document trust contract expectations
- Files:
  - Modify: `docs/PUBLISHING.md`
  - Modify: `docs/CAPABILITY_ASSESSMENT.md`
  - Modify: `README.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/PUBLISHING.md').read_text(encoding='utf-8')"`

## Stream F: Robustness Reporting And Comparability

### Task 26: Add failing tests for robustness export comparability fields
- Files:
  - Modify: `tests/test_robustness_export.py`
  - Modify: `tests/test_robustness_service.py`
- Verify:
  - `pytest --no-cov tests/test_robustness_export.py tests/test_robustness_service.py -q`

### Task 27: Export robustness protocol metadata and comparable summary fields
- Files:
  - Modify: `pyimgano/reporting/robustness_export.py`
  - Modify: `pyimgano/reporting/robustness_summary.py`
  - Modify: `pyimgano/services/robustness_service.py`
- Verify:
  - `pytest --no-cov tests/test_robustness_export.py tests/test_robustness_service.py -q`

### Task 28: Add failing tests for robustness CLI quality surfacing
- Files:
  - Modify: `tests/test_robust_cli_smoke.py`
- Verify:
  - `pytest --no-cov tests/test_robust_cli_smoke.py -q`

### Task 29: Surface robustness comparability metadata in CLI output
- Files:
  - Modify: `pyimgano/robust_cli.py`
- Verify:
  - `pytest --no-cov tests/test_robust_cli_smoke.py tests/test_robustness_export.py -q`

### Task 30: Document robustness benchmarking expectations
- Files:
  - Modify: `docs/ROBUSTNESS_BENCHMARK.md`
  - Modify: `docs/BENCHMARK_PUBLICATION.md`
  - Modify: `docs/COMPARISON.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/ROBUSTNESS_BENCHMARK.md').read_text(encoding='utf-8')"`

## Stream G: Usability And Shortest-Path CLI

### Task 31: Add failing tests for shorter root CLI discovery paths
- Files:
  - Modify: `tests/test_root_cli.py`
  - Modify: `tests/test_infer_cli_smoke.py`
- Verify:
  - `pytest --no-cov tests/test_root_cli.py tests/test_infer_cli_smoke.py -q`

### Task 32: Improve `pyimgano --help` and `pyimgano -- list` discovery flow
- Files:
  - Modify: `pyimgano/root_cli.py`
  - Modify: `pyimgano/cli.py`
- Verify:
  - `pytest --no-cov tests/test_root_cli.py tests/test_infer_cli_smoke.py -q`

### Task 33: Add failing tests for fastest audited-workflow examples
- Files:
  - Modify: `tests/test_examples_configs_load.py`
  - Modify: `tests/test_tools_audit_audited_fastpath_docs.py`
- Verify:
  - `pytest --no-cov tests/test_examples_configs_load.py tests/test_tools_audit_audited_fastpath_docs.py -q`

### Task 34: Add shortest audited workflow configs and CLI guidance
- Files:
  - Modify: `examples/configs/industrial_adapt_audited.json`
  - Modify: `docs/QUICKSTART.md`
  - Modify: `docs/CLI_REFERENCE.md`
- Verify:
  - `pytest --no-cov tests/test_examples_configs_load.py tests/test_tools_audit_audited_fastpath_docs.py -q`

### Task 35: Document shortest-path operator journeys
- Files:
  - Modify: `README.md`
  - Modify: `docs/QUICKSTART.md`
  - Modify: `docs/WORKBENCH.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/QUICKSTART.md').read_text(encoding='utf-8')"`

## Stream H: Trust Signals, Auditability, And Publication Readiness

### Task 36: Add failing tests for stronger publication/trust signal artifacts
- Files:
  - Modify: `tests/test_publication_quality.py`
  - Modify: `tests/test_tools_audit_benchmark_configs.py`
- Verify:
  - `pytest --no-cov tests/test_publication_quality.py tests/test_tools_audit_benchmark_configs.py -q`

### Task 37: Add explicit benchmark provenance and trust-signal summaries
- Files:
  - Modify: `pyimgano/reporting/publication_quality.py`
  - Modify: `pyimgano/reporting/benchmark_config.py`
- Verify:
  - `pytest --no-cov tests/test_publication_quality.py tests/test_tools_audit_benchmark_configs.py -q`

### Task 38: Add failing tests for package-level trust metadata surfaces
- Files:
  - Modify: `tests/test_weights_cli.py`
  - Modify: `tests/test_validate_infer_config_cli.py`
- Verify:
  - `pytest --no-cov tests/test_weights_cli.py tests/test_validate_infer_config_cli.py -q`

### Task 39: Surface trust metadata in validation and weights tooling
- Files:
  - Modify: `pyimgano/weights_cli.py`
  - Modify: `pyimgano/inference/validate_infer_config.py`
- Verify:
  - `pytest --no-cov tests/test_weights_cli.py tests/test_validate_infer_config_cli.py -q`

### Task 40: Write final trust/readiness guidance
- Files:
  - Modify: `docs/PUBLISHING.md`
  - Modify: `docs/MODEL_CARDS.md`
  - Modify: `docs/CAPABILITY_ASSESSMENT.md`
- Verify:
  - `python -c "from pathlib import Path; Path('docs/CAPABILITY_ASSESSMENT.md').read_text(encoding='utf-8')"`

## Suggested Execution Order

1. Stream B (`校准审计`) and Stream E (`run quality / trust contracts`)
2. Stream D (`deploy / infer artifact completeness`)
3. Stream G (`更短 CLI / 最短路径`) and Stream C (`postprocess`)
4. Stream F (`robustness`) and Stream A (`evaluation authority`)
5. Stream H (`trust signals / publication readiness`)

## Initial Batch For This Session

1. Extend `tests/test_calibration_card.py` and `tests/test_workbench_export_infer_config.py` for richer calibration context and exported audit metadata.
2. Extend `tests/test_run_quality.py` for calibration governance checks and trust warnings.
3. Extend `tests/test_deploy_bundle_manifest.py` for audit-reference entries in deploy bundle manifests.
4. Implement the minimal reporting/service changes to make those tests pass.
