# Train Export Deploy-Bundle Follow-up (Design)

**Date:** 2026-04-05

## Goal

Continue the `pyimgano-train` export hardening work by thinning deploy-bundle assembly
after the recent fit-only checkpoint persistence fix. This pass keeps the public train
and bundle contracts stable while making the deploy-bundle export stage easier to audit,
test, and extend.

## Context / Problem

The recent checkpoint fix proved that the deploy-bundle path is still a high-risk seam:

- the export path must copy supporting run artifacts into `deploy_bundle/`
- the infer-config payload must be rewritten from run-relative references into
  bundle-relative references
- `artifact_quality` metadata must be patched in a way that stays consistent with the
  generated bundle manifest

Some of this logic has already moved into
[`pyimgano.services.train_export_helpers`](/data/pyimgano/pyimgano/services/train_export_helpers.py),
but the top-level deploy-bundle exporter in
[`pyimgano.services.train_service`](/data/pyimgano/pyimgano/services/train_service.py)
still mixes multiple concerns in one function:

- fixed-file copying
- bundle payload mutation
- manifest-driven metadata patching
- final file persistence order

That concentration is now the most obvious remaining risk on the train-export side.

## Non-goals

- No new CLI flags.
- No schema changes in `infer_config.json`, `bundle_manifest.json`, or `handoff_report.json`.
- No changes to bundle directory names or file names.
- No broad redesign of `pyimgano.reporting.deploy_bundle`.
- No shift of export orchestration out of `train_service.py` in this pass.

## Approaches Considered

### A) Leave `_export_deploy_bundle(...)` mostly as-is

Pros:

- smallest diff
- near-zero short-term refactor risk

Cons:

- keeps the most failure-prone export stage concentrated in one function
- future contract tweaks will keep landing in the same high-risk block

### B) Continue extracting deploy-bundle assembly helpers from `train_service` (recommended)

Pros:

- directly addresses the most active export seam
- keeps the public contract unchanged
- fits naturally after the recent checkpoint regression fix
- improves helper-level testability without forcing a larger architecture move

Cons:

- requires careful regression coverage to avoid subtle metadata drift

### C) Jump to deploy-bundle validator refactoring first

Pros:

- improves validation readability
- already has helper scaffolding in place

Cons:

- does not reduce the concentration in the export path that just produced the recent bug
- weakens continuity with the latest checkpoint-related work

## Recommended Direction

Choose approach B.

The next pass should keep
[`_export_deploy_bundle(...)`](/data/pyimgano/pyimgano/services/train_service.py#L165)
as a thin orchestration shell and move the remaining internal mechanics behind narrow,
contract-focused helpers.

## Proposed Design

### 1) Keep `train_service` as the orchestration boundary

`train_service.py` should continue to own the export stage order:

1. create bundle directory
2. resolve source infer-config
3. copy supporting run artifacts
4. rewrite and patch the bundle payload
5. persist infer-config, handoff report, and manifest
6. patch infer-config metadata from the computed manifest

The service should orchestrate these stages, not implement all low-level details inline.

### 2) Extract supporting-file copy logic into `train_export_helpers`

The helper layer should own copying bundle sidecars such as:

- `report.json`
- `config.json`
- `environment.json`
- `calibration_card.json`
- `operator_contract.json`

This extraction isolates file-copy behavior from payload mutation behavior and makes it
easier to test optional-artifact handling independently.

### 3) Extract initial bundle payload patching into `train_export_helpers`

The helper layer should also own the bundle-local payload patch that happens before path
rewriting and manifest generation. This includes:

- rewriting `artifact_quality.audit_refs` for copied bundle-local artifacts
- stamping `artifact_quality.deploy_refs.bundle_manifest`
- setting `has_deploy_bundle`
- setting `has_bundle_manifest`
- clearing `required_bundle_artifacts_present`
- clearing `bundle_artifact_roles`

This logic is currently correct but spread inline in `train_service.py`, which makes it
harder to read and harder to regression-test at the helper seam.

### 4) Preserve the current manifest-driven patch flow

The existing flow where infer-config is written, handoff-report is built, manifest is
computed, infer-config metadata is patched from the manifest, and then the manifest is
written should stay unchanged. That ordering is part of the current deploy-bundle
contract and already has good downstream coverage.

### 5) Add seam-specific regression coverage

The most valuable new tests in this pass are:

- helper tests for supporting-file copy behavior
- helper tests for initial bundle payload patch behavior
- integration tests proving `run_train_request(...)` still produces the same supporting
  files and metadata after extraction
- targeted regressions proving the fit-only checkpoint path still survives the thinner
  export implementation

## Success Criteria

- `_export_deploy_bundle(...)` becomes materially thinner and easier to read.
- Supporting-file copy and initial payload patching each have helper-level tests.
- Existing deploy-bundle JSON fields and artifact layout remain unchanged.
- The fit-only checkpoint bundle flow continues to pass end-to-end.

## Risks

- Helper extraction could accidentally change which optional artifacts are copied.
- Bundle payload patching could drift from the existing `artifact_quality` contract.
- Manifest-generation ordering could be disturbed if persistence is refactored too
  aggressively.

These risks are acceptable if the extraction stays narrow and verification remains focused
on the current export contract.
