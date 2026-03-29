# Train Export Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of the `pyimgano-train` export path, focusing on:

- `--export-infer-config`
- `--export-deploy-bundle`

This pass preserves existing CLI flags, JSON field names, and exit-code semantics while
making the internal export pipeline easier to reason about.

## Context / Problem

`pyimgano-train` is now more than a recipe runner. In production-oriented workflows it also:

- exports `artifacts/infer_config.json`
- emits `operator_contract.json`
- emits `calibration_card.json`
- optionally assembles `deploy_bundle/`
- rewrites referenced artifact paths into deploy-bundle-relative form
- updates deploy-bundle manifest and artifact-quality metadata

That makes `pyimgano.services.train_service` a high-risk concentration point. The main
risk is not only a crash; it is semantic drift in exported artifacts:

- infer-config export can diverge from bundle export
- bundle path rewriting can behave differently across relative and absolute checkpoints
- artifact-quality metadata can be partially updated in one export path but not another

The package already has strong coverage around these flows, which is a signal that the
engineering risk is in internal complexity, not in missing major features.

## Non-goals

- No new `pyimgano-train` flags.
- No public export field renames.
- No changes to recipe semantics.
- No broad workbench or reporting redesign in this pass.
- No full rewrite of deploy-bundle schema logic.

## Approaches Considered

### A) Thin only the CLI

Tidy `pyimgano.train_cli` while leaving export logic mostly concentrated in
`train_service.py`.

Pros:
- small diff
- low short-term risk

Cons:
- leaves the highest-risk export logic mostly untouched
- limited payoff

### B) Harden the export pipeline inside `train_service` and keep the CLI thin (recommended)

Separate the export path into clearer stages:

- config loading / recipe execution
- infer-config export
- optional artifact export
- deploy-bundle assembly

Pros:
- directly targets the most fragile part of the train/deploy bridge
- best leverage while preserving compatibility
- makes future deploy-contract changes cheaper

Cons:
- requires careful regression testing because export artifacts are already relied on

### C) Restructure train, workbench, and deploy-bundle logic together

Pros:
- potentially the cleanest long-term architecture

Cons:
- too much churn for a compatibility-preserving hardening pass
- drags multiple subsystems into one change

## Proposed Design

### 1) Keep `train_cli.py` as the public entry shell

`pyimgano.train_cli` should mainly:

- parse args
- branch between list/info/dry-run/preflight/run modes
- call `train_service`
- emit text or JSON
- preserve current exit codes

### 2) Make export stages explicit inside `train_service`

The service layer should read more clearly as:

1. load config and apply overrides
2. validate export eligibility
3. run recipe
4. export infer-config and optional artifacts
5. optionally assemble deploy bundle

These stages already exist implicitly; this pass makes them easier to see and test.

### 3) Separate infer-config export from deploy-bundle assembly

The infer-config export stage should own:

- `infer_config.json`
- optional `operator_contract.json`
- optional `calibration_card.json`

The deploy-bundle stage should own:

- bundle directory creation
- artifact copying
- path rewriting
- manifest generation
- bundle-specific artifact-quality patching

This reduces the chance that deploy-specific assumptions leak back into infer-config export.

### 4) Prefer helper extraction over broad module shuffling

This pass does not need a new large package. It does need smaller helpers for:

- export request validation
- optional artifact emission
- bundle path resolution and rewriting
- bundle metadata patching

### 5) Expand regression tests around export contract seams

The most valuable tests here are:

- path rewriting for relative and absolute checkpoints
- infer-config / bundle manifest metadata consistency
- CLI delegation tests around export branches
- architecture tests if new helper modules are introduced

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: infer-config export helper extraction

- normalize infer-config and optional artifact export helpers
- reduce mixed responsibilities in `train_service.py`

### Stream B: deploy-bundle assembly helper extraction

- isolate path rewriting and bundle metadata patching
- reduce deploy-bundle logic concentration

### Stream C: train CLI thinning

- keep export branches orchestration-focused
- avoid reintroducing export details into the CLI

### Stream D: docs and boundary reinforcement

- update export docs where needed
- add architecture/export expectations if helper modules are introduced

## Error Handling Strategy

The service layer should own structured failure boundaries such as:

- invalid export request
- missing run_dir from recipe output
- missing referenced checkpoint artifact
- invalid path rewrite / bundle escape
- bundle assembly failure

The CLI should not recompute these meanings after the fact. It should route them through
the established CLI error path.

## Testing Strategy

The implementation will prioritize:

1. helper-level export contract tests
2. end-to-end infer-config / deploy-bundle regression tests
3. train CLI delegation tests
4. docs and architecture contract checks where helpful

This is more valuable than adding only broad smoke tests because the main risk is semantic
drift in exported artifacts and bundle assembly.

## Success Criteria

- `--export-infer-config` and `--export-deploy-bundle` stay compatible.
- `train_service.py` reads as a clearer staged export pipeline.
- infer-config export and bundle assembly are easier to test independently.
- future deploy-contract changes require touching fewer unrelated lines.

## Risks

- export helpers may have hidden coupling to recipe output shapes
- path rewriting changes can break absolute/relative checkpoint handling if not covered
- bundle metadata patching may rely on incidental ordering in current code

These risks are acceptable if each change remains small and regression tests stay focused
on the current export contract.
