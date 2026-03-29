# Workbench Preflight Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of the workbench preflight path, focusing on the
uniform orchestration layer across:

- `pyimgano.workbench.preflight`
- `pyimgano.workbench.preflight_summary`
- `pyimgano.workbench.manifest_preflight`
- `pyimgano.workbench.non_manifest_preflight`

This pass preserves existing preflight results and report shapes while making the
dispatch and early-return flow easier to reason about.

## Context / Problem

Workbench preflight already has a good component split, but the orchestration path is
still a high-risk maintenance area:

- the top-level preflight flow has to route between manifest and non-manifest datasets
- both routes have multiple early-return summary paths
- issue accumulation and category selection have to remain consistent across branches
- preflight behavior is relied on by train/export workflows, so regressions ripple outward

The biggest risk is semantic drift rather than obvious failure:

- one branch can start returning a slightly different summary shape
- a source-validation early exit can bypass an issue path unexpectedly
- top-level orchestration can become the place where too much dataset-specific knowledge leaks

## Non-goals

- No new preflight public APIs.
- No changes to `PreflightReport` field names.
- No broad rewrite of all manifest/non-manifest helper modules.
- No redesign of workbench config parsing.
- No attempt to add new dataset validation rules in this pass.

## Approaches Considered

### A) Touch component helpers directly

Focus on `manifest_record_preflight`, `manifest_preflight_categories`,
`non_manifest_source_validation`, and related leaf helpers.

Pros:
- very local changes

Cons:
- does not improve the orchestration layer where the user-selected risk sits

### B) Harden the orchestration and dispatch layer (recommended)

Treat preflight as a thin orchestrator over already-existing helper boundaries:

- `preflight.py` coordinates
- `preflight_summary.py` dispatches
- `manifest_preflight.py` and `non_manifest_preflight.py` orchestrate their own branches

Pros:
- best leverage under a compatibility constraint
- reduces the chance that orchestration grows new dataset-specific branches
- improves readability without needing a broad rewrite

Cons:
- requires careful regression testing because many existing tests already depend on subtle flow

### C) Full preflight subsystem redesign

Pros:
- potentially the cleanest long-term structure

Cons:
- too much churn for a hardening pass
- would drag many already-factored helpers into one risky change

## Proposed Design

### 1) Keep `preflight.py` as the top-level shell

`run_preflight(...)` should mainly:

- initialize issues
- run model compatibility checks
- delegate dataset-specific summary resolution
- assemble `PreflightReport`

It should not accumulate more dataset-branch knowledge than necessary.

### 2) Keep `preflight_summary.py` as the only dataset dispatch layer

`resolve_workbench_preflight_summary(...)` should remain the single decision point for:

- manifest vs non-manifest routing
- passing through shared issue builder / issue list
- returning a JSON-friendly summary payload

### 3) Treat manifest and non-manifest preflight modules as branch orchestrators

`manifest_preflight.py` and `non_manifest_preflight.py` should each read as:

- resolve source
- early-return if source invalid
- load or select categories/records
- early-return if necessary
- build final branch report

This pass should make that staged flow more explicit and more directly testable.

### 4) Prefer helper extraction only where orchestration still mixes concerns

This pass does not need many new modules. It does need smaller helpers if a function still
mixes:

- source resolution
- early-return summary handling
- category selection
- final branch report assembly

### 5) Expand regression tests around branch dispatch and early-return semantics

The most valuable tests here are:

- dispatch tests in `preflight_summary`
- early-return tests in manifest/non-manifest branch orchestrators
- architecture-boundary tests proving orchestration uses the existing helper seams

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: dispatch and early-return helper coverage

- make preflight branch routing and early exits easier to test directly

### Stream B: orchestration cleanup

- reduce mixed-responsibility patterns in manifest/non-manifest preflight modules

### Stream C: docs and boundary reinforcement

- update any docs that describe `--preflight`
- extend architecture tests if new helper boundaries are introduced

### Stream D: focused verification

- run the preflight-related suite to ensure no regression in train/export-facing behavior

## Error Handling Strategy

The branch-specific modules should own structured early-return summaries:

- invalid manifest source
- invalid non-manifest source
- empty records
- missing/invalid category selection

The top-level preflight shell should not reinterpret those outcomes after the fact. It
should carry them through into the final report.

## Testing Strategy

The implementation will prioritize:

1. branch dispatch contract tests
2. manifest/non-manifest orchestration tests
3. architecture boundary checks
4. focused preflight regression suites

This is more valuable than broad smoke tests because the main risk is subtle flow drift
in preflight orchestration.

## Success Criteria

- `run_preflight(...)` output stays compatible.
- dispatch between manifest and non-manifest paths becomes easier to reason about.
- early-return behavior in branch orchestrators is more directly covered by tests.
- future preflight rule changes require touching fewer orchestration lines.

## Risks

- existing tests may encode incidental control flow
- helper extraction can accidentally reorder issue accumulation
- branch orchestration changes can affect train/export preflight behavior indirectly

These risks are acceptable if changes remain small and regression tests stay focused on
the current preflight contract.
