# Deploy Bundle Validation Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of `pyimgano.reporting.deploy_bundle`, with the primary
focus on `validate_deploy_bundle_manifest(...)` and a secondary focus on reducing rule
duplication with `build_deploy_bundle_manifest(...)`.

This pass preserves manifest schema shape and public field names while making build and
validate rules easier to reason about.

## Context / Problem

`deploy_bundle.py` is now a shared contract point across multiple already-hardened paths:

- train export / deploy bundle assembly
- bundle validate/run
- runs quality / acceptance
- doctor readiness checks

That makes it high-risk for semantic drift. The main risks are:

- build and validate can encode the same rule in slightly different ways
- artifact ref / role / required-presence checks can become hard to audit
- operator-contract digest and consistency rules can grow into a long hard-to-read block
- weight-audit validation can remain “correct” but become fragile to maintain

The danger is not only breakage. It is also accepting subtly invalid manifests or
rejecting valid ones because related rules are too spread out.

## Non-goals

- No new deploy-bundle schema version.
- No public field renames.
- No redesign of `pyimgano-bundle` CLI semantics in this pass.
- No broad changes to run-quality or acceptance logic except where validation helpers are shared.

## Approaches Considered

### A) Tidy only the build side

Focus on `build_deploy_bundle_manifest(...)` and leave validation mostly intact.

Pros:
- smaller diff

Cons:
- does not reduce the highest current risk, which is validation rule complexity

### B) Harden validate first and share selected computed helpers with build (recommended)

Keep the schema stable, but restructure validation into smaller, clearly scoped helpers:

- entries / refs / roles validation
- required-flag validation
- operator-contract validation
- weight-audit validation

Pros:
- best leverage for a shared contract point
- directly addresses the most error-prone area
- still allows small build/validate deduplication where useful

Cons:
- needs careful regression testing because many tests already depend on exact validation semantics

### C) Large deploy-bundle subsystem redesign

Pros:
- potentially the cleanest long-term architecture

Cons:
- too much churn for a contract-preserving hardening pass
- would couple CLI, reporting, and validation changes unnecessarily

## Proposed Design

### 1) Keep `validate_deploy_bundle_manifest(...)` as the top-level validator shell

The validator should still be the public entrypoint, but it should read as:

1. structural manifest checks
2. entry/ref/role checks
3. required-presence checks
4. operator-contract checks
5. optional weight-audit checks

### 2) Extract validation rule helpers by concern

This pass should use smaller helpers for:

- manifest entry collection and hash checks
- artifact ref / role validation
- required flag validation
- operator-contract digest and consistency validation
- weight-audit validation

### 3) Share computed helpers only where it meaningfully reduces duplication

Some helpers are naturally shared between build and validate:

- artifact role construction
- artifact digest construction
- threshold summary / evaluation summary computation

This pass should only share or refactor these when it improves clarity, not as a
mechanical “dedupe everything” exercise.

### 4) Preserve error ordering and message stability where practical

Many tests assert fragments of error messages. This pass should preserve stable wording and
ordering as much as possible, unless a helper extraction exposes a clearly better boundary
and the tests are explicitly updated.

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: validation helper contract coverage

- add or tighten helper-level tests for refs, roles, required flags, and digests

### Stream B: operator-contract validation hardening

- extract and isolate digest/presence/consistency validation helpers

### Stream C: build/validate shared helper cleanup

- share low-risk computed helpers where they improve clarity

### Stream D: docs, boundaries, and verification

- update docs if needed
- add architecture/export expectations if new helper modules are introduced
- run focused deploy-bundle regression suites

## Error Handling Strategy

Each helper should validate one concern and return explicit errors for that concern only.
The top-level validator should aggregate those error lists in a stable order.

That means:

- helper returns errors
- top-level validator concatenates them
- no ad hoc mixing of unrelated concerns inside one block

## Testing Strategy

The implementation will prioritize:

1. helper-level validation contract tests
2. `tests/test_deploy_bundle_manifest.py` and `tests/test_deploy_bundle_contract_v1.py`
3. related `run_quality`, `runs_cli`, and `run_acceptance` regression subsets
4. docs and architecture checks where useful

This is more valuable than broad smoke tests because the main risk is subtle rule drift in
deploy-bundle validation.

## Success Criteria

- deploy-bundle schema stays compatible.
- `validate_deploy_bundle_manifest(...)` becomes easier to audit by concern.
- operator-contract and artifact-ref validation logic becomes more focused.
- future deploy-bundle rule changes require touching fewer unrelated lines.

## Risks

- existing tests may encode incidental error ordering
- deduping computed helpers can accidentally shift build/validate semantics
- operator-contract rules may have hidden coupling to run-quality and acceptance checks

These risks are acceptable if changes stay small and regression tests stay focused on the
current deploy-bundle contract.
