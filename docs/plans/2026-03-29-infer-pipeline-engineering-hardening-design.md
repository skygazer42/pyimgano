# Infer Pipeline Engineering Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of the highest-risk public inference path in `pyimgano`
without changing any existing public API or CLI behavior.

This design focuses on the internal path behind:

- direct `pyimgano-infer --model ...`
- config-backed `pyimgano-infer --infer-config ...`
- run-backed `pyimgano-infer --from-run ...`

The objective is to make the inference path easier to maintain, easier to reason about,
and less regression-prone while preserving full compatibility.

## Context / Problem

`pyimgano` already has broad model, workbench, and reporting coverage, but the inference
entry path carries a disproportionate amount of engineering risk:

- `infer_cli.py` is large and mixes parsing, orchestration, and request assembly.
- The path from CLI input to detector loading to runtime planning spans multiple modules
  with overlapping responsibilities.
- Similar metadata is threaded through multiple layers (`postprocess_summary`,
  defect thresholds, artifact payloads, decision summaries), which raises the risk of
  shape drift and partial regressions.
- The package already depends on architecture-boundary tests, which signals that internal
  layering discipline is important to long-term maintainability.

Compared with mainstream adjacent projects, the gap is not feature count but boundary
clarity:

- `anomalib` emphasizes separated training / inference / deployment flows.
- `PyOD` emphasizes a stable, predictable public API.
- method-specific repos like `patchcore-inspection` are narrower, so their execution paths
  are easier to reason about.

For `pyimgano`, the best leverage is to harden the inference orchestration path rather
than spread small changes across unrelated areas.

## Non-goals

- No public CLI flag removals, renames, or semantic changes.
- No public Python API removals or incompatible return-shape changes.
- No mandatory new runtime dependencies.
- No model algorithm rewrites.
- No broad re-architecture of the whole package.
- No attempt to optimize all engineering surfaces in one pass.

## Approaches Considered

### A) Broad package-wide quality pass

Spread 40 tasks across facades, reporting, docs, weights, discovery, and inference.

Pros:
- Covers more visible package areas.

Cons:
- Dilutes effort.
- Produces many shallow changes.
- Leaves the highest-risk execution path only partially improved.

### B) Public surface hardening only

Focus on `__init__` facades, `__all__`, import cost, and public export audits.

Pros:
- Very low compatibility risk.
- Good for package hygiene.

Cons:
- Does not directly reduce the operational risk of the inference path.
- Improves the shell of the package more than the internals users rely on most.

### C) Inference pipeline engineering hardening (recommended)

Concentrate the 40 tasks on `infer_cli`, `infer_*` services, and the closely coupled
`workbench` runtime boundary.

Pros:
- Highest leverage under a full-compatibility constraint.
- Reduces regression risk where the public CLI and internal runtime meet.
- Keeps future feature work cheaper by making the boundary more explicit.

Cons:
- Less breadth than a package-wide cleanup pass.
- Requires careful sequencing because this path already has substantial test coverage.

## Proposed Design

### 1) Keep the current public behavior, narrow the internal responsibilities

`infer_cli.py` should remain the stable public entrypoint, but its job should be reduced to:

- argument parsing
- branch selection (`direct`, `--infer-config`, `--from-run`)
- service orchestration
- final error rendering

Business rules and state assembly should move toward service-layer request/result contracts.

### 2) Treat the inference path as five explicit stages

The hardened path should read as:

1. Parse CLI arguments into normalized request values.
2. Build or restore an inference context.
3. Load a detector from direct/config/run-backed inputs.
4. Prepare a runtime plan.
5. Execute inference and materialize outputs/artifacts.

These stages already exist implicitly; this work makes them explicit and more uniform.

### 3) Strengthen the context boundary

`pyimgano.services.infer_context_service` should be the single place that converts
workbench or infer-config state into a stable `ConfigBackedInferContext`.

This includes:

- checkpoint path resolution
- threshold restoration inputs
- defects / prediction / tiling payload normalization
- postprocess summary construction
- warning collection

The CLI should not reconstruct these payloads ad hoc once the context object exists.

### 4) Strengthen the load boundary

`pyimgano.services.infer_load_service` should own the transition from explicit context to
loaded detector.

That means:

- aligning direct-load and config-backed-load request shapes where practical
- keeping checkpoint enforcement and threshold restoration in one place
- making final `model_kwargs` reconstruction explicit and testable

The output of this stage should make detector provenance and effective options easier to
inspect in tests.

### 5) Strengthen the runtime-plan boundary

`pyimgano.services.infer_runtime_service` should remain the only place that resolves:

- whether maps are enabled
- whether postprocess is active and from which source
- how pixel thresholds are obtained
- what provenance metadata is attached to the runtime decision

The runtime plan should expose summary fields in a single normalized shape so downstream
writers do not have to infer missing context.

### 6) Unify execution/output metadata rules

`inference_service`, `infer_continue_service`, `infer_output_service`, and
`infer_artifact_service` should continue to do separate jobs, but they should consume a
more consistent set of summaries and payload conventions.

Priority fields to keep consistent:

- `postprocess_summary`
- `decision_summary`
- runtime defect-threshold provenance
- artifact request / output metadata
- continue-on-error triage summaries

### 7) Expand architecture tests instead of relying on convention

This work should lean on small boundary tests rather than verbal discipline alone:

- `__all__` contract checks for selected boundary modules
- import whitelist checks for service modules
- request/result dataclass contract tests
- JSON-serializable payload shape tests

The goal is to make future regressions mechanically visible.

## Execution Streams

The 40 tasks will be grouped into five streams of eight tasks each:

### Stream A: CLI thinning and input normalization

- reduce non-orchestration helpers in `infer_cli.py`
- centralize parsing/normalization seams where practical
- preserve identical user-visible behavior

### Stream B: Context, load, and runtime contract alignment

- tighten `ConfigBackedInferContext`
- align load request/result behavior
- normalize runtime-plan summaries and provenance

### Stream C: Execution and output consistency

- align metadata threading across execution, fallback, output, and artifacts
- reduce duplicated payload assembly logic

### Stream D: Boundary enforcement

- expand architecture tests
- lock down `__all__`, import boundaries, and selected public contracts

### Stream E: Documentation and regression safety

- update inference docs/help surfaces where needed
- add narrow integration coverage for direct/config/run-backed flows
- improve failure-path tests

## Error Handling Strategy

User-visible CLI failures should keep the current overall style, but internal failures
should be more attributable to a single stage:

- parsing failure
- context restoration failure
- detector load failure
- runtime-plan failure
- execution failure
- artifact/output failure

Where possible, tests should target those stages directly instead of asserting only on the
outer CLI shell.

## Testing Strategy

The implementation will prioritize four classes of tests:

1. Boundary contract tests
2. Metadata shape consistency tests
3. Failure-path tests
4. Thin end-to-end inference path tests

This is preferred over a large number of broad smoke tests because the main risk here is
state drift across layers, not missing a single happy path.

## Success Criteria

- All public `pyimgano-infer` entry paths remain compatible.
- The inference path is split more clearly into context, load, runtime, execution, and
  output stages.
- Metadata payloads become more consistent across direct/config/run-backed inference.
- Architecture tests cover more of the selected inference boundary.
- Future changes to inference behavior require touching fewer unrelated call sites.

## Risks

- Existing tests may encode incidental behavior from the current coupling.
- Moving logic out of `infer_cli.py` can accidentally duplicate or reorder validation.
- Boundary cleanup may expose latent inconsistencies between direct inference and
  config-backed inference.

These risks are acceptable if each change stays small and remains covered by focused tests.
