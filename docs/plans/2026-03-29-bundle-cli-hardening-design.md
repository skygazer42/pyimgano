# Bundle CLI Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of `pyimgano-bundle` around its two public entry paths:

- `pyimgano-bundle validate`
- `pyimgano-bundle run`

This pass preserves existing command names, JSON field names, reason codes, and exit-code
semantics while making the internal validate/run logic easier to reason about.

## Context / Problem

`pyimgano-bundle` sits close to the deployment boundary. That makes it high-risk because
it combines:

- deploy-bundle manifest validation
- infer-config validation
- weights/model-card audit checks
- input source normalization
- offline inference orchestration
- batch gate / acceptance-style result summarization

The main risk is not only crashes; it is semantic drift:

- validate and run paths can interpret the same bundle differently
- text output can drift from JSON payloads
- reason-code and status assembly can become inconsistent

Today, `bundle_cli.py` carries substantial logic for both validation and run reporting,
which makes future changes expensive and harder to verify.

## Non-goals

- No new `pyimgano-bundle` commands or flags.
- No public reason-code renames.
- No JSON field renames for validate/run payloads.
- No redesign of the deploy-bundle manifest schema in this pass.
- No rewrite of the whole deployment stack.

## Approaches Considered

### A) Thin only text output

Move a few text summaries out of `bundle_cli.py` and keep most logic in place.

Pros:
- small diff
- low short-term risk

Cons:
- leaves the validate/run decision logic concentrated in the CLI
- only improves the shell of the problem

### B) Harden validate/run payload assembly and thin the CLI (recommended)

Keep the public interface stable, but separate:

- structured validate payload assembly
- structured run report assembly
- text rendering
- CLI exit-code mapping

Pros:
- directly addresses the highest-risk deployment entry path
- best leverage under a compatibility constraint
- makes future bundle/reporting work cheaper

Cons:
- requires focused regression tests because validate/run outputs are user-facing contracts

### C) Large deploy subsystem redesign

Pros:
- potentially the cleanest long-term architecture

Cons:
- too much churn for a compatibility-preserving hardening pass
- drags schema/reporting concerns into one large change

## Proposed Design

### 1) Keep `bundle_cli.py` as the public shell, not the source of truth

`pyimgano.bundle_cli` should mainly:

- parse arguments
- dispatch `validate` vs `run`
- print text or emit JSON
- map structured payloads onto the existing exit-code behavior

### 2) Treat validate and run as two structured payload pipelines

The validate path should first assemble a structured validation payload covering:

- infer-config validation
- bundle manifest validation
- weights/model-card audit readiness
- reason code mapping
- status derivation

The run path should first assemble a structured run report covering:

- input source summary
- processed/error/anomaly/reject counters
- batch gate results
- output artifact status
- reason code mapping

### 3) Introduce small internal helpers for validate/run rendering

This pass does not need a large new package. It does need smaller helpers for:

- validate summary lines
- run summary lines
- batch gate summary formatting
- reason/status helpers shared inside the CLI module or a narrow helper module

### 4) Keep schema logic in existing reporting/validation modules

This pass should not duplicate deploy-bundle schema logic already owned by:

- `pyimgano.reporting.deploy_bundle`
- `pyimgano.inference.validate_infer_config`
- `pyimgano.weights.bundle_audit`

Instead, `bundle_cli.py` should consume their structured results and translate them into
stable validate/run payloads.

### 5) Expand regression tests around validate/run contracts

The most valuable tests here are:

- validate JSON payload and reason-code tests
- run JSON/report payload tests
- text summary stability tests
- architecture boundary tests if new helper modules are introduced

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: validate-path helper extraction

- normalize infer-config / manifest / weight-audit payload handling
- reduce validate-path inline logic in `bundle_cli.py`

### Stream B: run-path helper extraction

- normalize run-report and batch-gate summaries
- reduce run-path inline logic in `bundle_cli.py`

### Stream C: rendering and contract tests

- add/expand validate/run rendering tests
- add focused JSON/text regression coverage

### Stream D: docs and boundary reinforcement

- update `pyimgano-bundle` docs for validate/run contracts
- add architecture/export expectations if helper modules are introduced

## Error Handling Strategy

The validate and run paths should each own structured states first:

- missing/invalid bundle
- invalid infer-config
- unsupported pixel outputs
- failing batch gates
- weight-audit failures

The CLI should not recompute these meanings after the fact. It should render them and
apply the already-established exit-code rules.

## Testing Strategy

The implementation will prioritize:

1. validate helper/payload contract tests
2. run helper/payload contract tests
3. JSON/text/exit-code regression tests
4. docs and architecture contract checks where useful

This is more valuable than broad smoke tests because the main risk is semantic drift in
deployment validation and execution summaries.

## Success Criteria

- `pyimgano-bundle validate/run` stay compatible.
- validate/run payload assembly becomes easier to reason about.
- `bundle_cli.py` becomes thinner and more orchestration-focused.
- reason-code and status behavior are better protected by regression tests.
- future bundle/deploy changes require touching fewer unrelated lines.

## Risks

- validate/run text output may have incidental formatting dependencies
- helper extraction can expose hidden coupling between validate and run code paths
- report assembly and schema validation may share assumptions that only become visible after cleanup

These risks are acceptable if changes stay small and regression tests stay focused on the
existing bundle contract.
