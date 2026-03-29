# Doctor Readiness / Extras Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of `pyimgano-doctor` for the two highest-risk
responsibility areas:

- environment / extras checks
- run / deploy readiness and acceptance-adjacent diagnostics

This pass preserves existing public CLI behavior, JSON field names, and exit-code
semantics while making the internal structure easier to reason about.

## Context / Problem

`pyimgano-doctor` has grown into a mixed surface:

- environment/runtime metadata
- optional extra validation
- suite readiness checks
- accelerator diagnostics
- run / deploy bundle readiness checks
- dataset-target profiling and recommendation

The highest-risk parts are no longer the raw environment probes themselves, but the
contract around readiness and extras:

- the JSON payload shape is increasingly important for CI and deployment gates
- text output must stay aligned with the same structured payload
- exit codes must reflect the same readiness/extras conclusions as the JSON output

Today, `doctor_cli.py` still contains a substantial amount of text rendering logic, and
`doctor_service.py` contains several different diagnostic domains in one large module.
That combination raises the chance of semantic drift between output modes.

## Non-goals

- No new `pyimgano-doctor` commands or flags.
- No public JSON field renames.
- No exit-code changes.
- No redesign of dataset-target recommendation logic in this pass.
- No broad rewrite of all readiness/reporting modules.

## Approaches Considered

### A) Thin only the CLI

Move a few text-rendering snippets into helpers and leave `doctor_service.py` mostly intact.

Pros:
- small diff
- low short-term risk

Cons:
- leaves the core extras/readiness complexity concentrated in one service file
- improves structure less than needed

### B) Harden extras/readiness helpers and thin the CLI (recommended)

Keep the public CLI stable, but separate:

- structured extras / readiness payload assembly
- text rendering
- CLI exit-code handling

Pros:
- directly addresses the user-selected high-risk areas
- best leverage without touching dataset-target recommendation internals
- makes future readiness additions cheaper

Cons:
- requires focused regression testing because the JSON payload is used as a deployment gate

### C) Full doctor-service domain split, including dataset-target recommendation

Pros:
- potentially the cleanest long-term structure

Cons:
- too much churn for the chosen focus
- risks dragging unrelated recommendation code into this pass

## Proposed Design

### 1) Keep dataset-target recommendation out of scope

This pass should not reorganize dataset-target recommendation logic unless a change is
required to preserve compatibility. The focus stays on extras/readiness surfaces.

### 2) Separate structured readiness/extras payloads from CLI rendering

`pyimgano.services.doctor_service` should remain the source of JSON-ready payloads for:

- `require_extras`
- suite checks
- accelerator checks
- run/deploy readiness

`pyimgano.doctor_cli` should primarily:

- parse args
- call `collect_doctor_payload(...)`
- choose JSON vs text output
- translate already-structured readiness/extras conclusions into exit codes

### 3) Introduce small internal helpers for doctor text summaries

The CLI text path should rely on helper functions for:

- quality / readiness summary lines
- extras summary lines
- publication / acceptance-adjacent readiness summaries when present

This keeps the text path compatible while reducing the chance that one output mode drifts.

### 4) Extract smaller extras/readiness helpers inside the service layer

This pass does not need to split `doctor_service.py` into many top-level files immediately.
It does need to reduce long mixed-responsibility blocks by extracting helpers for:

- extras requirement checks
- readiness target-kind resolution
- run/deploy readiness summary assembly
- accelerator payload normalization

### 5) Expand contract tests around JSON payloads and exit codes

The most important test targets are:

- `collect_doctor_payload(...)` shape and status semantics
- `pyimgano-doctor --json` exit-code behavior for missing extras and invalid readiness
- text output summaries for extras and readiness
- architecture/export expectations for any new helper modules

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: doctor-service extras/readiness helper extraction

- normalize helper boundaries in `doctor_service.py`
- make readiness/extras payload logic easier to test directly

### Stream B: doctor CLI rendering thinning

- move text summary formatting into helper functions
- keep CLI focused on parsing, delegation, and exit codes

### Stream C: contract and regression tests

- expand `tests/test_doctor_service.py`
- expand `tests/test_doctor_cli.py`
- expand targeted accelerator/readiness tests

### Stream D: docs and boundary reinforcement

- update doctor docs for stable extras/readiness contracts
- add architecture expectations for new helper modules if introduced

## Error Handling Strategy

The service layer should own structured readiness/extras states such as:

- missing required extras
- invalid deploy bundle
- blocked run readiness
- unavailable accelerators

The CLI should not recompute those meanings. It should only render them and map them to
stable exit-code behavior.

## Testing Strategy

The implementation will prioritize:

1. extras/readiness helper contract tests
2. doctor CLI JSON/text exit-code regression tests
3. accelerator/readiness path stability tests
4. architecture and docs contract checks where useful

This is more valuable than adding broad smoke tests because the key risk is semantic
drift in readiness/extras conclusions rather than basic command execution.

## Success Criteria

- `pyimgano-doctor` public behavior stays compatible.
- readiness/extras conclusions are more clearly encoded in service-layer helpers.
- CLI text output is less entangled with business rules.
- JSON payload and exit-code behavior are better protected by regression tests.
- future readiness additions require touching fewer unrelated lines.

## Risks

- doctor text output may have incidental formatting dependencies in tests
- helper extraction may expose hidden coupling between readiness payloads and CLI output
- readiness and dataset-target code may share assumptions that become visible only after cleanup

These risks are acceptable if each change stays small and the regression tests remain
focused on stable doctor contracts.
