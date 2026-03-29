# Runs / Reporting Engineering Hardening (Design)

**Date:** 2026-03-29

## Goal

Improve the engineering quality of the `pyimgano-runs` path without changing any
existing public CLI behavior, output field names, or exit codes.

This design focuses on the path behind:

- `pyimgano-runs list`
- `pyimgano-runs compare`
- `pyimgano-runs quality`
- `pyimgano-runs acceptance`

The objective is to make run comparison, trust evaluation, operator-contract status,
and CLI rendering easier to maintain and less regression-prone while preserving full
compatibility.

## Context / Problem

`pyimgano-runs` is now an audit and decision surface, not just a convenience CLI.
That raises the maintenance risk of:

- `pyimgano/reporting/run_index.py`, which has grown into a large concentration of
  comparability, trust, operator-contract, and regression logic.
- `pyimgano/runs_cli.py`, which mixes argparse, business rules, compatibility gates,
  and text formatting.
- subtle drift between JSON output, text summaries, and internal trust/comparability
  judgments.

This is a high-risk surface because failures are often semantic rather than obvious:

- a gate can still render, but mean something different than before
- text output can omit a reason users relied on
- trust/comparability fields can stay present while drifting in logic or shape

The package already treats run quality and acceptance as important release/deploy
signals, so the next leverage point is structure and testability rather than new
features.

## Non-goals

- No new `pyimgano-runs` commands.
- No public field renames in JSON output.
- No CLI flag removals or exit-code changes.
- No broad rewrite of all reporting modules.
- No attempt to unify every `reporting/*` file in one pass.

## Approaches Considered

### A) Thin only the CLI

Move a few formatting helpers out of `runs_cli.py` and leave `run_index.py` mostly as-is.

Pros:
- small diff
- low short-term risk

Cons:
- leaves the main complexity concentration untouched
- only improves the shell of the problem

### B) Harden `run_index` and make `runs_cli` thinner (recommended)

Keep the public surface stable, but split the internal responsibilities more clearly:

- `run_index` owns structured facts and normalized comparison/trust payloads
- `runs_cli` owns argument parsing and rendering only

Pros:
- best leverage under a full-compatibility constraint
- directly reduces the risk of semantic drift in comparison and quality output
- makes future trust/comparability changes cheaper

Cons:
- requires careful regression testing because this path has many user-facing summaries

### C) Large reporting subsystem redesign

Create several new submodules and reshuffle all run/reporting contracts at once.

Pros:
- potentially the cleanest long-term structure

Cons:
- too much churn for a compatibility-constrained hardening pass
- unnecessarily high regression risk

## Proposed Design

### 1) Separate structured judgments from CLI rendering

`pyimgano/reporting/run_index.py` should be the place where structured judgments are made:

- run-summary normalization
- metric direction/comparability hints
- trust gate normalization
- operator-contract and bundle-operator-contract normalization
- candidate comparability gate evaluation
- baseline-vs-candidate structured comparison payloads

`pyimgano/runs_cli.py` should primarily:

- parse flags
- call run/reporting helpers
- choose JSON vs text rendering
- print stable summaries

### 2) Keep `run_index` large in scope, but smaller in units

This pass does not need to split `run_index.py` into many files immediately.
It does need to reduce “long function with mixed responsibilities” patterns by
extracting smaller helpers for:

- trust comparison derivation
- operator contract status normalization
- candidate comparability digest assembly
- comparison recommendation / blocking-flag derivation
- metric display hints

### 3) Treat text summaries as renderers of already-structured payloads

Current text output should remain compatible, but the source of truth should be
structured payloads assembled first and rendered second.

That means helpers that currently both decide and render should be pushed toward:

- decide first
- render second

This reduces the chance that JSON and text paths drift apart.

### 4) Expand contract tests around compare/quality/list flows

This work should lean on targeted regression tests rather than broad smoke only:

- `run_index` helper contract tests
- `runs_cli` JSON payload stability tests
- text-brief regression tests for list/compare/quality summaries
- architecture tests for explicit exports and helper-module boundaries where useful

### 5) Preserve compatibility by making all changes additive or internal

This pass is about moving logic and reducing ambiguity, not redesigning outputs.
If a new internal helper exists, it must produce the same externally visible result
unless the user-facing behavior was already inconsistent and the tests explicitly
pin the intended contract.

## Execution Streams

The implementation plan will be grouped into four streams:

### Stream A: `run_index` trust/comparability hardening

- extract and normalize trust/comparison helpers
- make structured gate evaluation easier to test directly

### Stream B: `runs_cli` thinning

- move formatting/selection helpers out of command handlers where practical
- keep command handlers focused on CLI orchestration

### Stream C: contract and regression tests

- expand `tests/test_run_index.py`
- expand `tests/test_runs_cli.py`
- add focused summary-shape and text-brief tests

### Stream D: docs and boundary reinforcement

- update docs where compare/quality/trust fields are explained
- add/extend architecture tests if new helper modules are introduced

## Error Handling Strategy

The data/reporting layer should own structured states such as:

- missing run
- invalid comparison target
- unchecked vs compatible vs incompatible comparability gates
- missing or degraded trust/operator-contract signals

The CLI should not recompute those meanings. It should only render or escalate them.

## Testing Strategy

The implementation will prioritize:

1. helper-level contract tests
2. compare/list/quality JSON regression tests
3. text-summary regression tests
4. focused architecture/boundary checks where helpful

This is more valuable than adding many new smoke tests because the main risk here
is semantic drift in trust/comparison output, not basic command execution.

## Success Criteria

- `pyimgano-runs` commands remain compatible.
- `run_index.py` becomes easier to reason about in trust/comparability areas.
- `runs_cli.py` becomes thinner and more obviously orchestration-focused.
- JSON and text rendering paths are better protected by regression tests.
- future run-quality / trust / acceptance changes require touching fewer unrelated lines.

## Risks

- text summaries may have incidental formatting dependencies in tests or downstream scripts
- helper extraction can accidentally reorder comparison reasoning
- hidden duplication between `run_index` and `runs_cli` may only become visible after extraction

These risks are acceptable if each change remains small and is covered by focused
contract tests.
