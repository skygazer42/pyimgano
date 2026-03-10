# Parallel Algorithm Families Design

**Date:** 2026-03-10

**Objective**

Expand `pyimgano` with eight new anomaly-detection algorithm families in parallel, using isolated git worktrees and a single integration branch that absorbs shared registry, discovery, documentation, and verification changes before merging back to `main`.

## Scope

This wave targets eight algorithm families that are relatively recent and not already clearly covered by the current `pyimgano` registry:

1. `VisionAD` - training-free search-based few-shot anomaly detection
2. `UniVAD` - training-free unified few-shot anomaly detection
3. `FiLo++` - fine-grained visual-language prompting with deformable localization
4. `AdaCLIP` - hybrid prompt adaptation for zero-shot anomaly detection
5. `AA-CLIP` - anomaly-aware CLIP for zero-shot anomaly detection
6. `One-to-Normal` - anomaly personalization via one-to-normal transformation
7. `LogSAD` - training-free logical + structural anomaly detection
8. `AnoGen` - few-shot anomaly-driven generation for downstream AD

## Source Basis

These selections were verified against primary sources on 2026-03-10:

- VisionAD: arXiv `2025-04-16`, "Search is All You Need for Few-shot Anomaly Detection"
- UniVAD: CVPR 2025 open-access paper
- FiLo++: arXiv `2025-01-17`
- AdaCLIP: arXiv `2024-07-22`
- AA-CLIP: arXiv `2025-03-09`
- One-to-Normal: arXiv `2025-02-03`
- LogSAD: arXiv `2025-03-24`
- AnoGen: arXiv `2025-05-14`

## Constraints

- `pyimgano` registry and discovery must remain lightweight at import time.
- Optional heavy dependencies must remain gated behind runtime import checks.
- Each track must be testable without requiring the upstream research environment.
- Shared-file conflicts must be minimized; parallel tracks should mostly edit disjoint files.
- The final merge target is a single integration branch, not direct merges from eight branches into `main`.

## Architecture

### Branching Model

- Base branch: `feat/algo-families-2026q1`
- One feature branch + one worktree per algorithm family
- All feature branches merge into `feat/algo-families-2026q1`
- Only after integration verification passes does `feat/algo-families-2026q1` merge into `main`

### Code Organization

Each family gets:

- a dedicated module under `pyimgano/models/`
- a dedicated focused test file under `tests/`
- metadata and tags aligned to the existing registry/discovery contracts

Shared changes are intentionally centralized:

- `pyimgano/models/__init__.py`
- `pyimgano/discovery.py`
- `pyimgano/cli.py`
- `pyimgano/infer_cli.py`
- `docs/MODEL_INDEX.md`
- `docs/SOTA_ALGORITHMS.md`

### Implementation Pattern

Every family should follow the current `pyimgano` production pattern instead of reproducing the original research stack verbatim:

- prefer lazy imports
- accept injectable embedders/backends where possible
- expose stable `fit`, `decision_function`, and optional pixel-map APIs
- degrade cleanly when optional dependencies are missing
- ship smoke tests that validate registry behavior and deterministic scoring

### Two-Layer Delivery Strategy

Because several 2025 methods depend on large VLM/MLLM or diffusion stacks, delivery is split into two layers:

1. `pyimgano-native adapter layer`
   - production-safe wrapper
   - optional dependency gating
   - registry/discovery integration
   - deterministic tests using fake or injected components

2. `upstream-faithful enhancement layer`
   - deeper parity with the original paper
   - optional advanced code paths
   - heavier end-to-end tests guarded as optional or best-effort

This keeps the first merge operational even if full paper fidelity takes longer.

## Worktree Strategy

The repo already ignores `.worktrees/`, so project-local hidden worktrees are the lowest-friction option.

Planned worktrees:

- `.worktrees/algo-families-2026q1`
- `.worktrees/visionad-search-fsad`
- `.worktrees/univad-unified-fsad`
- `.worktrees/filopp-vlm-localization`
- `.worktrees/adaclip-hybrid-prompts`
- `.worktrees/aaclip-anomaly-aware`
- `.worktrees/one-to-normal-personalization`
- `.worktrees/logsad-logical-structural`
- `.worktrees/anogen-fewshot-generation`

## Risk Areas

### Shared registry collisions

`pyimgano/models/__init__.py` is a merge hotspot because `_MODEL_MODULE_ALLOWLIST` must know about new modules.

Mitigation:

- create shared placeholder modules on the integration branch first
- branch the eight feature worktrees only after placeholders exist
- keep family-specific logic inside family-specific modules

### Optional dependency sprawl

Several chosen families touch CLIP, VLM, diffusion, or detection backends.

Mitigation:

- reuse existing extras where possible (`clip`, `torch`, `diffusion`)
- add new extras only if a dependency set is clearly distinct
- keep tests injection-based and offline-safe

### Research-code mismatch

Some upstream repositories are not designed for `pyimgano`'s unified API or offline CI expectations.

Mitigation:

- implement production-oriented adapters, not brittle raw ports
- treat paper-faithful parity as iterative hardening, not the first acceptance gate

## Acceptance Criteria

- eight new family modules exist and register cleanly
- registry discovery works without importing heavy optional deps
- every family has at least one deterministic smoke test
- optional dependency errors contain actionable install hints
- integration branch passes focused smoke and registry verification
- documentation reflects the new families and their status

## Recommendation

Implement a shared foundation on the integration branch first, then dispatch parallel work per family. This is the highest-throughput path that still keeps merge risk under control.
