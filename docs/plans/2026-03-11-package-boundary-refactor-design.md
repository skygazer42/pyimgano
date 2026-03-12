# Package Boundary Refactor Design

**Date:** 2026-03-11

**Objective**

Refactor `pyimgano` into a coherent Python package with explicit layer boundaries, a real runtime detector contract, and thinner CLI entrypoints, without breaking the current user-facing CLI surface in the first milestone.

## Problem Summary

The current repository is feature-rich, but its architecture has drifted into a shape where package boundaries are no longer reliable:

- CLI helpers are imported by core orchestration code.
- Runtime behavior is inferred from registry metadata, tags, and `hasattr(...)` checks instead of a shared detector contract.
- Model base classes directly construct datasets and transforms, so model code owns some data-loading concerns.
- `pyimgano.datasets` acts as a broad facade for benchmark datasets, training datasets, converters, synthetic data, and corruption utilities.
- Large entrypoint modules such as `pyimgano/cli.py` and `pyimgano/infer_cli.py` mix parsing, discovery, orchestration, artifact writing, and compatibility logic.

This is why the codebase feels inconsistent: the inconsistency is structural, not just stylistic.

## Goals

1. Define a stable detector runtime contract that all registry models can conform to.
2. Make CLI modules thin adapters over reusable service-layer code.
3. Remove reverse dependencies from `workbench`, `recipes`, and `inference` into CLI modules.
4. Preserve external behavior for the first milestone wherever feasible.
5. Create a migration path for deeper cleanup of datasets and recipes.

## Non-Goals

- Rewriting every algorithm implementation.
- Renaming all public CLI commands.
- Removing legacy compatibility modules such as `pyimgano.detectors` in the first milestone.
- Performing a large breaking reorganization of the entire repository in one pass.

## Target Architecture

### 1. Domain Layer

This layer owns stable behavior contracts and small reusable primitives:

- `pyimgano.models.base_detector`
- `pyimgano.models.protocols` (new)
- `pyimgano.models.deep_contract`
- `pyimgano.features.protocols`
- capability and metadata helpers that read contracts instead of guessing from ad hoc behavior

The key change is that registry models should be describable by a real typed runtime contract, not only by constructor metadata.

### 2. Application / Service Layer

This layer owns orchestration but not CLI parsing:

- `pyimgano.services.model_options` (new)
- `pyimgano.services.inference_service` (new)
- `pyimgano.services.workbench_service` (new)
- `pyimgano.services.discovery_service` (optional follow-up)

The service layer should accept already-parsed Python objects and return structured results. It should not print, parse `argparse.Namespace`, or own command-specific UX.

### 3. Adapter Layer

This layer owns parsing, printing, and compatibility shims:

- `pyimgano.cli`
- `pyimgano.infer_cli`
- `pyimgano.train_cli`
- `pyimgano.robust_cli`
- `pyimgano.demo_cli`
- `pyimgano.detectors`

CLI modules should become thin wrappers over the service layer. Legacy adapters stay here, not in the core runtime.

### 4. Data Subsystem Boundaries

The long-term dataset boundary should split into three concerns while keeping `pyimgano.datasets` as a compatibility facade:

- `pyimgano.datasets.benchmarks`: benchmark dataset loaders and category listing
- `pyimgano.datasets.runtime`: train/eval datasets, transforms, datamodule
- `pyimgano.datasets.converters`: manifest and dataset conversion tools

This split is not the first milestone, but the design should avoid making it harder later.

## Runtime Detector Contract

The most important missing abstraction is a shared runtime detector protocol.

The contract should make the following explicit:

- `fit(inputs, y=None) -> self`
- `decision_function(inputs) -> np.ndarray` with shape `(N,)`
- `predict(inputs) -> np.ndarray` with shape `(N,)`
- stable score convention: higher score means more anomalous
- minimal fitted-state fields such as `decision_scores_` and `threshold_` when applicable
- declared input mode such as `paths`, `numpy`, or `features`
- optional pixel-map support through a normalized batch API
- optional checkpoint/save-load capability

The protocol should support backward compatibility by allowing adapters to normalize older behavior:

- `get_anomaly_map(single)` can be wrapped into batch semantics
- `predict_anomaly_map(batch)` returning lists can be normalized to `(N, H, W)`
- `(scores, maps)` tuple returns from `decision_function` can be normalized centrally

This removes guesswork from `inference.api` and reduces pressure on metadata-based heuristics.

## Refactor Strategy

### Milestone 1: Contract Foundation + Reverse Dependency Removal

This milestone is the recommended starting point because it has the highest leverage and the lowest risk.

Deliverables:

- add `pyimgano.models.protocols`
- add a small runtime normalization adapter used by inference
- move preset/kwargs assembly out of CLI modules into a reusable service module
- update `workbench` and recipes to depend on the service module instead of CLI internals

Expected effect:

- `workbench` no longer imports `pyimgano.cli`
- detector behavior is normalized in one place instead of multiple code paths
- future service extraction becomes straightforward

### Milestone 2: Thin CLI Adapters

After the contract and model-option service exist:

- create `inference_service` and `workbench_service`
- make `infer_cli.py` and `train_cli.py` mostly argument parsing plus output formatting
- keep the CLI output stable while shifting orchestration into testable service functions

### Milestone 3: Dataset and Recipe Boundary Cleanup

Once service-layer seams exist:

- split dataset responsibilities into runtime, benchmarks, and converters
- make recipes depend on service-layer workflows rather than CLI helpers
- keep `pyimgano.datasets` and `pyimgano.recipes` facades for compatibility

## Acceptance Criteria

The architecture refactor should be considered successful when:

- core orchestration modules no longer import CLI helpers
- detector behavior is normalized through a shared runtime contract
- CLI modules become smaller and easier to reason about
- the first migration steps are covered by focused tests
- public CLI behavior remains stable through the first milestone

## Risks and Mitigations

### Risk: Breaking obscure model behavior

Mitigation:

- centralize normalization in an adapter layer instead of forcing every model to change immediately
- add tests that cover tuple-return detectors, list-return pixel maps, and legacy single-image map APIs

### Risk: Refactor churn without user-visible benefit

Mitigation:

- start with reverse dependencies that already block maintainability
- preserve current CLI behavior and algorithm implementations in milestone 1

### Risk: Half-migrated architecture

Mitigation:

- define the destination layers up front
- keep each milestone independently shippable
- add compatibility wrappers rather than partially moving code with no stable seam

## Recommendation

Start with the detector runtime contract and model-option extraction. That gives `pyimgano` a stable center of gravity. Once those seams exist, the CLI and dataset cleanup become straightforward refactors instead of risky rewrites.
