# Model Metadata Contract

This page defines the structured metadata contract used by PyImgAno model discovery,
model info payloads, and metadata audits.

The contract has two kinds of fields:

- Registry metadata fields: explicitly stored in `@register_model(..., metadata={...})`
- Derived fields: computed from tags or constructor capabilities

## Fields

### `paper`

- Source: registry metadata
- Requirement: recommended
- Meaning: canonical paper title or upstream algorithm title for the model entry
- Notes:
  - Use the paper title when the implementation maps cleanly to a publication.
  - For wrappers, use the wrapped algorithm title rather than the wrapper class name.

### `year`

- Source: registry metadata
- Requirement: recommended
- Meaning: publication year for the paper or algorithm family backing the model entry
- Validation:
  - integer year
  - between `1900` and the current calendar year

### `family`

- Source: derived from registry tags
- Requirement: required
- Meaning: one or more curated algorithm families resolved from the tag taxonomy in discovery
- Examples:
  - `patchcore`
  - `distillation`
  - `density`

### `type`

- Source: derived from registry tags
- Requirement: required
- Meaning: one or more high-level model types resolved from the tag taxonomy in discovery
- Examples:
  - `memory-bank`
  - `flow-based`
  - `density-estimation`
  - `neighbor-based`

### `supervision`

- Source: registry metadata or derived from explicit supervision tags
- Requirement: recommended
- Meaning: training supervision regime for discovery and recommendation layers
- Allowed values:
  - `unsupervised`
  - `self-supervised`
  - `weakly-supervised`
  - `supervised`
  - `few-shot`
  - `zero-shot`
  - `one-class`

### `supports_pixel_map`

- Source: derived from constructor capabilities
- Requirement: required
- Meaning: whether the model exposes pixel-level anomaly maps

### `requires_checkpoint`

- Source: derived from constructor capabilities and metadata
- Requirement: required
- Meaning: whether the model needs an external checkpoint or saved artifact to run

### `weights_source`

- Source: registry metadata
- Requirement: conditional
- Required when:
  - `requires_checkpoint == true`
- Meaning: where the recommended checkpoint or weights come from
- Examples:
  - `official`
  - `upstream-project`
  - `local-training-only`

## Audit Entry Points

### CLI

```bash
pyim --list metadata-contract --json
pyim --audit-metadata --json
```

### Tool Script

```bash
python tools/audit_registry.py --metadata-contract --json
```

## Current Policy

- The audit distinguishes required issues from recommended issues.
- Required issues are structural gaps that break discovery or deployment reasoning.
- Recommended issues are metadata quality gaps that should be cleaned up over time.
- Current repo status is expected to contain many recommended gaps while the contract rollout is in progress.
