# Study Notes: LVLM / Agentic Anomaly Detection (Not Implemented in v4)

This document is a **study-only** placeholder for large vision-language / agentic
anomaly detection papers (e.g. AgentIAD / AutoIAD).

In the v4 batch we intentionally keep the implementation scope narrow:
- stable, auditable industrial MVP loop
- manifest datasets + pixel maps + defects export
- deep methods only when they can be made **offline-by-default** and dependency-stable

LVLM pipelines typically introduce:
- heavyweight model dependencies
- large checkpoints and frequent remote downloads
- non-trivial inference infrastructure (tool use, retrieval, memory)

Those are valuable research directions, but out of scope for v4 production
defaults.

## Where they might fit (future work)

If we later add LVLM/agentic methods, they should align with these contracts:

- **Inference contract**: `BaseDetector` (scores, thresholding, predict)
- **Pixel contract** (optional): `get_anomaly_map` / `predict_anomaly_map`
- **No-download by default**: explicit checkpoint path required
- **Manifests**: evaluation should run on `manifest.jsonl` splits
- **Auditable outputs**: decisions should export structured provenance (prompts, retrieved refs, tool calls)

## References (study)

- AgentIAD (arXiv 2025)
- AutoIAD (arXiv 2025)

See `docs/INDUSTRIAL_REFERENCE_PROJECTS.md` for the current research index.

