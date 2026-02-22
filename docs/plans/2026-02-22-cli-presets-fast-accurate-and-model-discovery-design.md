# CLI Presets — `industrial-fast` / `industrial-accurate` + Model Discovery — Design

**Date:** 2026-02-22

## Goal

1. Expand `pyimgano-benchmark` presets beyond `industrial-balanced` with:
   - `industrial-fast`: speed-oriented defaults (quick iteration / “first run”).
   - `industrial-accurate`: accuracy-oriented defaults (higher resource use by default).
2. Improve CLI discoverability without requiring dataset args by adding:
   - `--list-models`: list available registry names and exit.
   - `--model-info <name>`: show tags/metadata/signature/accepted kwargs and exit.
   - `--json` (for discovery modes): emit JSON instead of text.

## Non-Goals

- No new models are implemented in this change.
- No argparse subcommands (keep backwards compatible CLI shape).
- No tag-based filtering / searching of models (can be added later).

## CLI UX

### Discovery modes

- `pyimgano-benchmark --list-models`:
  - Default output: one model name per line, sorted.
  - `--json`: JSON array of model names.

- `pyimgano-benchmark --model-info vision_patchcore`:
  - Default output: human-readable text showing name/tags/metadata/signature and kwarg acceptance.
  - `--json`: JSON object with the same data.

### Run mode (existing behavior)

If neither `--list-models` nor `--model-info` are provided, the CLI must run a benchmark.
In that case `--dataset`, `--root`, and `--category` are required (validated at runtime).

## Preset Defaults (High-Level)

All presets follow stable precedence:

1) `--model-kwargs` (highest)  
2) `--preset` defaults  
3) auto kwargs inferred from CLI (`device`, `contamination`, `pretrained`) (lowest)

The preset applies only to models known to work with `pyimgano-benchmark`’s
path-based workflow (`fit(list[str])`, `decision_function(list[str])`).

### Model coverage (all three presets)

- `vision_patchcore`
- `vision_padim`
- `vision_spade`
- `vision_anomalydino`
- `vision_softpatch`
- `vision_simplenet`
- `vision_fastflow`
- `vision_cflow`
- `vision_stfpm`
- `vision_reverse_distillation` (alias: `vision_reverse_dist`)
- `vision_draem`

### Parameter intent

- `industrial-fast`: reduce compute and memory (smaller coreset / smaller image_size / fewer epochs).
- `industrial-accurate`: increase compute and memory where it tends to help (larger coreset / larger image_size / more epochs).
- KNN backends: prefer FAISS if importable, else fall back to sklearn.

## Testing Strategy

- Unit tests for `_resolve_preset_kwargs(preset, model_name)` for fast/accurate on a few representative models.
- Parser acceptance tests for new `--preset` values.
- CLI smoke tests:
  - `main(["--list-models"])` prints model names and exits 0.
  - `main(["--model-info","vision_patchcore"])` prints text and exits 0.
  - Both also validate `--json` mode emits valid JSON.
  - Mutual exclusion (`--list-models` + `--model-info`) returns non-zero.

## Docs

- Update `README.md` and `docs/QUICKSTART.md`:
  - Mention `industrial-fast` / `industrial-accurate` alongside `industrial-balanced`.
  - Mention `--list-models` / `--model-info` usage (text default; `--json` optional).

