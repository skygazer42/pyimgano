# Design: `pyimgano-doctor --require-extras` + `pyimgano-demo --infer-defects`

Date: 2026-03-05

## Goals

- Add a **CI/deploy-friendly gate** to `pyimgano-doctor` so pipelines can fail fast when required extras are missing.
- Upgrade `pyimgano-demo` to support a **one-command closed loop** that produces *inference + defects artifacts* without manually running `pyimgano-infer`.
- Keep the default install lightweight and preserve strict optional dependency boundaries (no heavy imports unless the user explicitly opts in).

## Non-goals

- Selecting the “best baseline from suite results” automatically for inference (this is higher-level orchestration and can be added later).
- Adding new deep baselines or changing suite composition.

## CLI UX

### `pyimgano-doctor --require-extras`

- New flag: `--require-extras EXTRA[,EXTRA]` (repeatable and comma-separated).
- Behavior:
  - Always prints JSON when `--json` is used.
  - Exits with code `1` if any requested extras are missing.
  - Includes an actionable `pip install 'pyimgano[extra1,extra2]'` hint in the payload / text output.

### `pyimgano-demo --infer-defects`

- New flag: `--infer-defects` (optional).
- Behavior:
  - Requires `--save-run` because artifacts must be written somewhere.
  - After completing the suite run, performs an inference pass over the demo dataset and writes:
    - `<suite_run_dir>/infer/results.jsonl`
    - `<suite_run_dir>/infer/masks/`
  - Uses a **core-only** model preset by default (`industrial-template-ncc-map`) plus the built-in industrial defects preset (`industrial-defects-fp40`).
  - Stays offline-safe by explicitly passing `--no-pretrained` unless the demo itself requests `--pretrained`.

## Implementation notes

- Extras availability check is import-light: it uses `importlib.util.find_spec` against known extra root modules (no importing torch/onnx/openvino at check time).
- Demo inference is implemented by calling `pyimgano.infer_cli.main([...])` programmatically with explicit output paths.

## Testing

- Add a unit test ensuring missing `--require-extras` causes a non-zero exit and returns a JSON payload that includes `require_extras.missing`.
- Add a smoke test ensuring `pyimgano-demo --infer-defects` produces the expected `infer/` artifacts.
