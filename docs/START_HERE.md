# Start Here

If this is your first contact with `pyimgano`, follow this path:

```bash
pyimgano-doctor
pyimgano-doctor --recommend-extras --for-command export-onnx --json
pyimgano-doctor --recommend-extras --for-command benchmark --json
pyimgano-doctor --recommend-extras --for-command train --json
pyimgano-doctor --recommend-extras --for-command infer --json
pyimgano-doctor --recommend-extras --for-command runs --json
pyimgano-demo --smoke --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps
```

What this does:

- `pyimgano-doctor` shows the current environment and optional extras status.
- `--recommend-extras` turns a command or model into a copy-pasteable install hint.
- `--for-command benchmark` surfaces starter benchmark configs, optional extras, and how many starter baselines are gated behind those extras.
- `pyimgano-demo --smoke` creates a tiny offline-safe dataset, runs a bounded suite, and prints the next commands to try.

Recommended next steps:

- Model discovery: `pyim --list models --objective latency --selection-profile cpu-screening --topk 5`
- Starter benchmark discovery: `pyimgano benchmark --list-starter-configs`
- Deploy-oriented validation: `python -m pyimgano --help`

## Guided Workflow

- `Discover`: `pyim --list models --objective latency --selection-profile cpu-screening --topk 5`
- `Benchmark`: `pyimgano-doctor --recommend-extras --for-command benchmark --json`
- `Train`: `pyimgano-doctor --recommend-extras --for-command train --json`
- `Export`: `pyimgano-doctor --recommend-extras --for-command export-onnx --json`
- `Infer`: `pyimgano-doctor --recommend-extras --for-command infer --json`
- `Validate`: `pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json`
- `Gate`: `pyimgano-doctor --recommend-extras --for-command runs --json`
