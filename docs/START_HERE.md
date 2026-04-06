# Start Here

If this is your first contact with `pyimgano`, start with the deployment smoke path:

```bash
pyimgano-doctor --profile deploy-smoke --json
pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained
pyimgano-train --config examples/configs/deploy_smoke_custom_cpu.json --root ./_demo_custom_dataset --export-infer-config --export-deploy-bundle
pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json
pyimgano bundle validate runs/<run_dir>/deploy_bundle --json
pyimgano runs quality runs/<run_dir> --json
```

If you want the broader benchmark-and-infer starter loop, follow this path:

```bash
pyimgano-doctor --profile first-run --json
pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained
pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json
pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset --suite industrial-ci --resize 32 32 --limit-train 2 --limit-test 2 --no-pretrained --save-run --output-dir ./_demo_benchmark_run --suite-export csv
pyimgano-infer --model-preset industrial-template-ncc-map --train-dir ./_demo_custom_dataset/train/normal --input ./_demo_custom_dataset/test --save-jsonl ./_demo_results.jsonl
pyimgano runs quality ./_demo_benchmark_run --json
```

What this does:

- `pyimgano-doctor --profile deploy-smoke` validates the smallest offline-safe path that reaches an exported deploy bundle.
- `pyimgano-train --config examples/configs/deploy_smoke_custom_cpu.json ...` creates a CPU-safe run, exports `infer_config.json`, and assembles a deploy bundle from the same tiny dataset.
- `pyimgano validate-infer-config ...` and `pyimgano bundle validate ...` verify the exported deployment payload before you look at bigger benchmarks.
- `pyimgano-doctor --profile first-run` validates the base offline-safe starter path and prints the guided command chain.
- `pyimgano-demo --smoke` creates a tiny offline-safe dataset plus a bounded suite run you can inspect immediately.
- `pyimgano-doctor --profile benchmark --dataset-target ...` checks whether that dataset is benchmark-ready and reminds you which artifacts to expect.
- `pyimgano-benchmark ... --save-run` writes a real benchmark run directory you can gate with `pyimgano runs quality`.
- `pyimgano-infer ...` turns the same demo dataset into a concrete JSONL inference artifact.

Recommended next steps:

- Root CLI discovery: `pyimgano --help`
- Python module entrypoint: `python -m pyimgano --help`
- Export extras readiness: `pyimgano-doctor --recommend-extras --for-command export-onnx --json`
- Train extras readiness: `pyimgano-doctor --recommend-extras --for-command train --json`
- Model discovery: `pyim --list models --objective latency --selection-profile cpu-screening --topk 5`
- Starter benchmark discovery: `pyimgano benchmark --list-starter-configs`
- Deploy-oriented validation: `pyimgano --help`

## Guided Workflow

- `Discover`: `pyim --list models --objective latency --selection-profile cpu-screening --topk 5`
- `Benchmark`: `pyimgano-doctor --recommend-extras --for-command benchmark --json`
- `Train`: `pyimgano-doctor --recommend-extras --for-command train --json`
- `Export`: `pyimgano-doctor --recommend-extras --for-command export-onnx --json`
- `Infer`: `pyimgano-doctor --recommend-extras --for-command infer --json`
- `Validate`: `pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json`
- `Gate`: `pyimgano-doctor --recommend-extras --for-command runs --json`
