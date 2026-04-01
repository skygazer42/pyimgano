# Starter Paths

Use this page when you already know your goal and want one exact command chain.

If you want the umbrella entrypoint first, run:

```bash
python -m pyimgano --help
```

## First 10 Minutes

Smallest offline-safe path from environment check to benchmarked inference artifacts.

```bash
pyimgano-doctor --profile first-run --json
pyimgano-demo --smoke --dataset-root ./_demo_custom_dataset --output-dir ./_demo_suite_run --summary-json /tmp/pyimgano_demo_summary.json --emit-next-steps --no-pretrained
pyimgano-doctor --profile benchmark --dataset-target ./_demo_custom_dataset --json
pyimgano-benchmark --dataset custom --root ./_demo_custom_dataset --suite industrial-ci --resize 32 32 --limit-train 2 --limit-test 2 --no-pretrained --save-run --output-dir ./_demo_benchmark_run --suite-export csv
pyimgano-infer --model-preset industrial-template-ncc-map --train-dir ./_demo_custom_dataset/train/normal --input ./_demo_custom_dataset/test --save-jsonl ./_demo_results.jsonl
pyimgano runs quality ./_demo_benchmark_run --json
```

## First Benchmark Run

Use this when you want starter-config discovery plus a benchmark/publication-oriented loop.

```bash
pyimgano-doctor --profile benchmark --dataset-target /path/to/dataset_root --json
pyimgano benchmark --list-starter-configs
pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json
pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json
pyimgano runs publication /path/to/suite_export --json
```

## Deployment Check

Use this after a train/export run when you need to validate deploy artifacts and gate acceptance.

```bash
pyimgano-doctor --profile deploy --run-dir runs/<run_dir> --json
pyimgano validate-infer-config runs/<run_dir>/deploy_bundle/infer_config.json
pyimgano bundle validate runs/<run_dir>/deploy_bundle --json
pyimgano runs acceptance runs/<run_dir> --require-status audited --check-bundle-hashes --json
```

## Publication Gate

Use this when you have a suite export and need to verify publication trust signals.

```bash
pyimgano-doctor --profile publish --publication-target /path/to/suite_export --json
pyimgano runs acceptance /path/to/suite_export --json
pyimgano runs publication /path/to/suite_export --json
```
