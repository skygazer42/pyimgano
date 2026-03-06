#!/usr/bin/env bash
set -euo pipefail

MANIFEST_PATH="${1:-./data/manifest.jsonl}"
ROOT_DIR="${2:-./data}"
CATEGORY="${3:-bottle}"
OUT_DIR="${4:-./runs/manifest_benchmark_demo}"

pyimgano-benchmark \
  --dataset manifest \
  --root "$ROOT_DIR" \
  --manifest-path "$MANIFEST_PATH" \
  --category "$CATEGORY" \
  --suite industrial-v4 \
  --suite-sweep industrial-feature-small \
  --suite-sweep-max-variants 1 \
  --manifest-test-normal-fraction 0.2 \
  --manifest-split-seed 0 \
  --device cpu \
  --no-pretrained \
  --output-dir "$OUT_DIR"
