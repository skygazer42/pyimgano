# Benchmark Getting Started

Start with the lightest built-in benchmark discovery surface:

```bash
pyimgano-doctor --recommend-extras --for-command benchmark --json
pyimgano benchmark --list-starter-configs
pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json
```

The starter config metadata includes `optional_extras`, `optional_baseline_count`, and an install hint so you can tell which optional backends broaden the suite before you run it.

Starter configs are intended to be:

- CPU-friendly
- Offline-safe by default
- Easy to copy and adapt into a project-specific benchmark run

Typical flow:

```bash
pyimgano benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json
pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json
```

If you need the full reproducibility-oriented preset list instead, use:

```bash
pyimgano benchmark --list-official-configs
```
