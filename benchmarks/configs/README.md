# Benchmark Config Presets

These JSON files are **reproducibility-oriented presets** for `pyimgano-benchmark`.

They are intended to be:

- explicit about suite/sweep/device/offline safety flags
- easy to copy and customize
- versionable (so benchmark runs can be compared across time)

Usage:

```bash
pyimgano-benchmark --list-starter-configs
pyimgano-benchmark --starter-config-info official_mvtec_industrial_v4_cpu_offline.json --json
pyimgano-benchmark --list-official-configs
pyimgano-benchmark --official-config-info official_mvtec_industrial_v4_cpu_offline.json --json
pyimgano-benchmark --config benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json
pyimgano-benchmark --config official_mvtec_industrial_v4_cpu_offline.json
```

Notes:

- Paths in these configs are placeholders; edit `root`/`output_dir` before use.
- `--list-starter-configs` is the fastest discovery surface when you want a CPU-friendly first benchmark.
- `--config` accepts either the full path or the bare official filename for built-in presets.
- CLI flags always override config values because config argv is applied first.
