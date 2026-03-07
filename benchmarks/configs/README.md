# Benchmark Config Presets

These JSON files are **reproducibility-oriented presets** for `pyimgano-benchmark`.

They are intended to be:

- explicit about suite/sweep/device/offline safety flags
- easy to copy and customize
- versionable (so benchmark runs can be compared across time)

Usage:

```bash
pyimgano-benchmark --config benchmarks/configs/official_mvtec_industrial_v4_cpu_offline.json
```

Notes:

- Paths in these configs are placeholders; edit `root`/`output_dir` before use.
- CLI flags always override config values because config argv is applied first.

