# Industrial Workbench

`pyimgano` includes a **recipe-driven workbench** layer for industrial anomaly
detection. The goal is to standardize the “industrial loop” and make runs
comparable via artifacts (reports, per-image JSONL, optional maps/checkpoints).

Primary entrypoint:

```bash
pyimgano-train --config cfg.json
```

See also:

- `docs/RECIPES.md` (config schema + builtin recipes)
- `docs/CLI_REFERENCE.md` (CLI flags)
- `docs/MANIFEST_DATASET.md` (JSONL manifest dataset)

---

## Quickstart (end-to-end)

1) Start from an example config:

```
examples/configs/industrial_adapt_fast.json
examples/configs/industrial_adapt_maps_tiling.json
```

2) Edit at least:

- `dataset.root`
- (manifest only) `dataset.manifest_path`
- `output.output_dir` (optional; omit to write under `runs/`)
- `model.device` (`cpu` or `cuda`)

3) Validate the config without running:

```bash
pyimgano-train --config cfg.json --dry-run
```

4) Run the recipe and export an inference helper artifact:

```bash
pyimgano-train --config cfg.json --export-infer-config
```

This writes a run directory containing:

- `report.json` / `config.json` / `environment.json`
- `categories/<cat>/per_image.jsonl`
- `artifacts/infer_config.json`
- optional: `artifacts/maps/*.npy`
- optional: `checkpoints/<cat>/...`

5) Reuse the run for inference:

```bash
pyimgano-infer --from-run /path/to/run_dir --input /path/to/images --save-jsonl out.jsonl
```

If the run contains multiple categories, select one:

```bash
pyimgano-infer --from-run /path/to/run_dir --from-run-category bottle --input /path/to/images
```

---

## Notes

- `pyimgano-infer --from-run` is **best-effort**: it loads model settings, applies `threshold_`,
  and loads checkpoints when the detector supports it.
- For high-resolution inference, you can still use `pyimgano-infer` tiling flags (see
  `docs/INDUSTRIAL_INFERENCE.md`).

## Manifest datasets (JSONL)

For real industrial repos (multi-category, mixed sources, group-aware splits),
prefer the manifest dataset:

- `dataset.name="manifest"`
- `dataset.manifest_path="/path/to/manifest.jsonl"`
- optional: `dataset.split_policy` and `group_id` fields inside the manifest

See: `docs/MANIFEST_DATASET.md`
