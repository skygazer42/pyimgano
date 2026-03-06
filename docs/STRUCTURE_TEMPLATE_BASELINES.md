# Structure And Template Baselines

These baselines are useful when your inspection station is stable:

- camera pose is mostly fixed
- object layout is repeatable
- anomalies appear as local structural differences, misalignment, or template
  mismatch

They are usually cheaper than deep pixel-map models and can be strong
sanity-checks before you move to PatchCore-style methods.

## When They Work Well

Good fits:

- aligned or nearly aligned products
- stable background and lighting
- defects that change edges, shapes, or local texture
- CPU-first deployments where a simple baseline is valuable

Poor fits:

- large pose variation
- big scale changes
- strong background clutter
- semantic anomalies that require high-level representation

## Model Families

| Model | What it does | Start when |
|---|---|---|
| `ssim_template` | image-to-template SSIM score | image-level aligned sanity check |
| `ssim_template_map` | SSIM anomaly map (`1 - SSIM`) | you need cheap pixel localization |
| `ssim_struct` | structural/edge-aware SSIM baseline | edges matter more than raw intensity |
| `ssim_struct_map` | structural SSIM anomaly map | local structural defects matter |
| `lof_structure` | structural features + LOF | stable geometry with local outliers |
| `isolation_forest_struct` | structural features + Isolation Forest | robust general-purpose structural baseline |
| `kmeans_anomaly` | structural features + centroid distance | cheap clustering-style baseline |
| `dbscan_anomaly` | structural features + density / core distance | cluster-shaped normal data |

## Practical Ordering

For a stable production line, a good escalation path is:

1. `ssim_template_map`
2. `ssim_struct_map`
3. `vision_pixel_mad_map` or `vision_phase_correlation_map`
4. `vision_patchcore` or `vision_softpatch`

That sequence helps separate:

- simple alignment / contrast issues
- structural mismatch
- per-pixel distribution drift
- higher-capacity embedding-based defects

## Failure Modes To Watch

Common false positives:

- small XY drift
- illumination changes
- specular highlights
- background contamination near the ROI border

Useful mitigations:

- apply ROI gating in `pyimgano-infer`
- ignore borders for highly aligned stations
- add light smoothing or morphology in defects export
- move from `ssim_template_map` to `vision_phase_correlation_map` when drift is small but frequent

## Quick CLI Examples

Image-level template sanity check:

```bash
pyimgano-benchmark \
  --dataset custom \
  --root /path/to/dataset \
  --category custom \
  --model ssim_template \
  --device cpu \
  --no-pretrained
```

Pixel-map baseline with defects export:

```bash
pyimgano-infer \
  --model ssim_template_map \
  --train-dir /path/to/train/normal \
  --input /path/to/test \
  --defects \
  --save-jsonl /tmp/results.jsonl \
  --save-masks /tmp/masks
```

## Recommended Mindset

Treat these models as:

- cheap industrial baselines
- alignment/debugging tools
- regression anchors for a stable station

Do not expect them to replace embedding-based methods on datasets with semantic
variation or major viewpoint drift.

See also:

- `docs/ALGORITHM_SELECTION_GUIDE.md`
- `docs/INDUSTRIAL_INFERENCE.md`
- `docs/RECIPES_PIXEL_FIRST_BASELINES.md`
