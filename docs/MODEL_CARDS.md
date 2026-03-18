# Model Cards (Policy + Template)

This project intentionally keeps the wheel lightweight:

- no large weights/checkpoints shipped inside `pyimgano`
- no implicit downloads by default (CLI defaults to `--no-pretrained`)

For production use, you usually need a **model card** for any deployed model
or checkpoint.

## Why model cards matter here

In industrial visual anomaly detection, the highest risk is not "the code runs",
but "the model drifts silently or is used outside its intended domain".

A model card makes these things explicit:

- what data distribution the model expects
- how the threshold was calibrated
- what the known failure modes are (false positives, lighting sensitivity, alignment, ROI)
- where the weights came from and whether they are reproducible/auditable

## Minimal model card template

You can either copy the template below into your internal wiki, or emit a JSON
starter file directly:

```bash
pyimgano-weights template model-card > ./model_card.json
```

There is also a checked-in example asset at `examples/configs/example_model_card.json`.

Keep the model card next to your run artifacts or deploy bundle:

```markdown
# <Model Name / Checkpoint ID>

## Summary
- Purpose:
- Intended inputs:
- Output contract (image-level / pixel-level / defects):

## Weights / Checkpoint
- Path:
- Manifest entry:
- SHA256:
- Source:
- License:

## Training / Calibration
- Dataset:
- Split policy:
- Preprocessing:
- Threshold strategy:

## Evaluation
- Datasets evaluated on:
- Key metrics:
- Known weak spots:

## Deployment Notes
- Required runtime (cpu/cuda/onnx/openvino):
- Cache locations (TORCH_HOME/HF_HOME/XDG_CACHE_HOME):
- Expected throughput:

## Limitations
- Out-of-distribution failure modes:
- Sensitivity (lighting, alignment, ROI, camera changes):
```

## Weights inventory (manifest)

If you manage multiple checkpoints, keep a local manifest JSON and validate it:

- See: `docs/WEIGHTS.md`
- CLI: `pyimgano-weights validate ./weights_manifest.json --check-hashes --json`
- JSON template: `pyimgano-weights template manifest > ./weights_manifest.json`
- Model card validation: `pyimgano-weights validate-model-card ./model_card.json --json`

If you want the model card check to verify the referenced checkpoint asset too:

```bash
pyimgano-weights \
  validate-model-card ./model_card.json \
  --check-files \
  --check-hashes \
  --json
```

That mode resolves `weights.path` relative to the model card location by default
and reports the resolved asset path in the JSON output.

If you also keep a local weights manifest, you can cross-check the model card
against it:

```bash
pyimgano-weights \
  validate-model-card ./model_card.json \
  --manifest ./weights_manifest.json \
  --check-files \
  --check-hashes \
  --json
```

For the most stable linkage, add `weights.manifest_entry` to the model card so
the validator can bind to a named manifest entry instead of inferring by path or
sha256.
