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

Copy this into your internal wiki or keep it next to your run artifacts:

```markdown
# <Model Name / Checkpoint ID>

## Summary
- Purpose:
- Intended inputs:
- Output contract (image-level / pixel-level / defects):

## Weights / Checkpoint
- Path:
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

